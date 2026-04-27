"""
Inference engine implementations.
"""
import numpy as np
import logging
from typing import Optional
from interfaces import InferenceEngineInterface
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_openpi_policy(checkpoint_path: Path):
    """Load an openpi policy from a JAX checkpoint using create_trained_policy."""
    import sys
    src_path = str(checkpoint_path.parent.parent.parent.parent / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    # Infer config name from path: checkpoints/<config_name>/<exp_name>/<step>
    config_name = checkpoint_path.parts[-3]
    train_config = _config.get_config(config_name)
    return _policy_config.create_trained_policy(train_config, checkpoint_path)


def _extract_action(result: dict) -> np.ndarray:
    """Extract [pitch, yaw, grip, ...] from policy.infer() result."""
    actions = result["actions"]  # (action_horizon, action_dim)
    action = np.asarray(actions[0], dtype=np.float32)  # first timestep
    if len(action) < 6:
        action = np.pad(action, (0, 6 - len(action)))
    return action[:6]


class JAXCheckpointInference(InferenceEngineInterface):
    """
    JAX checkpoint inference (openpi LeRobot flax format).
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.policy = None
        self._is_loaded = False

    def load_model(self, checkpoint_path: str) -> bool:
        try:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                return False
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            self.policy = _load_openpi_policy(checkpoint_path)
            self._is_loaded = True
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def infer(
        self,
        image: np.ndarray,
        instruction: str,
        prev_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not self.is_loaded:
            logger.error("Model not loaded. Call load_model() first.")
            return np.array([0.5, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        try:
            if prev_state is not None:
                state = np.asarray(prev_state, dtype=np.float32).reshape(-1)
            else:
                state = np.array([0.5, -1.0, 0.0], dtype=np.float32)
            # Pad state to 8 dims (model expects 8)
            if len(state) < 8:
                state = np.pad(state, (0, 8 - len(state)))
            obs = {
                "observation/image": image,
                "observation/state": state,
                "prompt": instruction,
            }
            result = self.policy.infer(obs)
            return _extract_action(result)
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return np.array([0.5, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def unload_model(self):
        self.policy = None
        self._is_loaded = False
        logger.info("Model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded


class JAXCheckpointWithEgomotionInference(InferenceEngineInterface):
    """
    JAX checkpoint inference with egomotion encoder support.
    For models trained with use_egomotion_z=True.

    Computes ego_motion_z in real time using VideoEncoder from rolling frame buffer.
    Falls back to zeros(256) if VideoEncoder is not available.
    """

    _EGOMOTION_CKPT = str(Path(__file__).parent.parent / "EgoMotion" / "checkpoints" / "best_encoder_frame_diff.pt")
    _EGOMOTION_SPAN = 16
    _EGOMOTION_DIM = 256

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.policy = None
        self._is_loaded = False
        self._ego_encoder = None
        self._ego_device = "cpu"
        self._frame_buffer: list = []  # rolling uint8 HxWx3 frames (224x224)

    def load_model(self, checkpoint_path: str) -> bool:
        try:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                return False
            logger.info(f"Loading checkpoint with egomotion support from {checkpoint_path}")
            self.policy = _load_openpi_policy(checkpoint_path)
            self._try_load_egomotion_encoder()
            self._is_loaded = True
            logger.info("Model with egomotion support loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _try_load_egomotion_encoder(self):
        """Try to load VideoEncoder; log warning if unavailable."""
        try:
            import sys
            import torch
            ego_src = str(Path(__file__).parent.parent / "EgoMotion")
            if ego_src not in sys.path:
                sys.path.insert(0, ego_src)
            from src.models.encoder import EncoderConfig, VideoEncoder
            cfg = EncoderConfig(
                backbone="resnet18", latent_dim=256, temporal_layers=4,
                temporal_heads=8, dropout=0.0, pretrained=False,
                freeze_backbone=False, use_motion_branch=True, aggregate_last_k=4,
            )
            encoder = VideoEncoder(cfg)
            ckpt = torch.load(self._EGOMOTION_CKPT, map_location="cpu", weights_only=False)
            encoder.load_state_dict(ckpt["encoder_state_dict"])
            encoder.eval()
            self._ego_encoder = encoder
            self._ego_device = "cpu"
            logger.info("VideoEncoder for egomotion loaded successfully")
        except Exception as e:
            logger.warning(f"VideoEncoder not available, ego_motion_z will be zeros: {e}")
            self._ego_encoder = None

    def _compute_egomotion_z(self, image: np.ndarray) -> np.ndarray:
        """Compute 256-dim ego_motion_z from rolling frame buffer."""
        import cv2
        frame = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        self._frame_buffer.append(frame)
        if len(self._frame_buffer) > self._EGOMOTION_SPAN:
            self._frame_buffer.pop(0)

        if self._ego_encoder is None:
            return np.zeros(self._EGOMOTION_DIM, dtype=np.float32)

        try:
            import torch
            n = len(self._frame_buffer)
            window = np.stack(self._frame_buffer, axis=0)  # (n, 224, 224, 3)
            if n < self._EGOMOTION_SPAN:
                pad = np.repeat(window[:1], self._EGOMOTION_SPAN - n, axis=0)
                window = np.concatenate([pad, window], axis=0)  # (16, 224, 224, 3)
            frames_t = torch.from_numpy(window).permute(0, 3, 1, 2).unsqueeze(0)  # (1, 16, 3, 224, 224)
            with torch.no_grad():
                z = self._ego_encoder(frames_t)  # (1, 256)
            return z[0].numpy().astype(np.float32)
        except Exception as e:
            logger.warning(f"egomotion computation failed: {e}")
            return np.zeros(self._EGOMOTION_DIM, dtype=np.float32)

    def infer(
        self,
        image: np.ndarray,
        instruction: str,
        prev_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not self.is_loaded:
            logger.error("Model not loaded. Call load_model() first.")
            return np.array([0.5, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        try:
            if prev_state is not None:
                state = np.asarray(prev_state, dtype=np.float32).reshape(-1)
            else:
                state = np.array([0.5, -1.0, 0.0], dtype=np.float32)
            if len(state) < 8:
                state = np.pad(state, (0, 8 - len(state)))
            ego_z = self._compute_egomotion_z(image)
            obs = {
                "observation/image": image,
                "observation/state": state,
                "prompt": instruction,
                "ego_motion_z": ego_z,
            }
            result = self.policy.infer(obs)
            return _extract_action(result)
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return np.array([0.5, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def unload_model(self):
        self.policy = None
        self._is_loaded = False
        self._frame_buffer = []
        logger.info("Model with egomotion support unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded


class VLASHInference(InferenceEngineInterface):
    """
    VLASH model inference (PyTorch + LoRA adapters).
    Loads safetensors checkpoints with LoRA weights.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None
        self._is_loaded = False
    
    def load_model(self, checkpoint_path: str) -> bool:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
            from peft import AutoPeftModelForCausalLM
            
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                return False
            
            logger.info(f"Loading VLASH checkpoint from {checkpoint_path}")
            
            # Load model with LoRA adapters merged
            pretrained_model_path = checkpoint_path / "pretrained_model"
            if not pretrained_model_path.exists():
                logger.error(f"pretrained_model not found in {checkpoint_path}")
                return False
            
            # Try to load with LoRA adapters
            try:
                self.model = AutoPeftModelForCausalLM.from_pretrained(
                    str(pretrained_model_path),
                    torch_dtype=torch.bfloat16,
                    device_map=self.device,
                )
                logger.info("Loaded model with LoRA adapters")
            except Exception as e:
                logger.warning(f"Failed to load with LoRA: {e}, trying standard load")
                self.model = AutoModel.from_pretrained(
                    str(pretrained_model_path),
                    torch_dtype=torch.bfloat16,
                    device_map=self.device,
                )
            
            self.model.eval()
            self._is_loaded = True
            logger.info("VLASH model loaded successfully")
            return True
        
        except ImportError as e:
            logger.error(f"Missing dependency: {e}. Install with: pip install vlash")
            return False
        except Exception as e:
            logger.error(f"Failed to load VLASH model: {e}")
            return False
    
    def infer(
        self,
        image: np.ndarray,
        instruction: str,
        prev_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not self.is_loaded:
            logger.error("Model not loaded. Call load_model() first.")
            return np.array([0.5, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        try:
            import torch
            
            # Prepare inputs (VLASH-specific preprocessing)
            # This should match the training preprocessing
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            image_tensor = image_tensor.to(self.device)
            
            # Prepare state if available
            if prev_state is not None:
                state_input = torch.from_numpy(prev_state).unsqueeze(0).float().to(self.device)
            else:
                state_input = torch.tensor([[[0.5, -1.0, 0.0]]], dtype=torch.float32).to(self.device)
            
            # Run inference
            with torch.no_grad():
                # VLASH model forward pass
                # Adjust based on actual VLASH model API
                outputs = self.model(
                    pixel_values=image_tensor,
                    state=state_input,
                    task_description=instruction,
                )
            
            # Extract action from outputs
            if hasattr(outputs, "action"):
                action = outputs.action
            elif isinstance(outputs, dict) and "action" in outputs:
                action = outputs["action"]
            elif isinstance(outputs, torch.Tensor):
                action = outputs
            else:
                logger.warning(f"Unexpected output format: {type(outputs)}")
                action = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            
            # Convert to numpy
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            else:
                action = np.array(action)
            
            # Ensure shape and format
            if action.ndim > 1:
                action = action.squeeze()
            if len(action) < 6:
                action = np.pad(action, (0, 6 - len(action)), mode="constant")
            
            return action.astype(np.float32)
        
        except Exception as e:
            logger.error(f"VLASH inference failed: {e}")
            return np.array([0.5, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    def unload_model(self):
        if self.model is not None:
            import torch
            del self.model
            torch.cuda.empty_cache()
        self._is_loaded = False
        logger.info("VLASH model unloaded")
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded


class DummyInference(InferenceEngineInterface):
    """
    Dummy inference for testing - returns fixed action.
    """
    
    def __init__(self):
        self._is_loaded = False
    
    def load_model(self, checkpoint_path: str) -> bool:
        logger.info("Dummy model 'loaded' (no actual model)")
        self._is_loaded = True
        return True
    
    def infer(
        self,
        image: np.ndarray,
        instruction: str,
        prev_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # Return fixed action for testing
        logger.debug(f"Dummy inference: instruction='{instruction}'")
        return np.array([0.5, -1.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)
    
    def unload_model(self):
        self._is_loaded = False
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded


def create_inference_engine(config: dict) -> InferenceEngineInterface:
    """Factory function to create inference engine based on config."""
    engine_type = config.get("type", "dummy")
    device = config.get("device", "cuda")
    
    if engine_type == "jax_checkpoint":
        engine = JAXCheckpointInference(device=device)
    elif engine_type == "vlash_checkpoint":
        engine = VLASHInference(device=device)
    elif engine_type == "dummy":
        engine = DummyInference()
    elif engine_type == "jax_checkpoint_with_egomotion":
        engine = JAXCheckpointWithEgomotionInference(device=device)
    else:
        logger.warning(f"Unknown engine type: {engine_type}, using dummy")
        engine = DummyInference()
    
    checkpoint_path = config.get("checkpoint_path")
    if engine_type != "dummy" and checkpoint_path:
        engine.load_model(checkpoint_path)
    
    return engine
