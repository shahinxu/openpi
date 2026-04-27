"""
Inference engine implementations.
"""
import numpy as np
import logging
from typing import Optional
from interfaces import InferenceEngineInterface
from pathlib import Path

logger = logging.getLogger(__name__)


class JAXCheckpointInference(InferenceEngineInterface):
    """
    JAX checkpoint inference (for LeRobot models).
    Loads from flax checkpoint format.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None
        self.config = None
        self._is_loaded = False
    
    def load_model(self, checkpoint_path: str) -> bool:
        try:
            from lerobot.common.policies.factory import make_policy
            
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                return False
            
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            
            # Load policy using LeRobot factory
            self.model = make_policy(
                policy_path=str(checkpoint_path),
                device=self.device,
                use_raw_action=False,  # Use normalized actions
            )
            
            self._is_loaded = True
            logger.info("Model loaded successfully")
            return True
        
        except ImportError:
            logger.error("LeRobot library not installed. Install with: pip install lerobot")
            return False
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
            # Return dummy action
            return np.array([0.5, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        try:
            import torch
            from PIL import Image
            
            # Prepare input
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            image_tensor = image_tensor.to(self.device)
            
            # Prepare state (if available)
            if prev_state is not None:
                state_tensor = torch.from_numpy(prev_state).unsqueeze(0).float().to(self.device)
            else:
                # Default initial state
                state_tensor = torch.tensor([[[0.5, -1.0, 0.0]]], dtype=torch.float32).to(self.device)
            
            # Run inference
            with torch.no_grad():
                # Model input format depends on specific architecture
                # Adjust based on your model's expected input
                action_output = self.model.select_action(
                    {
                        "observation": {
                            "pixels": image_tensor,  # (B, C, H, W)
                        },
                        "state": state_tensor,  # (B, 3)
                        "instruction": instruction,  # Text instruction
                    }
                )
            
            # Extract action (adjust based on model output format)
            if isinstance(action_output, dict):
                action = action_output.get("action", np.zeros(6, dtype=np.float32))
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
            elif isinstance(action_output, torch.Tensor):
                action = action_output.cpu().numpy()
            else:
                action = np.array(action_output)
            
            # Ensure output is [pitch, yaw, grip, ...]
            if action.ndim > 1:
                action = action.squeeze()
            if len(action) < 6:
                action = np.pad(action, (0, 6 - len(action)), mode="constant")
            
            return action.astype(np.float32)
        
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return np.array([0.5, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    def unload_model(self):
        self.model = None
        self._is_loaded = False
        logger.info("Model unloaded")
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded


class JAXCheckpointWithEgomotionInference(InferenceEngineInterface):
    """
    JAX checkpoint inference with egomotion encoder support.
    For models trained with use_egomotion_z=True.
    
    Computes egomotion_z in real time from state delta.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None
        self.config = None
        self._is_loaded = False
        self.last_state: Optional[np.ndarray] = None
    
    def load_model(self, checkpoint_path: str) -> bool:
        try:
            from lerobot.common.policies.factory import make_policy
            
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                return False
            
            logger.info(f"Loading checkpoint with egomotion support from {checkpoint_path}")
            
            # Load policy using LeRobot factory
            self.model = make_policy(
                policy_path=str(checkpoint_path),
                device=self.device,
                use_raw_action=False,  # Use normalized actions
            )
            
            self._is_loaded = True
            logger.info("Model with egomotion support loaded successfully")
            return True
        
        except ImportError:
            logger.error("LeRobot library not installed. Install with: pip install lerobot")
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def compute_egomotion_from_state(
        self,
        prev_state: Optional[np.ndarray],
        curr_state: np.ndarray,
    ) -> np.ndarray:
        """Compute egomotion as state delta."""
        if prev_state is None:
            # Default egomotion (zero delta)
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return (curr_state - prev_state).astype(np.float32)
    
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
            
            # Prepare input
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            image_tensor = image_tensor.to(self.device)
            
            # Prepare state
            if prev_state is not None:
                curr_state = np.asarray(prev_state, dtype=np.float32).reshape(-1)[:3]
            else:
                curr_state = np.array([0.5, -1.0, 0.0], dtype=np.float32)

            state_tensor = torch.from_numpy(curr_state).unsqueeze(0).float().to(self.device)

            # Compute real-time egomotion from the previous observed state.
            egomotion = self.compute_egomotion_from_state(self.last_state, curr_state)
            egomotion_tensor = torch.from_numpy(egomotion).unsqueeze(0).float().to(self.device)
            self.last_state = curr_state.copy()
            
            # Run inference with egomotion
            with torch.no_grad():
                action_output = self.model.select_action(
                    {
                        "observation": {
                            "pixels": image_tensor,  # (B, C, H, W)
                        },
                        "state": state_tensor,  # (B, 3)
                        "egomotion_z": egomotion_tensor,  # (B, 3) - delta movement
                        "instruction": instruction,  # Text instruction
                    }
                )
            
            # Extract action
            if isinstance(action_output, dict):
                action = action_output.get("action", np.zeros(6, dtype=np.float32))
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
            elif isinstance(action_output, torch.Tensor):
                action = action_output.cpu().numpy()
            else:
                action = np.array(action_output)
            
            # Ensure output is [pitch, yaw, grip, ...]
            if action.ndim > 1:
                action = action.squeeze()
            if len(action) < 6:
                action = np.pad(action, (0, 6 - len(action)), mode="constant")
            
            return action.astype(np.float32)
        
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return np.array([0.5, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    def unload_model(self):
        self.model = None
        self._is_loaded = False
        self.last_state = None
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
