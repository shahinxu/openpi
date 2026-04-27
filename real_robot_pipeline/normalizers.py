"""
Action normalization utilities.
"""
import numpy as np
import logging
from typing import Dict
from interfaces import ActionNormalizerInterface

logger = logging.getLogger(__name__)


class DataRangeNormalizer(ActionNormalizerInterface):
    """
    Normalize actions from data ranges to standardized command ranges.
    
    Data ranges from dataset_new analysis:
    - pitch: [0.48, 0.52] -> [0, 100]
    - yaw: [-1.02, 0.69] -> [-100, 100]
    - grip: [0, 1] -> [0, 100]
    """
    
    def __init__(self, config: dict):
        """
        Initialize normalizer with ranges from config.
        
        config should have:
        {
            "pitch": {"data_min": 0.48, "data_max": 0.52, "cmd_min": 0, "cmd_max": 100},
            "yaw": {"data_min": -1.02, "data_max": 0.69, "cmd_min": -100, "cmd_max": 100},
            "grip": {"data_min": 0.0, "data_max": 1.0, "cmd_min": 0, "cmd_max": 100},
        }
        """
        self.config = config
        
        # Validate config
        for joint in ["pitch", "yaw", "grip"]:
            if joint not in config:
                logger.warning(f"Joint '{joint}' not in config, using defaults")
        
        logger.info("ActionNormalizer initialized")
        logger.debug(f"Config: {config}")
    
    def normalize(self, raw_action: np.ndarray) -> Dict[str, int]:
        """
        Convert raw model output to standardized integer commands.
        
        Args:
            raw_action: [pitch, yaw, grip, ...] in data range
            
        Returns:
            {"pitch": int, "yaw": int, "grip": int}
        """
        try:
            pitch_cmd = self._normalize_joint("pitch", raw_action[0])
            yaw_cmd = self._normalize_joint("yaw", raw_action[1])
            grip_cmd = self._normalize_joint("grip", raw_action[2])
            
            return {
                "pitch": pitch_cmd,
                "yaw": yaw_cmd,
                "grip": grip_cmd,
            }
        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            # Return safe default
            return {"pitch": 50, "yaw": 0, "grip": 0}
    
    def _normalize_joint(self, joint_name: str, raw_value: float) -> int:
        """Normalize a single joint value."""
        if joint_name not in self.config:
            logger.warning(f"Joint '{joint_name}' not configured")
            return 0
        
        cfg = self.config[joint_name]
        data_min = cfg.get("data_min", 0.0)
        data_max = cfg.get("data_max", 1.0)
        cmd_min = cfg.get("cmd_min", 0)
        cmd_max = cfg.get("cmd_max", 100)
        
        # Normalize to [0, 1]
        normalized = (raw_value - data_min) / (data_max - data_min)
        
        # Map to command range
        cmd_value = normalized * (cmd_max - cmd_min) + cmd_min
        
        # Clip to valid range
        cmd_int = int(np.clip(cmd_value, cmd_min, cmd_max))
        
        logger.debug(f"{joint_name}: {raw_value:.4f} -> {cmd_int}")
        
        return cmd_int


class SimpleScalingNormalizer(ActionNormalizerInterface):
    """
    Simple linear scaling normalizer.
    Just multiply by 100 and clip.
    """
    
    def normalize(self, raw_action: np.ndarray) -> Dict[str, int]:
        """Simple: raw * 100, clipped."""
        pitch_cmd = int(np.clip(raw_action[0] * 100, 0, 100))
        yaw_cmd = int(np.clip(raw_action[1] * 100, -100, 100))
        grip_cmd = int(np.clip(raw_action[2] * 100, 0, 100))
        
        return {
            "pitch": pitch_cmd,
            "yaw": yaw_cmd,
            "grip": grip_cmd,
        }


def create_normalizer(config: dict) -> ActionNormalizerInterface:
    """Factory function to create action normalizer."""
    normalizer_type = config.get("normalizer_type", "data_range")
    
    if normalizer_type == "data_range":
        norm_config = config.get("action_normalization", {})
        return DataRangeNormalizer(norm_config)
    elif normalizer_type == "simple_scaling":
        return SimpleScalingNormalizer()
    else:
        logger.warning(f"Unknown normalizer type: {normalizer_type}, using data_range")
        return DataRangeNormalizer(config.get("action_normalization", {}))
