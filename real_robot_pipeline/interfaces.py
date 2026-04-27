"""
Interface definitions for modular pipeline components.
Implement these interfaces to create new camera/model/robot backends.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple
import numpy as np


class CameraInterface(ABC):
    """Abstract camera interface."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to camera. Returns True if successful."""
        pass
    
    @abstractmethod
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame.
        Returns: (H, W, 3) uint8 RGB image, or None if failed.
        """
        pass
    
    @abstractmethod
    def close(self):
        """Close camera connection."""
        pass
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if camera is connected."""
        pass


class InferenceEngineInterface(ABC):
    """Abstract inference engine interface."""
    
    @abstractmethod
    def load_model(self, checkpoint_path: str) -> bool:
        """Load model from checkpoint. Returns True if successful."""
        pass
    
    @abstractmethod
    def infer(
        self,
        image: np.ndarray,
        instruction: str,
        prev_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Run inference on single frame.
        
        Args:
            image: (H, W, 3) uint8 RGB frame
            instruction: Task instruction text
            prev_state: [pitch, yaw, grip] or None
            
        Returns:
            action: [pitch, yaw, grip, ...] in data range
        """
        pass
    
    @abstractmethod
    def unload_model(self):
        """Unload model to free memory."""
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass


class RobotControllerInterface(ABC):
    """Abstract robot controller interface."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to robot. Returns True if successful."""
        pass
    
    @abstractmethod
    def send_command(self, action_cmd: Dict[str, int]) -> bool:
        """
        Send standardized action command.
        
        Args:
            action_cmd: {"pitch": int, "yaw": int, "grip": int}
            
        Returns:
            True if command sent successfully.
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Optional[np.ndarray]:
        """
        Get current robot state (if available).
        
        Returns:
            [pitch, yaw, grip] or None if not available
        """
        pass
    
    @abstractmethod
    def close(self):
        """Close robot connection."""
        pass
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to robot."""
        pass


class ActionNormalizerInterface(ABC):
    """Abstract action normalizer."""
    
    @abstractmethod
    def normalize(self, raw_action: np.ndarray) -> Dict[str, int]:
        """
        Convert raw model output to standardized integer commands.
        
        Args:
            raw_action: [pitch, yaw, grip, ...] in model output range
            
        Returns:
            {"pitch": int, "yaw": int, "grip": int}
        """
        pass
