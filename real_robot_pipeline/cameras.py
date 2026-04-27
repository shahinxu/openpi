"""
Camera implementations.
"""
import numpy as np
import logging
from typing import Optional
from interfaces import CameraInterface

logger = logging.getLogger(__name__)


class GradioCameraAdapter(CameraInterface):
    """
    Adapter for Gradio webcam input.
    Used in Gradio interface - receives frames from UI.
    """
    
    def __init__(self):
        self._is_connected = False
        self._last_frame = None
    
    def connect(self) -> bool:
        """Mark as connected (actual connection happens in Gradio)."""
        self._is_connected = True
        logger.info("GradioCamera connected (via UI)")
        return True
    
    def receive_frame(self, frame: np.ndarray):
        """Called by Gradio interface to pass frame."""
        if frame is not None and len(frame.shape) == 3:
            self._last_frame = frame.copy()
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Return last received frame."""
        if self._last_frame is not None:
            return self._last_frame.copy()
        return None
    
    def close(self):
        self._is_connected = False
        logger.info("GradioCamera closed")
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected


class OpenCVCamera(CameraInterface):
    """
    OpenCV-based camera for local testing.
    """
    
    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.cap = None
        self._is_connected = False
    
    def connect(self) -> bool:
        try:
            import cv2
            self.cap = cv2.VideoCapture(self.device_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.device_index}")
                return False
            
            self._is_connected = True
            logger.info(f"OpenCV camera {self.device_index} connected")
            return True
        except Exception as e:
            logger.error(f"OpenCV camera connection failed: {e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        if not self.is_connected:
            return None
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                return None
            
            # Convert BGR to RGB
            import cv2
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
            return None
    
    def close(self):
        if self.cap is not None:
            self.cap.release()
            self._is_connected = False
            logger.info("OpenCV camera closed")
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected


def create_camera(config: dict) -> CameraInterface:
    """Factory function to create camera based on config."""
    camera_type = config.get("type", "gradio_webcam")
    
    if camera_type == "gradio_webcam":
        return GradioCameraAdapter()
    elif camera_type == "opencv":
        return OpenCVCamera(
            device_index=config.get("device_index", 0),
            width=config.get("width", 640),
            height=config.get("height", 480),
        )
    else:
        raise ValueError(f"Unknown camera type: {camera_type}")
