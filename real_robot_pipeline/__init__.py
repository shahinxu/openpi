"""
Real Robot Control Pipeline

A modular, production-ready system for running trained VLA models on real robots.
Easy to extend and modify without changing interfaces.

Usage:
    python main.py [config.yaml]

Components:
- CameraInterface: Webcam/OpenCV input
- InferenceEngineInterface: Model inference (JAX/dummy)
- RobotControllerInterface: Hardware control (Serial/ROS/dummy)
- ActionNormalizerInterface: Action discretization

Configuration:
    Edit config.yaml to switch implementations, adjust ranges, etc.
"""

__version__ = "1.0.0"
