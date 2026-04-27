"""
Robot controller implementations.
"""
import logging
from typing import Optional, Dict
import numpy as np
from interfaces import RobotControllerInterface

logger = logging.getLogger(__name__)


class DummyRobotController(RobotControllerInterface):
    """
    Dummy robot controller for testing.
    Prints commands but doesn't actually control anything.
    """
    
    def __init__(self):
        self._is_connected = False
        self._last_state = np.array([0.5, -1.0, 0.0], dtype=np.float32)
    
    def connect(self) -> bool:
        self._is_connected = True
        logger.info("Dummy robot 'connected'")
        return True
    
    def send_command(self, action_cmd: Dict[str, int]) -> bool:
        if not self.is_connected:
            logger.error("Robot not connected")
            return False
        
        logger.info(f"[CMD] Pitch={action_cmd['pitch']:3d}, Yaw={action_cmd['yaw']:4d}, Grip={action_cmd['grip']:3d}")
        return True
    
    def get_state(self) -> Optional[np.ndarray]:
        # Dummy: return last predicted state
        return self._last_state.copy()
    
    def set_predicted_state(self, state: np.ndarray):
        """For testing: manually set predicted state."""
        self._last_state = state.copy()
    
    def close(self):
        self._is_connected = False
        logger.info("Dummy robot closed")
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected


class SerialRobotController(RobotControllerInterface):
    """
    Serial port robot controller.
    Sends commands via UART (e.g., to real Hannes robot).
    """
    
    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_port = None
        self._is_connected = False
    
    def connect(self) -> bool:
        try:
            import serial
            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1.0
            )
            self._is_connected = True
            logger.info(f"Serial robot connected: {self.port} @ {self.baudrate} baud")
            return True
        except ImportError:
            logger.error("pyserial not installed. Install with: pip install pyserial")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to serial port {self.port}: {e}")
            return False
    
    def send_command(self, action_cmd: Dict[str, int]) -> bool:
        if not self.is_connected:
            logger.error("Serial robot not connected")
            return False
        
        try:
            # Format: "P{pitch},Y{yaw},G{grip}\n"
            command = f"P{action_cmd['pitch']},Y{action_cmd['yaw']},G{action_cmd['grip']}\n"
            self.serial_port.write(command.encode())
            
            logger.debug(f"Sent: {command.strip()}")
            return True
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return False
    
    def get_state(self) -> Optional[np.ndarray]:
        """
        Try to read state feedback from robot (if implemented).
        Format: "P0.500,Y-1.000,G0.000\n"
        """
        if not self.is_connected or self.serial_port is None:
            return None
        
        try:
            if self.serial_port.in_waiting > 0:
                line = self.serial_port.readline().decode().strip()
                # Parse: "P0.500,Y-1.000,G0.000"
                if line.startswith("P"):
                    parts = line.split(",")
                    pitch = float(parts[0][1:])
                    yaw = float(parts[1][1:])
                    grip = float(parts[2][1:])
                    return np.array([pitch, yaw, grip], dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to parse state feedback: {e}")
        
        return None
    
    def close(self):
        if self.serial_port is not None:
            self.serial_port.close()
            self._is_connected = False
            logger.info("Serial robot closed")
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected


class ROSRobotController(RobotControllerInterface):
    """
    ROS-based robot controller.
    Publishes to joint commands topic and subscribes to state topic.
    """
    
    def __init__(self, cmd_topic: str, state_topic: str):
        self.cmd_topic = cmd_topic
        self.state_topic = state_topic
        self.pub = None
        self.sub = None
        self._is_connected = False
        self._last_state = None
    
    def connect(self) -> bool:
        try:
            import rospy
            rospy.init_node("robot_control_pipeline", anonymous=True)
            
            from geometry_msgs.msg import JointCommand
            
            self.pub = rospy.Publisher(self.cmd_topic, JointCommand, queue_size=1)
            
            # TODO: Implement state subscriber
            # self.sub = rospy.Subscriber(self.state_topic, ..., self._state_callback)
            
            self._is_connected = True
            logger.info(f"ROS robot connected: {self.cmd_topic}")
            return True
        except ImportError:
            logger.error("rospy not installed. Install ROS.")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to ROS: {e}")
            return False
    
    def send_command(self, action_cmd: Dict[str, int]) -> bool:
        if not self.is_connected:
            logger.error("ROS robot not connected")
            return False
        
        try:
            from geometry_msgs.msg import JointCommand
            
            msg = JointCommand()
            msg.header.stamp = __import__('rospy').Time.now()
            msg.position = [
                action_cmd['pitch'],
                action_cmd['yaw'],
                action_cmd['grip'],
            ]
            
            self.pub.publish(msg)
            logger.debug(f"Published: {msg}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish ROS message: {e}")
            return False
    
    def get_state(self) -> Optional[np.ndarray]:
        return self._last_state
    
    def close(self):
        self._is_connected = False
        logger.info("ROS robot closed")
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected


def create_robot_controller(config: dict) -> RobotControllerInterface:
    """Factory function to create robot controller based on config."""
    robot_type = config.get("type", "dummy")
    
    if robot_type == "dummy":
        return DummyRobotController()
    elif robot_type == "serial":
        return SerialRobotController(
            port=config.get("port", "/dev/ttyUSB0"),
            baudrate=config.get("baudrate", 115200),
        )
    elif robot_type == "ros":
        return ROSRobotController(
            cmd_topic=config.get("cmd_topic", "/hannes/joint_commands"),
            state_topic=config.get("state_topic", "/hannes/joint_states"),
        )
    else:
        logger.warning(f"Unknown robot type: {robot_type}, using dummy")
        return DummyRobotController()
