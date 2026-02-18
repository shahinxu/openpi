"""
Hannes prosthetic hand robot model for robosuite
"""
import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Hannes(ManipulatorModel):
    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/hannes/robot.xml"), idn=idn)

    @property
    def default_base(self):
        return None

    @property
    def default_gripper(self):
        return {"right": None}

    @property
    def default_controller_config(self):
        return {"right": "default_hannes"}

    @property
    def init_qpos(self):
        return np.array([0.0, 0.0, 0.5,  # Position: floating at 0.5m height
                        1.0, 0.0, 0.0, 0.0,  # Quaternion: no rotation
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Hand joints: all at zero

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 0.15))
    @property
    def _horizontal_radius(self):
        return 0.15
    @property
    def arm_type(self):
        return "single"
