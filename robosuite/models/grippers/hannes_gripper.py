"""
Gripper for Hannes prosthetic hand (coupled finger movement).
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class HannesGripperBase(GripperModel):
    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/hannes_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.0, 0.0, 0.0, 0.0])  # forefinger, midfinger, ringfinger, littlefinger

    @property
    def _important_sites(self):
        return {
            "grip_site": "grip_site",
            "grip_cylinder": "grip_site_cylinder",
        }

    @property
    def _important_geoms(self):
        return {
            "forefinger": ["forefinger_collision"],
            "midfinger": ["midfinger_collision"],
            "ringfinger": ["ringfinger_collision"],
            "littlefinger": ["littlefinger_collision"],
        }


class HannesGripper(HannesGripperBase):
    def format_action(self, action):
        return np.array([])

    @property
    def dof(self):
        return 0

    @property
    def init_qpos(self):
        return np.array([])  # No init positions needed
