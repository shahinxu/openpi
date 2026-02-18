import copy
import os
from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.controllers import composite_controller_factory
from robosuite.robots.fixed_base_robot import FixedBaseRobot
from robosuite.models.grippers import gripper_factory
from robosuite.models.robots import create_robot


class FloatingHandRobot(FixedBaseRobot):
    def __init__(
        self,
        robot_type: str,
        idn=0,
        composite_controller_config=None,
        initial_qpos=None,
        initialization_noise=None,
        base_type="default",
        gripper_type="default",
        control_freq=20,
        lite_physics=True,
    ):
        super().__init__(
            robot_type=robot_type,
            idn=idn,
            composite_controller_config=composite_controller_config,
            initial_qpos=initial_qpos,
            initialization_noise=initialization_noise,
            base_type=base_type,
            gripper_type=gripper_type,
            control_freq=control_freq,
            lite_physics=lite_physics,
        )

    def _load_controller(self):
        # Flag for loading urdf once (only applicable for IK controllers)
        self.composite_controller = composite_controller_factory(
            type=self.composite_controller_config.get("type", "BASIC"),
            sim=self.sim,
            robot_model=self.robot_model,
            grippers={self.get_gripper_name(arm): self.gripper[arm] for arm in self.arms},
        )
        self._load_arm_controllers()

        self._postprocess_part_controller_config()
        self.composite_controller.load_controller_config(
            self.part_controller_config,
            self.composite_controller_config.get("composite_controller_specific_configs", {}),
        )

        self.enable_parts()

    def _load_arm_controllers(self):
        urdf_loaded = False
        for arm in self.arms:
            # Add to the controller dict additional relevant params
            self.part_controller_config[arm]["robot_name"] = self.name
            self.part_controller_config[arm]["sim"] = self.sim
            
            # Set ref_name based on whether gripper exists
            if self.gripper[arm] is not None:
                # Use gripper's grip_site if gripper exists
                self.part_controller_config[arm]["ref_name"] = self.gripper[arm].important_sites["grip_site"]
            else:
                # Use robot's center site for integrated grippers
                self.part_controller_config[arm]["ref_name"] = f"robot{self.robot_model.idn}_{arm}_center"
            
            self.part_controller_config[arm]["part_name"] = arm
            self.part_controller_config[arm]["naming_prefix"] = self.robot_model.naming_prefix
            self.part_controller_config[arm]["eef_rot_offset"] = self.eef_rot_offset[arm]
            self.part_controller_config[arm]["ndim"] = self._joint_split_idx
            self.part_controller_config[arm]["policy_freq"] = self.control_freq
            self.part_controller_config[arm]["lite_physics"] = self.lite_physics
            
            # Set joint indexes for the arm
            (start, end) = (None, self._joint_split_idx) if arm == "right" else (self._joint_split_idx, None)
            self.part_controller_config[arm]["joint_indexes"] = {
                "joints": self.arm_joint_indexes[start:end],
                "qpos": self._ref_arm_joint_pos_indexes[start:end],
                "qvel": self._ref_arm_joint_vel_indexes[start:end],
            }
            self.part_controller_config[arm]["actuator_range"] = (
                self.torque_limits[0][start:end],
                self.torque_limits[1][start:end],
            )
            
            # Only load urdf the first time
            self.part_controller_config[arm]["load_urdf"] = True if not urdf_loaded else False
            urdf_loaded = True
            
            # Load gripper controllers if gripper exists and is not None
            if self.has_gripper[arm] and self.gripper[arm] is not None:
                assert "gripper" in self.part_controller_config[arm], "Gripper controller config not found!"
                gripper_name = self.get_gripper_name(arm)
                self.part_controller_config[gripper_name] = {}
                self.part_controller_config[gripper_name]["type"] = self.part_controller_config[arm]["gripper"]["type"]
                # Add other gripper controller params as needed

    def load_model(self):
        # Create robot model
        self.robot_model = create_robot(self.name, idn=self.idn)

        # Skip base loading for floating hand - no base needed!
        # (Comment out the base loading section)
        
        # Update joints and actuators
        self.robot_model.update_joints()
        self.robot_model.update_actuators()
        
        # Use default from robot model for initial joint positions if not specified
        if self.init_qpos is None:
            self.init_qpos = self.robot_model.init_qpos

        # Load grippers
        for arm in self.arms:
            if self.has_gripper[arm]:
                if self.gripper_type[arm] == "default":
                    # Load the default gripper from the robot file
                    idn = "_".join((str(self.idn), arm))
                    # Check if default gripper is None (for robots with integrated grippers like Hannes)
                    if self.robot_model.default_gripper[arm] is None:
                        self.gripper[arm] = None
                    else:
                        self.gripper[arm] = gripper_factory(self.robot_model.default_gripper[arm], idn=idn)
                else:
                    # Load user-specified gripper
                    self.gripper[arm] = gripper_factory(self.gripper_type[arm], idn="_".join((str(self.idn), arm)))
            else:
                # No gripper
                self.gripper[arm] = None
            
            # Only set eef rotation offset and add gripper if it's not None
            if self.gripper[arm] is not None:
                # Grab eef rotation offset
                self.eef_rot_offset[arm] = T.quat_multiply(
                    self.robot_model.hand_rotation_offset[arm], self.gripper[arm].rotation_offset
                )

                # Adjust gripper mount offset and quaternion if users specify custom values
                custom_gripper_mount_pos_offset = self.robot_model.gripper_mount_pos_offset.get(arm, None)
                custom_gripper_mount_quat_offset = self.robot_model.gripper_mount_quat_offset.get(arm, None)

                if custom_gripper_mount_pos_offset is not None:
                    self.gripper[arm].set_mount_pos_offset(custom_gripper_mount_pos_offset)
                if custom_gripper_mount_quat_offset is not None:
                    self.gripper[arm].set_mount_quat_offset(custom_gripper_mount_quat_offset)

                # Add gripper to robot
                if self.gripper[arm].naming_prefix is None:
                    gripper_eef_name = self.robot_model.eef_name[arm]
                else:
                    gripper_eef_name = self.gripper[arm].naming_prefix + "_" + self.robot_model.eef_name[arm].split("_")[-1]
                self.robot_model.add_gripper(self.gripper[arm], arm_name=gripper_eef_name)
            else:
                # For robots without external gripper, set a default eef_rot_offset
                self.eef_rot_offset[arm] = np.array([0, 0, 0, 1])  # identity quaternion

    def reset(self, deterministic=False, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        init_qpos = np.array(self.init_qpos)
        if not deterministic:
            if self.initialization_noise["type"] == "gaussian":
                noise = rng.standard_normal(len(self.init_qpos)) * self.initialization_noise["magnitude"]
            elif self.initialization_noise["type"] == "uniform":
                noise = rng.uniform(-1.0, 1.0, len(self.init_qpos)) * self.initialization_noise["magnitude"]
            else:
                raise ValueError("Error: Invalid noise type specified. Options are 'gaussian' or 'uniform'.")
            init_qpos += noise

        qpos_idx = 0
        for i, joint_name in enumerate(self.robot_joints):
            joint_idx = self.sim.model.joint_name2id(joint_name)
            joint_type = self.sim.model.jnt_type[joint_idx]
            
            if joint_type == 0:
                qpos_addr = self.sim.model.get_joint_qpos_addr(joint_name)
                free_joint_qpos = init_qpos[qpos_idx:qpos_idx+7].copy()
                free_joint_qpos[0] = 0.125
                free_joint_qpos[1] = 0
                free_joint_qpos[2] = 0.84

                free_joint_qpos[3:7] = [0.707, 0, 0, -0.707]
                self.sim.data.qpos[qpos_addr[0]:qpos_addr[1]] = free_joint_qpos
                qpos_idx += 7
            else:
                qpos_addr = self.sim.model.get_joint_qpos_addr(joint_name)
                self.sim.data.qpos[qpos_addr] = init_qpos[qpos_idx]
                qpos_idx += 1

        self._load_controller()

        self.base_pos = self.sim.data.get_body_xpos(self.robot_model.root_body)
        self.base_ori = self.sim.data.get_body_xmat(self.robot_model.root_body).reshape((3, 3))

        from robosuite.utils.buffers import DeltaBuffer, RingBuffer
        controllable_dof = sum(1 for jname in self.robot_joints 
                              if self.sim.model.jnt_type[self.sim.model.joint_name2id(jname)] != 0)
        
        self.recent_qpos = DeltaBuffer(dim=controllable_dof)
        self.recent_actions = DeltaBuffer(dim=self.action_dim)
        self.recent_torques = DeltaBuffer(dim=controllable_dof)

        for arm in self.arms:
            if self.has_gripper[arm] and self.gripper[arm] is not None:
                if not deterministic:
                    self.sim.data.qpos[self._ref_gripper_joint_pos_indexes[arm]] = self.gripper[arm].init_qpos

                self.gripper[arm].current_action = np.zeros(self.gripper[arm].dof)

            self.recent_ee_forcetorques[arm] = DeltaBuffer(dim=6)
            self.recent_ee_pose[arm] = DeltaBuffer(dim=7)
            self.recent_ee_vel[arm] = DeltaBuffer(dim=6)
            self.recent_ee_vel_buffer[arm] = RingBuffer(dim=6, length=10)
            self.recent_ee_acc[arm] = DeltaBuffer(dim=6)

        # Update controller
        self.composite_controller.update_state()

        # Reset controller
        self.composite_controller.reset()

    def _visualize_grippers(self, visible):
        for arm in self.arms:
            if self.gripper[arm] is not None:
                self.gripper[arm].set_sites_visibility(sim=self.sim, visible=visible)

    def setup_references(self):
        # First setup robot joint lists, filtering out free joint
        self.robot_joints = self.robot_model.joints
        
        # Separate free joint from controllable joints
        controllable_joints = []
        for joint_name in self.robot_joints:
            joint_idx = self.sim.model.joint_name2id(joint_name)
            joint_type = self.sim.model.jnt_type[joint_idx]
            if joint_type != 0:  # not a free joint
                controllable_joints.append(joint_name)
            else:
                # Store free joint info for later use
                self._free_joint_name = joint_name
                self._free_joint_qpos_addr = self.sim.model.get_joint_qpos_addr(joint_name)
        
        # Setup references for controllable joints only
        self._ref_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in controllable_joints]
        self._ref_joint_vel_indexes = [self.sim.model.get_joint_qvel_addr(x) for x in controllable_joints]
        self._ref_joint_indexes = [self.sim.model.joint_name2id(joint) for joint in controllable_joints]

        # Setup arm joint references
        self._ref_arm_joint_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator) for actuator in self.robot_model.arm_actuators
        ]

        # Filter arm joints to exclude free joint
        self.robot_arm_joints = [
            joint for joint in self.robot_model.arm_joints
            if self.sim.model.jnt_type[self.sim.model.joint_name2id(joint)] != 0
        ]
        self._ref_arm_joint_indexes = [self.sim.model.joint_name2id(joint) for joint in self.robot_arm_joints]
        self._ref_arm_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(joint) for joint in self.robot_arm_joints
        ]
        self._ref_arm_joint_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(joint) for joint in self.robot_arm_joints
        ]

        # Setup per-arm references
        self._ref_joints_indexes_dict = {}
        self._ref_actuators_indexes_dict = {}

        for arm in self.arms:
            # For single-arm robots, use all controllable arm joints
            self._ref_joints_indexes_dict[arm] = self._ref_arm_joint_indexes
            self._ref_actuators_indexes_dict[arm] = [
                self.sim.model.actuator_name2id(actuator) for actuator in self.robot_model.arm_actuators
            ]
            
            # Setup gripper references if present and not None
            if self.has_gripper[arm] and self.gripper[arm] is not None:
                self.gripper_joints[arm] = list(self.gripper[arm].joints)
                self._ref_gripper_joint_pos_indexes[arm] = [
                    self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_joints[arm]
                ]
                self._ref_gripper_joint_vel_indexes[arm] = [
                    self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_joints[arm]
                ]
                self._ref_joint_gripper_actuator_indexes[arm] = [
                    self.sim.model.actuator_name2id(actuator) for actuator in self.gripper[arm].actuators
                ]
                # Add gripper-specific entries
                gripper_name = self.get_gripper_name(arm)
                self._ref_joints_indexes_dict[gripper_name] = [
                    self.sim.model.joint_name2id(joint) for joint in self.gripper_joints[arm]
                ]
                self._ref_actuators_indexes_dict[gripper_name] = self._ref_joint_gripper_actuator_indexes[arm]
            else:
                # For robots with integrated grippers, initialize empty lists
                self.gripper_joints[arm] = []
                self._ref_gripper_joint_pos_indexes[arm] = []
                self._ref_gripper_joint_vel_indexes[arm] = []
                self._ref_joint_gripper_actuator_indexes[arm] = []

            # IDs of sites for eef visualization
            if self.gripper[arm] is not None:
                self.eef_site_id[arm] = self.sim.model.site_name2id(self.gripper[arm].important_sites["grip_site"])
                self.eef_cylinder_id[arm] = self.sim.model.site_name2id(self.gripper[arm].important_sites["grip_cylinder"])
            else:
                # For robots with integrated grippers, use the robot's center site
                # The site name is in the format "robot{idn}_{arm}_center"
                eef_site_name = f"robot{self.robot_model.idn}_{arm}_center"
                self.eef_site_id[arm] = self.sim.model.site_name2id(eef_site_name)
                self.eef_cylinder_id[arm] = -1  # No cylinder for integrated grippers


    @property
    def ee_force(self):
        vals = {}
        for arm in self.arms:
            if self.gripper[arm] is not None:
                vals[arm] = self.get_sensor_measurement(self.gripper[arm].important_sensors["force_ee"])
            else:
                # For robots with integrated grippers, return zero force
                vals[arm] = np.zeros(3)
        return vals

    @property
    def ee_torque(self):
        vals = {}
        for arm in self.arms:
            if self.gripper[arm] is not None:
                vals[arm] = self.get_sensor_measurement(self.gripper[arm].important_sensors["torque_ee"])
            else:
                # For robots with integrated grippers, return zero torque
                vals[arm] = np.zeros(3)
        return vals

    def control(self, action, policy_step=False):
        # Store the initial free joint position if not already stored
        if not hasattr(self, '_locked_free_joint_qpos'):
            # Find the free joint
            for joint_name in self.robot_joints:
                joint_idx = self.sim.model.joint_name2id(joint_name)
                joint_type = self.sim.model.jnt_type[joint_idx]
                if joint_type == 0:  # free joint
                    qpos_addr = self.sim.model.get_joint_qpos_addr(joint_name)
                    self._locked_free_joint_qpos = self.sim.data.qpos[qpos_addr[0]:qpos_addr[1]].copy()
                    self._free_joint_qpos_addr = qpos_addr
                    break
        
        # Call parent control method
        super().control(action, policy_step=policy_step)
        
        # Partially lock free joint: zero velocities to prevent gravity accumulation, lock orientation, allow position change
        if hasattr(self, '_free_joint_qpos_addr'):
            qvel_addr = self.sim.model.get_joint_qvel_addr(self.robot_joints[0])  # base_free joint
            # Zero all velocities (translation + rotation) to prevent drift
            self.sim.data.qvel[qvel_addr[0]:qvel_addr[1]] = 0.0
            
            # Lock orientation (quaternion) but allow position change
            if hasattr(self, '_locked_free_joint_qpos'):
                # Only restore orientation part (quat), not position part (xyz)
                self.sim.data.qpos[self._free_joint_qpos_addr[0]+3:self._free_joint_qpos_addr[1]] = self._locked_free_joint_qpos[3:7]
