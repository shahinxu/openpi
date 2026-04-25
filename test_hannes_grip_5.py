import argparse
import os
from datetime import datetime

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import robosuite as suite

from openpi_client import websocket_client_policy as _websocket_client_policy


# ─────────────────────────────────── helpers ────────────────────────────────

def set_base_pose(env, position, quaternion):
    qpos_addr = env.sim.model.get_joint_qpos_addr("robot0_base_free")
    qvel_addr = env.sim.model.get_joint_qvel_addr("robot0_base_free")
    env.sim.data.qpos[qpos_addr[0]:qpos_addr[0] + 3] = position
    env.sim.data.qpos[qpos_addr[0] + 3:qpos_addr[0] + 7] = quaternion
    env.sim.data.qvel[qvel_addr[0]:qvel_addr[0] + 6] = 0.0


def resolve_joint_name(env, joint_candidates):
    for joint_name in joint_candidates:
        try:
            _ = env.sim.model.joint_name2id(joint_name)
            return joint_name
        except Exception:
            continue
    return None


def resolve_camera_name(env, camera_base_name):
    candidates = [camera_base_name, f"robot0_{camera_base_name}"]
    for name in candidates:
        try:
            _ = env.sim.model.camera_name2id(name)
            return name
        except Exception:
            continue
    return None


def get_grip_pos_mean(env, finger_joint_names):
    finger_qpos = []
    for joint_name in finger_joint_names:
        qpos_addr = env.sim.model.get_joint_qpos_addr(joint_name)
        finger_qpos.append(float(env.sim.data.qpos[qpos_addr]))
    if len(finger_qpos) == 0:
        return 0.0
    return float(np.mean(finger_qpos))


def get_compact_state(env, pitch_joint_name, yaw_joint_name, finger_joint_names):
    pitch_addr = env.sim.model.get_joint_qpos_addr(pitch_joint_name)
    yaw_addr = env.sim.model.get_joint_qpos_addr(yaw_joint_name)
    wrist_pitch = float(env.sim.data.qpos[pitch_addr])
    wrist_yaw = float(env.sim.data.qpos[yaw_addr])
    grip_pos_mean = get_grip_pos_mean(env, finger_joint_names)
    return np.array([wrist_pitch, wrist_yaw, grip_pos_mean], dtype=np.float32)


def get_object_pos_from_joint(env, joint_name):
    qpos_addr = env.sim.model.get_joint_qpos_addr(joint_name)
    if not isinstance(qpos_addr, tuple):
        return None
    return env.sim.data.qpos[qpos_addr[0]:qpos_addr[0] + 3].copy()


def get_object_quat_from_joint(env, joint_name):
    qpos_addr = env.sim.model.get_joint_qpos_addr(joint_name)
    if not isinstance(qpos_addr, tuple):
        return None
    return env.sim.data.qpos[qpos_addr[0] + 3:qpos_addr[0] + 7].copy()


def set_object_pos(env, joint_name, xyz, quat=None):
    qpos_addr = env.sim.model.get_joint_qpos_addr(joint_name)
    if not isinstance(qpos_addr, tuple):
        return
    env.sim.data.qpos[qpos_addr[0]:qpos_addr[0] + 3] = xyz
    if quat is not None:
        env.sim.data.qpos[qpos_addr[0] + 3:qpos_addr[0] + 7] = quat
    qvel_addr = env.sim.model.get_joint_qvel_addr(joint_name)
    if isinstance(qvel_addr, tuple):
        env.sim.data.qvel[qvel_addr[0]:qvel_addr[0] + 6] = 0.0


def move_towards(current, target, max_step):
    delta = target - current
    dist = np.linalg.norm(delta)
    if dist < 1e-8:
        return target.copy(), np.zeros_like(current)
    if dist <= max_step:
        next_pos = target.copy()
    else:
        next_pos = current + delta / dist * max_step
    return next_pos, next_pos - current


def move_towards_xy_then_z(current, target, max_step, xy_tol=0.008, z_tol=0.002):
    delta = target - current
    xy_delta = delta[:2]
    xy_dist = np.linalg.norm(xy_delta)
    if xy_dist > xy_tol:
        step_xy = min(max_step, xy_dist)
        direction_xy = xy_delta / max(xy_dist, 1e-8)
        next_pos = current.copy()
        next_pos[:2] = current[:2] + direction_xy * step_xy
        return next_pos, next_pos - current
    if abs(delta[2]) > z_tol:
        next_pos = current.copy()
        next_pos[2] = current[2] + np.sign(delta[2]) * min(max_step, abs(delta[2]))
        return next_pos, next_pos - current
    return current.copy(), np.zeros_like(current)


def move_along_x_only(current, target_x, max_step):
    delta_x = target_x - current[0]
    if abs(delta_x) < 1e-8:
        return current.copy(), np.zeros_like(current)
    step_x = np.sign(delta_x) * min(max_step, abs(delta_x))
    next_pos = current.copy()
    next_pos[0] = current[0] + step_x
    return next_pos, next_pos - current


def is_object_in_contact(env, object_keyword):
    for i in range(env.sim.data.ncon):
        con = env.sim.data.contact[i]
        g1 = env.sim.model.geom_id2name(int(con.geom1)) or ""
        g2 = env.sim.model.geom_id2name(int(con.geom2)) or ""
        if (object_keyword in g1 and "robot" in g2) or (object_keyword in g2 and "robot" in g1):
            return True
    return False


def get_eef_pos(obs, env):
    if isinstance(obs, dict) and "robot0_eef_pos" in obs:
        return np.array(obs["robot0_eef_pos"], dtype=np.float64)
    if hasattr(env, "_eef_xpos"):
        return np.array(env._eef_xpos, dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


def get_palm_pos(env):
    for name in ("robot0_right_center", "robot0_grip_site", "gripper0_grip_site"):
        try:
            site_id = env.sim.model.site_name2id(name)
            return env.sim.data.site_xpos[site_id].copy()
        except Exception:
            continue
    return None


def get_hand_site_ids(env):
    names = [
        "robot0_right_center",
        "robot0_forefinger_pivot",
        "robot0_midfinger_pivot",
        "robot0_ringfinger_pivot",
        "robot0_littlefinger_pivot",
    ]
    ids = []
    for name in names:
        try:
            ids.append(env.sim.model.site_name2id(name))
        except Exception:
            continue
    return ids


def get_hand_x_span(env, hand_site_ids, obs):
    xs = []
    for site_id in hand_site_ids:
        xs.append(float(env.sim.data.site_xpos[site_id][0]))
    if len(xs) == 0:
        xs.append(float(get_eef_pos(obs, env)[0]))
    return min(xs), max(xs)


def hand_is_on_front_side(hand_min_x, hand_max_x, obj_x, clearance, front_sign):
    if front_sign > 0:
        return hand_min_x >= (obj_x + clearance)
    return hand_max_x <= (obj_x - clearance)


def set_joint_scalar(env, joint_name, value):
    addr = env.sim.model.get_joint_qpos_addr(joint_name)
    env.sim.data.qpos[addr] = value
    vel_addr = env.sim.model.get_joint_qvel_addr(joint_name)
    env.sim.data.qvel[vel_addr] = 0.0


def interpolate_joint_targets(start, target, alpha):
    return (1.0 - alpha) * start + alpha * target


def clip_joint_targets(targets, low, high):
    return np.minimum(np.maximum(targets, low), high)


def apply_scripted_human_pose(
    env,
    base_pos,
    base_quat,
    arm_slide_joint_names,
    arm_slide_des,
    arm_rot_joint_names,
    arm_rot_des,
):
    set_base_pose(env, base_pos, base_quat)
    for joint_name, joint_value in zip(arm_slide_joint_names, arm_slide_des):
        set_joint_scalar(env, joint_name, float(joint_value))
    for joint_name, joint_value in zip(arm_rot_joint_names, arm_rot_des):
        set_joint_scalar(env, joint_name, float(joint_value))
    env.sim.forward()


def _render_agentview(env, height=512, width=512):
    """Render agentview; flip to standard upright orientation."""
    frame = env.sim.render(height=height, width=width, camera_name="agentview")
    return frame[::-1]


# ──────────────────────────────── run_episode ────────────────────────────────

def run_episode(
    env,
    rng,
    chosen_obj,
    joints,
    policy,
    prompt,
    max_steps=None,
    base_speed=0.002,
    far_start_distance=2.5,
    far_align_distance=0.5,
    approach_turn_noise_amp=0.28,
    approach_turn_noise_period=60,
    approach_turn_noise_steps=60,
    approach_ry_rotate_deg=30.0,
    yaw_comp_total_deg=90.0,
    pre_approach_speed=0.018,
    align_speed=0.008,
    sticky_after_close=True,
    place_near_object=None,
    place_offset=0.08,
    near_bias_x=0.0,
    near_bias_y=0.03,
    lift_height=0.18,
    image_height=512,
    image_width=512,
    policy_image_size=256,
):
    obs = env.reset()

    base_quat = np.array([0.707, 0.0, 0.0, -0.707], dtype=np.float64)

    obj_joint = joints[chosen_obj]
    hand_site_ids = get_hand_site_ids(env)
    obj_initial = get_object_pos_from_joint(env, obj_joint)
    obj_initial_quat = get_object_quat_from_joint(env, obj_joint)
    if obj_initial is None:
        raise RuntimeError(f"Cannot read object joint {obj_joint}")
    if obj_initial_quat is None:
        obj_initial_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    obj_upright_quat = np.array(obj_initial_quat, dtype=np.float64)

    try:
        table_top_z = float(env.model.mujoco_arena.table_offset[2])
    except Exception:
        table_top_z = 0.8
    base_z_floor = table_top_z + 0.05

    obj_random = obj_initial.copy()
    obj_random[0] = np.clip(obj_initial[0] + rng.uniform(-0.06, 0.06), 0.02, 0.28)
    obj_random[1] = np.clip(obj_initial[1] + rng.uniform(-0.08, 0.08), -0.14, 0.14)
    set_object_pos(env, obj_joint, obj_random, quat=obj_upright_quat)

    desired_align_z = max(float(obj_random[2] - 0.035), base_z_floor)
    sampled_base_z = desired_align_z + rng.uniform(-0.08, 0.25)
    base_z_min = max(float(obj_random[2] + 0.12), base_z_floor)
    far_sign = 1.0
    base_pos = np.array(
        [
            obj_random[0] + far_sign * abs(far_start_distance),
            obj_random[1] + rng.uniform(-0.4, 0.4),
            max(sampled_base_z, base_z_min),
        ],
        dtype=np.float64,
    )

    set_base_pose(env, base_pos, base_quat)
    env.sim.forward()

    obj_pos = get_object_pos_from_joint(env, obj_joint)
    front_sign = 1.0

    front_clearance = 0.10
    pregrasp_clearance = 0.060
    grasp_clearance = 0.020
    target_y_offset = 0.03

    actions = []
    states = []
    rewards = []
    dones = []
    base_pos_seq = []
    base_delta_seq = []
    arm_rot_seq = []
    agent_view_seq = []

    obj_z_start = float(obj_pos[2])
    desired_eef_z = float(get_eef_pos(obs, env)[2])
    yaw_joint_name = "robot0_wrist_yaw"
    pitch_joint_name = "robot0_wrist_pitch"
    finger_joint_names = []
    for base_name in ("forefinger_joint", "midfinger_joint", "ringfinger_joint", "littlefinger_joint"):
        joint_name = resolve_joint_name(env, [f"robot0_{base_name}", base_name])
        if joint_name is not None:
            finger_joint_names.append(joint_name)
    yaw_joint_id = env.sim.model.joint_name2id(yaw_joint_name)
    yaw_low, yaw_high = env.sim.model.jnt_range[yaw_joint_id]
    yaw_initial = float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(yaw_joint_name)])
    pitch_initial = float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(pitch_joint_name)])
    arm_slide_joint_names = ["robot0_arm_tx", "robot0_arm_ty", "robot0_arm_tz"]
    arm_slide_initial = np.array(
        [float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(name)]) for name in arm_slide_joint_names],
        dtype=np.float64,
    )
    arm_rot_joint_names = ["robot0_arm_rx", "robot0_arm_ry", "robot0_arm_rz"]
    arm_rot_initial = np.array(
        [float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(name)]) for name in arm_rot_joint_names],
        dtype=np.float64,
    )
    arm_rot_low = np.array(
        [env.sim.model.jnt_range[env.sim.model.joint_name2id(name)][0] for name in arm_rot_joint_names],
        dtype=np.float64,
    )
    arm_rot_high = np.array(
        [env.sim.model.jnt_range[env.sim.model.joint_name2id(name)][1] for name in arm_rot_joint_names],
        dtype=np.float64,
    )
    arm_ry_baseline = float(arm_rot_initial[1])
    yaw_dir = -front_sign
    yaw_margin = 0.01
    yaw_target = float((yaw_high - yaw_margin) if yaw_dir > 0 else (yaw_low + yaw_margin))
    yaw_comp_total_rad = float(np.deg2rad(max(0.0, yaw_comp_total_deg)))
    approach_ry_rotate_rad = float(np.deg2rad(max(0.0, approach_ry_rotate_deg)))
    arm_rot_target = clip_joint_targets(
        arm_rot_initial + np.array([0.18, 0.0, 0.0], dtype=np.float64),
        arm_rot_low,
        arm_rot_high,
    )
    rotate_total_steps = 90
    rotate_yaw_tol = 0.04
    rotate_ready_count = 0
    forward_push_steps = 16
    forward_push_distance = 0.130
    forward_push_step_size = forward_push_distance / max(1, forward_push_steps)
    forward_min_steps = 6
    palm_center_tol = 0.010
    grasp_total_steps = 45
    hold_close_value = 0.42
    grasp_start_value = 0.18
    grasp_lock_steps = 16
    release_pose_steps = 30
    release_total_steps = 36
    release_open_value = -0.75
    release_pose_yaw_target = float(np.clip(yaw_initial - np.pi / 2.0, yaw_low, yaw_high))
    release_target_pos = None
    release_arm_rot_start = None
    release_arm_rot_target = clip_joint_targets(arm_rot_initial.copy(), arm_rot_low, arm_rot_high)
    lift_steps = 22
    move_near_steps = 80
    move_near_tol = 0.03
    lower_steps = 24
    lower_tol = 0.01
    near_bias_xy = np.array([near_bias_x, near_bias_y], dtype=np.float64)

    near_obj_name = None
    near_obj_joint = None
    if place_near_object in joints and place_near_object != chosen_obj:
        near_obj_name = place_near_object
        near_obj_joint = joints[place_near_object]
    else:
        for cand_name, cand_joint in joints.items():
            if cand_name != chosen_obj:
                near_obj_name = cand_name
                near_obj_joint = cand_joint
                break

    phase = "approach_far"
    rotate_progress = 0
    forward_progress = 0
    forward_anchor_pos = None
    forward_target_x = None
    prelift_y_target = None
    grasp_progress = 0
    grasp_lock_progress = 0
    hold_progress = 0
    lift_progress = 0
    move_near_progress = 0
    move_near_max_steps = move_near_steps
    lower_progress = 0
    release_pose_progress = 0
    release_progress = 0
    lift_target_z = None
    lift_anchor_pos = None
    lift_target_pos = None
    move_target_pos = None
    move_near_base_to_palm_xy = np.zeros(2, dtype=np.float64)
    lower_target_pos = None
    locked_wrist_yaw = None
    locked_arm_rot_des = None
    sticky_success = 0
    sticky_attached = False
    sticky_ever_attached = False
    sticky_offset = np.zeros(3, dtype=np.float64)
    arm_rot_hold_des = arm_rot_initial.copy()
    front_ready_count = 0
    rotate_anchor_x = None
    rotate_anchor_pos = None
    front_enter_clearance = front_clearance
    front_exit_clearance = max(0.07, front_clearance - 0.02)
    front_latched = False
    align_contact_count = 0
    align_contact_retreat_steps = 5
    align_y_tol = 0.01
    align_z_tol = 0.008
    approach_far_progress = 0
    approach_rz_side_sign = 1.0 if rng.random() < 0.5 else -1.0

    step_limit = max_steps if max_steps is not None else 5000

    for t in range(step_limit):
        obj_now = get_object_pos_from_joint(env, obj_joint)
        target_y_now_align = float(obj_now[1] + target_y_offset)
        target_y_now_push = float(obj_now[1] + target_y_offset)
        front_target = np.array(
            [obj_now[0] + front_sign * front_clearance, target_y_now_align, desired_align_z],
            dtype=np.float64,
        )
        push_target = np.array(
            [obj_now[0] + front_sign * pregrasp_clearance, target_y_now_push, desired_align_z],
            dtype=np.float64,
        )
        grasp_target = np.array(
            [obj_now[0] + front_sign * grasp_clearance, target_y_now_push, desired_align_z],
            dtype=np.float64,
        )
        hand_min_x, hand_max_x = get_hand_x_span(env, hand_site_ids, obs)
        hand_center_x = 0.5 * (hand_min_x + hand_max_x)
        robot_can_contact = is_object_in_contact(env, chosen_obj)
        front_ready_enter = hand_is_on_front_side(
            hand_min_x=hand_min_x,
            hand_max_x=hand_max_x,
            obj_x=float(obj_now[0]),
            clearance=front_enter_clearance,
            front_sign=front_sign,
        )
        front_ready_exit = hand_is_on_front_side(
            hand_min_x=hand_min_x,
            hand_max_x=hand_max_x,
            obj_x=float(obj_now[0]),
            clearance=front_exit_clearance,
            front_sign=front_sign,
        )
        if front_latched:
            front_latched = front_ready_exit
        else:
            front_latched = front_ready_enter

        target = base_pos.copy()
        arm_rot_phase_override = None

        if phase == "approach_far":
            staging_target = np.array(
                [obj_now[0] + front_sign * max(far_align_distance, front_clearance + 0.05),
                 obj_now[1] + target_y_offset, desired_align_z],
                dtype=np.float64,
            )
            target = staging_target
            approach_alpha = min(1.0, approach_far_progress / max(1, int(approach_turn_noise_steps) - 1))
            ry_walk = -approach_ry_rotate_rad * approach_alpha
            if approach_far_progress < max(0, int(approach_turn_noise_steps)):
                period = max(1, int(approach_turn_noise_period))
                phase_alpha = (approach_far_progress % period) / period
                side_sign = approach_rz_side_sign
                rz_noise = float(approach_turn_noise_amp) * side_sign * np.sin(np.pi * phase_alpha)
                arm_rot_phase_override = clip_joint_targets(
                    arm_rot_initial + np.array([0.0, ry_walk, rz_noise], dtype=np.float64),
                    arm_rot_low,
                    arm_rot_high,
                )
            else:
                arm_rot_phase_override = clip_joint_targets(
                    arm_rot_initial + np.array([0.0, ry_walk, 0.0], dtype=np.float64),
                    arm_rot_low,
                    arm_rot_high,
                )
            approach_far_progress += 1
            dist_to_stage = float(np.linalg.norm(base_pos - staging_target))
            if dist_to_stage <= 0.05:
                phase = "align"

        elif phase == "align":
            target = front_target
            if robot_can_contact:
                align_contact_count += 1
            else:
                align_contact_count = 0
            need_retreat = (not front_latched) or (align_contact_count >= align_contact_retreat_steps)
            if need_retreat:
                if front_sign > 0:
                    retreat_x = max(base_pos[0] + 0.015, obj_now[0] + front_enter_clearance + 0.03)
                else:
                    retreat_x = min(base_pos[0] - 0.015, obj_now[0] - front_enter_clearance - 0.03)
                target = np.array([retreat_x, front_target[1], desired_align_z], dtype=np.float64)
            else:
                target = np.array([base_pos[0], front_target[1], desired_align_z], dtype=np.float64)
            y_aligned = abs(float(base_pos[1] - front_target[1])) <= align_y_tol
            z_aligned = abs(float(base_pos[2] - desired_align_z)) <= align_z_tol
            align_ready = front_latched and y_aligned and z_aligned and (not need_retreat)
            if align_ready:
                front_ready_count += 1
            else:
                front_ready_count = 0
            if front_ready_count >= 8:
                arm_rot_initial = np.array(
                    [float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(name)]) for name in arm_rot_joint_names],
                    dtype=np.float64,
                )
                arm_rot_target = clip_joint_targets(
                    arm_rot_initial + np.array([0.18, 0.0, 0.0], dtype=np.float64),
                    arm_rot_low,
                    arm_rot_high,
                )
                phase = "rotate"
                rotate_progress = 0
                rotate_anchor_x = float(base_pos[0])
                rotate_anchor_pos = base_pos.copy()

        elif phase == "rotate":
            if rotate_anchor_pos is None:
                rotate_anchor_pos = base_pos.copy()
            target = rotate_anchor_pos.copy()
            rotate_progress += 1
            yaw_now = float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(yaw_joint_name)])
            if abs(yaw_now - yaw_target) <= rotate_yaw_tol:
                rotate_ready_count += 1
            else:
                rotate_ready_count = 0
            if rotate_progress >= rotate_total_steps and rotate_ready_count >= 3:
                phase = "forward"
                forward_progress = 0
                forward_anchor_pos = base_pos.copy()
                prelift_y_target = float(0.7 * base_pos[1] + 0.3 * target_y_now_push)
                forward_target_x = float(obj_now[0] + front_sign * grasp_clearance)
                if front_sign > 0:
                    forward_target_x = min(forward_target_x, float(forward_anchor_pos[0] + forward_push_distance))
                else:
                    forward_target_x = max(forward_target_x, float(forward_anchor_pos[0] - forward_push_distance))

        elif phase == "forward":
            if forward_anchor_pos is None:
                forward_anchor_pos = base_pos.copy()
            if prelift_y_target is None:
                prelift_y_target = float(0.7 * base_pos[1] + 0.3 * target_y_now_push)
            prelift_y_target = float(0.95 * prelift_y_target + 0.05 * target_y_now_push)
            if forward_target_x is None:
                forward_target_x = float(obj_now[0] + front_sign * grasp_clearance)
                if front_sign > 0:
                    forward_target_x = min(forward_target_x, float(forward_anchor_pos[0] + forward_push_distance))
                else:
                    forward_target_x = max(forward_target_x, float(forward_anchor_pos[0] - forward_push_distance))
            target = np.array([forward_target_x, prelift_y_target, base_pos[2]], dtype=np.float64)
            forward_progress += 1
            palm_aligned = abs(float(obj_now[0]) - float(hand_center_x)) <= palm_center_tol
            near_forward_target = abs(base_pos[0] - forward_target_x) <= 0.0015
            ready_to_close = forward_progress >= forward_min_steps
            if (robot_can_contact or palm_aligned) and ready_to_close:
                target = np.array([base_pos[0], base_pos[1], desired_align_z], dtype=np.float64)
                prelift_y_target = float(base_pos[1])
                phase = "grasp"
                grasp_progress = 0
            else:
                if ready_to_close and (near_forward_target or forward_progress >= forward_push_steps):
                    phase = "grasp"
                    grasp_progress = 0

        elif phase == "grasp":
            if prelift_y_target is None:
                prelift_y_target = float(base_pos[1])
            target = np.array([base_pos[0], prelift_y_target, grasp_target[2]], dtype=np.float64)
            grasp_progress += 1
            if grasp_progress >= grasp_total_steps:
                phase = "grasp_lock"
                grasp_lock_progress = 0

        elif phase == "grasp_lock":
            if prelift_y_target is None:
                prelift_y_target = float(base_pos[1])
            target = np.array([base_pos[0], prelift_y_target, grasp_target[2]], dtype=np.float64)
            grasp_lock_progress += 1
            if grasp_lock_progress >= grasp_lock_steps:
                locked_wrist_yaw = float(yaw_target)
                locked_arm_rot_des = arm_rot_hold_des.copy()
                phase = "lift"
                lift_progress = 0
                lift_target_z = float(max(base_pos[2] + lift_height, base_z_floor + 0.05))
                lift_anchor_pos = base_pos.copy()
                lift_target_pos = lift_anchor_pos.copy()
                lift_target_pos[2] = lift_target_z

        elif phase == "lift":
            if lift_anchor_pos is None:
                lift_anchor_pos = base_pos.copy()
            if lift_target_pos is None:
                lift_target_pos = lift_anchor_pos.copy()
                lift_target_pos[2] = float(max(base_pos[2] + lift_height, base_z_floor + 0.05))
            target = lift_target_pos.copy()
            lift_progress += 1
            if np.linalg.norm(base_pos - lift_target_pos) <= 0.003 or lift_progress >= lift_steps:
                phase = "move_near"
                move_near_progress = 0
                move_target_pos = lift_target_pos.copy()
                palm_pos = get_palm_pos(env)
                if palm_pos is not None:
                    move_near_base_to_palm_xy = palm_pos[:2] - base_pos[:2]
                else:
                    move_near_base_to_palm_xy = np.zeros(2, dtype=np.float64)
                if near_obj_joint is not None:
                    near_obj_pos = get_object_pos_from_joint(env, near_obj_joint)
                    if near_obj_pos is not None:
                        desired_palm_x = float(near_obj_pos[0] - near_bias_xy[0])
                        side_sign = -1.0 if float(base_pos[1] - near_obj_pos[1]) >= 0.0 else 1.0
                        desired_palm_y = float(near_obj_pos[1] + side_sign * place_offset - near_bias_xy[1])
                        move_target_pos[0] = np.clip(desired_palm_x - float(move_near_base_to_palm_xy[0]), 0.02, 0.30)
                        move_target_pos[1] = np.clip(desired_palm_y - float(move_near_base_to_palm_xy[1]), -0.18, 0.18)
                move_near_dist = float(np.linalg.norm(move_target_pos[:2] - base_pos[:2]))
                move_near_max_steps = max(move_near_steps, int(np.ceil(move_near_dist / max(base_speed, 1e-6))) + 10)

        elif phase == "move_near":
            if move_target_pos is None:
                move_target_pos = base_pos.copy()
            target = move_target_pos.copy()
            move_near_progress += 1
            reached_xy = float(np.linalg.norm(base_pos[:2] - move_target_pos[:2])) <= 0.003
            if reached_xy or move_near_progress >= move_near_max_steps:
                phase = "lower"
                lower_progress = 0
                lower_target_pos = move_target_pos.copy()
                if lift_anchor_pos is None:
                    lift_anchor_pos = base_pos.copy()
                lower_target_pos[2] = lift_anchor_pos[2]

        elif phase == "lower":
            if lower_target_pos is None:
                lower_target_pos = base_pos.copy()
                if lift_anchor_pos is not None:
                    lower_target_pos[2] = lift_anchor_pos[2]
            target = lower_target_pos.copy()
            lower_progress += 1
            if abs(float(base_pos[2] - lower_target_pos[2])) <= 0.003 or lower_progress >= lower_steps:
                phase = "release"
                release_progress = 0
                release_target_pos = lower_target_pos.copy()

        elif phase == "release_pose":
            if release_target_pos is None:
                release_target_pos = base_pos.copy()
            target = release_target_pos.copy()
            release_pose_progress += 1
            if release_pose_progress >= release_pose_steps:
                phase = "release"
                release_progress = 0

        elif phase == "release":
            if release_target_pos is None:
                release_target_pos = lower_target_pos.copy() if lower_target_pos is not None else base_pos.copy()
            target = release_target_pos.copy()
            release_progress += 1

        else:
            target = grasp_target
            hold_progress += 1

        # Keep object upright during manipulation
        if chosen_obj in ("can", "milk"):
            obj_now_stable = get_object_pos_from_joint(env, obj_joint)
            set_object_pos(env, obj_joint, obj_now_stable, quat=obj_upright_quat)

        # Sticky palm tracking
        if sticky_after_close:
            palm_pos = get_palm_pos(env)
            if palm_pos is not None:
                obj_qpos_addr = env.sim.model.get_joint_qpos_addr(obj_joint)
                obj_now_for_sticky = env.sim.data.qpos[obj_qpos_addr[0]:obj_qpos_addr[0] + 3].copy()
                if sticky_attached and phase == "release" and release_progress >= 3:
                    sticky_attached = False
                if (not sticky_attached) and phase in ("grasp_lock", "lift", "move_near", "lower"):
                    near_palm = np.linalg.norm(obj_now_for_sticky - palm_pos) < 0.05
                    if robot_can_contact or near_palm:
                        sticky_offset = obj_now_for_sticky - palm_pos
                        sticky_attached = True
                        sticky_ever_attached = True
                if sticky_attached:
                    set_object_pos(env, obj_joint, palm_pos + sticky_offset, quat=obj_upright_quat)

        # EEF height compensation
        eef_now = get_eef_pos(obs, env)
        z_err = float(eef_now[2] - desired_eef_z)
        if z_err > 0.0 and phase in ("rotate", "forward", "grasp", "grasp_lock",
                                     "lift", "move_near", "lower", "release_pose", "release"):
            target = target.copy()
            target[2] -= min(0.03, 1.2 * z_err)

        # Prevent z from rising during contact phases
        if phase in ("rotate", "forward", "grasp", "grasp_lock", "release_pose"):
            target = target.copy()
            target[2] = min(target[2], base_pos[2], desired_align_z)

        # Dynamic yaw target during rotate phase (for display/reference only; policy controls actual yaw)
        if phase == "rotate":
            arm_ry_now = float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr("robot0_arm_ry")])
            ry_rotated = abs(arm_ry_now - arm_ry_baseline)
            yaw_need = max(0.0, yaw_comp_total_rad - ry_rotated)
            yaw_target = float(np.clip(
                yaw_initial + yaw_dir * yaw_need, yaw_low + yaw_margin, yaw_high - yaw_margin
            ))

        # Compute arm_rot_des for scripted pose
        if phase == "rotate":
            arm_rot_alpha = min(1.0, rotate_progress / max(1, rotate_total_steps))
            arm_rot_des = interpolate_joint_targets(arm_rot_initial, arm_rot_target, arm_rot_alpha)
        elif arm_rot_phase_override is not None:
            arm_rot_des = arm_rot_phase_override
        elif phase in ("grasp_lock", "lift", "move_near", "lower", "release_pose", "release"):
            arm_rot_des = locked_arm_rot_des.copy() if locked_arm_rot_des is not None else arm_rot_target.copy()
        elif phase in ("forward", "grasp", "hold"):
            arm_rot_des = arm_rot_target.copy()
        else:
            arm_rot_des = arm_rot_hold_des.copy()
        arm_rot_hold_des = arm_rot_des.copy()

        # Base movement speed
        step_speed = base_speed
        if phase == "approach_far":
            step_speed = max(1e-6, float(pre_approach_speed))
        elif phase == "align":
            step_speed = max(1e-6, float(align_speed))
        elif phase == "forward":
            step_speed = forward_push_step_size

        if phase == "forward":
            next_base, base_delta = move_along_x_only(base_pos, forward_target_x, step_speed)
        elif phase in ("approach_far", "align", "rotate", "grasp", "grasp_lock", "release_pose"):
            next_base, base_delta = move_towards_xy_then_z(base_pos, target, step_speed)
        else:
            next_base, base_delta = move_towards(base_pos, target, step_speed)
        next_base[2] = max(next_base[2], base_z_floor)
        base_pos = next_base

        # Apply scripted base + arm_rot + lock pitch
        apply_scripted_human_pose(
            env,
            base_pos,
            base_quat,
            arm_slide_joint_names,
            arm_slide_initial,
            arm_rot_joint_names,
            arm_rot_des,
        )
        set_joint_scalar(env, pitch_joint_name, float(pitch_initial))
        env.sim.forward()

        # Render agentview for policy input
        frame = _render_agentview(env, height=image_height, width=image_width)
        agent_view_seq.append(np.asarray(frame, dtype=np.uint8))

        # Build policy observation
        compact_state = get_compact_state(env, pitch_joint_name, yaw_joint_name, finger_joint_names)
        policy_state = np.zeros((8,), dtype=np.float32)
        policy_state[:compact_state.shape[0]] = compact_state
        policy_img = cv2.resize(frame, (policy_image_size, policy_image_size), interpolation=cv2.INTER_AREA)
        policy_obs = {
            "observation/state": policy_state,
            "observation/image": policy_img,
            "prompt": prompt,
        }
        out = policy.infer(policy_obs)
        action = np.asarray(out["actions"], dtype=np.float32)[0, :6].copy()

        states.append(compact_state.copy())
        actions.append(action.copy())
        base_pos_seq.append(base_pos.copy())
        base_delta_seq.append(base_delta.astype(np.float32))
        arm_rot_seq.append(arm_rot_des.astype(np.float32))

        # Step environment with policy wrist+finger actions
        env_action = np.zeros((env.action_dim,), dtype=np.float32)
        env_action[-6:] = action
        obs, reward, done, _ = env.step(env_action)

        # Re-apply scripted pose after physics step
        apply_scripted_human_pose(
            env,
            base_pos,
            base_quat,
            arm_slide_joint_names,
            arm_slide_initial,
            arm_rot_joint_names,
            arm_rot_des,
        )
        set_joint_scalar(env, pitch_joint_name, float(pitch_initial))
        env.sim.forward()
        obs = env._get_observations(force_update=True)

        rewards.append(float(reward))
        dones.append(bool(done))

        if phase == "release" and release_progress >= release_total_steps:
            break

    obj_z_end = float(get_object_pos_from_joint(env, obj_joint)[2])
    near_xy_dist_end = float("nan")
    if near_obj_joint is not None:
        obj_end_pos = get_object_pos_from_joint(env, obj_joint)
        near_end_pos = get_object_pos_from_joint(env, near_obj_joint)
        if obj_end_pos is not None and near_end_pos is not None:
            near_xy_dist_end = float(np.linalg.norm(obj_end_pos[:2] - near_end_pos[:2]))
    lifted = int(sticky_ever_attached)

    return {
        "actions": np.asarray(actions, dtype=np.float32),
        "states": np.asarray(states, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "base_pos": np.asarray(base_pos_seq, dtype=np.float32),
        "base_delta": np.asarray(base_delta_seq, dtype=np.float32),
        "arm_rot": np.asarray(arm_rot_seq, dtype=np.float32),
        "agent_view_seq": np.asarray(agent_view_seq, dtype=np.uint8),
        "chosen_object": chosen_obj,
        "sticky_success": int(sticky_ever_attached),
        "lift_success": lifted,
        "obj_z_start": obj_z_start,
        "obj_z_end": obj_z_end,
        "near_xy_dist_end": near_xy_dist_end,
    }


# ────────────────────────────── plotting helpers ──────────────────────────────

def _prompt_for_object(name: str) -> str:
    prompts = {
        "can": "Grip the can",
        "milk": "Grip the milk",
        "bread": "Grip the bread",
        "lemon": "Grip the lemon",
        "hammer": "Grip the hammer",
    }
    return prompts.get(name, f"Hold the {name}")


def _plot_and_save(data: np.ndarray, out_path: str, title: str, color: str, label: str) -> None:
    if data.size == 0:
        return
    plot_dims = min(3, data.shape[1])
    dim_names = ["wrist_pitch", "wrist_yaw", "grip_mean"][:plot_dims]
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 220, "font.size": 12,
                          "axes.labelsize": 12, "axes.titlesize": 14,
                          "xtick.labelsize": 11, "ytick.labelsize": 11, "grid.alpha": 0.25})
    fig, axes = plt.subplots(plot_dims, 1, figsize=(10, 6.8), sharex=True)
    if plot_dims == 1:
        axes = [axes]
    fig.suptitle(title, y=0.98, fontweight="bold")
    t_axis = np.arange(data.shape[0])
    for i in range(plot_dims):
        ax = axes[i]
        ax.plot(t_axis, data[:, i], color=color, linewidth=2.0, label=label)
        ax.set_ylabel(dim_names[i])
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time step")
    fig.align_ylabels(axes)
    plt.tight_layout(rect=(0.03, 0.03, 1, 0.95))
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def save_predicted_action_plot(actions: np.ndarray, out_path: str, title: str) -> None:
    _plot_and_save(actions, out_path, title, color="#c84c09", label="pred")


def save_state_plot(states: np.ndarray, out_path: str, title: str) -> None:
    _plot_and_save(states, out_path, title, color="#1f4e79", label="state")


# ───────────────────────────────────── main ───────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hannes policy inference with full autocruise state machine."
    )
    parser.add_argument("--task", type=str, default="Lift")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="hannes_test_autocruise")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--target-object", type=str, default="can",
                        help="Fixed target object: can / milk / bread / lemon / hammer")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    # autocruise motion params
    parser.add_argument("--far-start-distance", type=float, default=2.5)
    parser.add_argument("--far-align-distance", type=float, default=0.5)
    parser.add_argument("--approach-turn-noise-amp", type=float, default=0.28)
    parser.add_argument("--approach-turn-noise-period", type=int, default=60)
    parser.add_argument("--approach-turn-noise-steps", type=int, default=60)
    parser.add_argument("--approach-ry-rotate-deg", type=float, default=30.0)
    parser.add_argument("--yaw-comp-total-deg", type=float, default=90.0)
    parser.add_argument("--pre-approach-speed", type=float, default=0.018)
    parser.add_argument("--align-speed", type=float, default=0.008)
    parser.add_argument("--base-speed", type=float, default=0.002)
    parser.add_argument("--lift-height", type=float, default=0.18)
    parser.add_argument("--place-near-object", type=str, default=None)
    parser.add_argument("--place-offset", type=float, default=0.08)
    parser.add_argument("--near-bias-x", type=float, default=0.0)
    parser.add_argument("--near-bias-y", type=float, default=0.03)
    parser.add_argument("--sticky-after-close", action="store_true", default=True)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = os.path.join(args.output_dir, "videos")
    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    env_horizon = args.horizon if args.horizon is not None else 5000

    env = suite.make(
        args.task,
        robots="Hannes",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        control_freq=args.control_freq,
        horizon=env_horizon,
        ignore_done=True,
        reward_shaping=True,
        use_object_obs=False,
    )

    joint_candidates = {
        "milk": "milk_joint0",
        "can": "can_joint0",
        "lemon": "lemon_joint0",
        "bread": "bread_joint0",
        "hammer": "hammer_joint0",
    }
    valid_joints = {}
    for name, joint in joint_candidates.items():
        try:
            _ = env.sim.model.get_joint_qpos_addr(joint)
            valid_joints[name] = joint
        except Exception:
            pass

    if not valid_joints:
        raise RuntimeError("No target object joints found.")

    print("=== Hannes Policy Test (autocruise format) ===")
    print(f"Output dir : {args.output_dir}")
    print(f"Episodes   : {args.episodes}")
    print(f"Objects    : {list(valid_joints.keys())}")

    policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print("Connected to policy server, metadata:", policy.get_server_metadata())

    rng = np.random.default_rng(args.seed)
    object_names = list(valid_joints.keys())

    for ep in range(args.episodes):
        if args.target_object:
            if args.target_object not in valid_joints:
                raise ValueError(
                    f"--target-object '{args.target_object}' not available. "
                    f"Choose from {list(valid_joints.keys())}"
                )
            chosen = args.target_object
        else:
            chosen = object_names[ep % len(object_names)]

        prompt = _prompt_for_object(chosen)
        print(f"\n[Episode {ep:02d}] object={chosen}  prompt={prompt!r}")

        ep_result = run_episode(
            env,
            rng,
            chosen,
            valid_joints,
            policy=policy,
            prompt=prompt,
            max_steps=args.horizon,
            base_speed=args.base_speed,
            far_start_distance=args.far_start_distance,
            far_align_distance=args.far_align_distance,
            approach_turn_noise_amp=args.approach_turn_noise_amp,
            approach_turn_noise_period=args.approach_turn_noise_period,
            approach_turn_noise_steps=args.approach_turn_noise_steps,
            approach_ry_rotate_deg=args.approach_ry_rotate_deg,
            yaw_comp_total_deg=args.yaw_comp_total_deg,
            pre_approach_speed=args.pre_approach_speed,
            align_speed=args.align_speed,
            sticky_after_close=args.sticky_after_close,
            place_near_object=args.place_near_object,
            place_offset=args.place_offset,
            near_bias_x=args.near_bias_x,
            near_bias_y=args.near_bias_y,
            lift_height=args.lift_height,
        )

        print(
            f"  steps={ep_result['actions'].shape[0]}  "
            f"lift={ep_result['lift_success']}  "
            f"sticky={ep_result['sticky_success']}  "
            f"obj_z: {ep_result['obj_z_start']:.3f} -> {ep_result['obj_z_end']:.3f}  "
            f"near_xy_dist={ep_result['near_xy_dist_end']:.3f}"
        )

        # Save video
        video_path = os.path.join(
            video_dir,
            f"{args.task}_{chosen}_{timestamp}_ep{ep:02d}_agentview.mp4",
        )
        imageio.mimsave(video_path, ep_result["agent_view_seq"], fps=args.control_freq)
        print(f"  Video  -> {video_path}")

        # Save action plot
        action_plot_path = os.path.join(
            plot_dir,
            f"{args.task}_{chosen}_{timestamp}_ep{ep:02d}_pred_actions.png",
        )
        save_predicted_action_plot(
            ep_result["actions"],
            action_plot_path,
            f"Predicted Actions | {chosen} | ep{ep:02d} | lift={ep_result['lift_success']}",
        )
        print(f"  Action -> {action_plot_path}")

        # Save state plot
        state_plot_path = os.path.join(
            plot_dir,
            f"{args.task}_{chosen}_{timestamp}_ep{ep:02d}_states.png",
        )
        save_state_plot(
            ep_result["states"],
            state_plot_path,
            f"States | {chosen} | ep{ep:02d}",
        )
        print(f"  State  -> {state_plot_path}")

    env.close()
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
