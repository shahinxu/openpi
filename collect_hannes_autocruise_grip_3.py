import argparse
import json
import os
from datetime import datetime

import h5py
import imageio.v2 as imageio
import numpy as np
import robosuite as suite


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

    # Stage 1: move in XY plane first, keep Z fixed
    if xy_dist > xy_tol:
        step_xy = min(max_step, xy_dist)
        direction_xy = xy_delta / max(xy_dist, 1e-8)
        next_pos = current.copy()
        next_pos[:2] = current[:2] + direction_xy * step_xy
        return next_pos, next_pos - current

    # Stage 2: after XY aligned, adjust Z
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


def export_episode_videos(hdf5_path, fps):
    video_dir = os.path.join(os.path.dirname(hdf5_path), "preview_videos")
    os.makedirs(video_dir, exist_ok=True)
    exported = []

    with h5py.File(hdf5_path, "r") as f:
        for group_name in f.keys():
            if not group_name.startswith("episode_"):
                continue
            grp = f[group_name]
            key = "eye_view_images"
            if key not in grp:
                continue
            frames = np.asarray(grp[key])
            mp4_name = f"{os.path.splitext(os.path.basename(hdf5_path))[0]}_{group_name}_{key}.mp4"
            mp4_path = os.path.join(video_dir, mp4_name)
            imageio.mimsave(mp4_path, frames, fps=fps)
            exported.append(mp4_path)

    return exported


def run_episode(
    env,
    rng,
    chosen_obj,
    joints,
    max_steps=None,
    base_speed=0.002,
    capture_images=False,
    far_start_distance=2.5,
    far_align_distance=0.5,
    approach_turn_noise_amp=0.28,
    approach_turn_noise_period=60,
    approach_turn_noise_steps=60,
    approach_ry_rotate_deg=30.0,
    yaw_comp_total_deg=90.0,
    pre_approach_speed=0.018,
    align_speed=0.008,
    sticky_after_close=False,
    image_height=512,
    image_width=512,
):
    obs = env.reset()

    base_quat = np.array([0.707, 0.0, 0.0, -0.707], dtype=np.float64)

    obj_joint = joints[chosen_obj]
    arm_eye_camera_name = resolve_camera_name(env, "arm_eyeview")
    arm_eye_cam_id = None
    arm_eye_cam_base_pos = None
    arm_eye_cam_offset = np.zeros(3, dtype=np.float64)
    arm_eye_drift_max_delta = np.array([0.16, 0.12, 0.10], dtype=np.float64)
    arm_eye_drift_target = np.zeros(3, dtype=np.float64)
    arm_eye_target_refresh_steps = 32
    arm_eye_follow_alpha = 0.055
    arm_eye_walk_bob_amp = 0.010
    arm_eye_walk_bob_period_steps = 26
    arm_eye_walk_bob_phase = float(rng.uniform(0.0, 2.0 * np.pi))
    if arm_eye_camera_name is not None:
        arm_eye_cam_id = env.sim.model.camera_name2id(arm_eye_camera_name)
        arm_eye_cam_base_pos = env.sim.model.cam_pos[arm_eye_cam_id].copy()
    hand_site_ids = get_hand_site_ids(env)
    obj_initial = get_object_pos_from_joint(env, obj_joint)
    obj_initial_quat = get_object_quat_from_joint(env, obj_joint)
    if obj_initial is None:
        raise RuntimeError(f"Cannot read object joint {obj_joint}")
    if obj_initial_quat is None:
        obj_initial_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    obj_upright_quat = np.array(obj_initial_quat, dtype=np.float64)

    # Determine table top Z so the arm never clips below the surface.
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
    eye_view_seq = []

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
    yaw_initial = -np.pi/2
    set_joint_scalar(env, yaw_joint_name, yaw_initial)
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
    yaw_target = float((np.pi / 2.0 - yaw_margin) if yaw_dir > 0 else (-np.pi / 2.0 + yaw_margin))
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

    phase = "approach_far"
    rotate_progress = 0
    forward_progress = 0
    forward_anchor_pos = None
    forward_target_x = None
    prelift_y_target = None
    grasp_progress = 0
    grasp_lock_progress = 0
    hold_progress = 0
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
        action = np.zeros(6, dtype=np.float32)
        obj_now = get_object_pos_from_joint(env, obj_joint)
        target_y_now_align = float(obj_now[1] + target_y_offset)
        target_y_now_push = float(obj_now[1] + target_y_offset)
        front_target = np.array([obj_now[0] + front_sign * front_clearance, target_y_now_align, desired_align_z], dtype=np.float64)
        push_target = np.array([obj_now[0] + front_sign * pregrasp_clearance, target_y_now_push, desired_align_z], dtype=np.float64)
        grasp_target = np.array([obj_now[0] + front_sign * grasp_clearance, target_y_now_push, desired_align_z], dtype=np.float64)
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

        arm_rot_phase_override = None

        if phase == "approach_far":
            staging_target = np.array(
                [obj_now[0] + front_sign * max(far_align_distance, front_clearance + 0.05), obj_now[1] + target_y_offset, desired_align_z],
                dtype=np.float64,
            )
            target = staging_target
            action[0] = 0.0
            action[1] = 0.0
            approach_alpha = min(1.0, approach_far_progress / max(1, int(approach_turn_noise_steps) - 1))
            ry_walk = -approach_ry_rotate_rad * approach_alpha
            if approach_far_progress < max(0, int(approach_turn_noise_steps)):
                # One "turn" is defined as: move to one side, then return to center.
                # This is a half-sine profile over one period.
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
            action[0] = 0.0
            action[1] = 0.0
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

            allow_rotate = front_ready_count >= 8

            if allow_rotate:
                # Rebase rotate interpolation from current pose to avoid sudden reset.
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
            action[0] = 0.0
            action[1] = 0.0
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
            # Controlled micro-forward: move object into palm center with strict cap
            if forward_anchor_pos is None:
                forward_anchor_pos = base_pos.copy()
            if prelift_y_target is None:
                prelift_y_target = float(0.7 * base_pos[1] + 0.3 * target_y_now_push)
            # Keep a small tracking on object's lateral position to avoid ending on one side.
            prelift_y_target = float(0.95 * prelift_y_target + 0.05 * target_y_now_push)
            if forward_target_x is None:
                forward_target_x = float(obj_now[0] + front_sign * grasp_clearance)
                if front_sign > 0:
                    forward_target_x = min(forward_target_x, float(forward_anchor_pos[0] + forward_push_distance))
                else:
                    forward_target_x = max(forward_target_x, float(forward_anchor_pos[0] - forward_push_distance))
            target = np.array([forward_target_x, prelift_y_target, base_pos[2]], dtype=np.float64)
            action[0] = 0.0
            action[1] = 0.0
            forward_progress += 1
            palm_aligned = abs(float(obj_now[0]) - float(hand_center_x)) <= palm_center_tol
            near_forward_target = abs(base_pos[0] - forward_target_x) <= 0.0015
            ready_to_close = forward_progress >= forward_min_steps
            if (robot_can_contact or palm_aligned) and ready_to_close:
                target = np.array([base_pos[0], base_pos[1], desired_align_z], dtype=np.float64)
                action[2:] = hold_close_value
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
            action[0] = 0.0
            action[1] = 0.0
            grasp_alpha = min(1.0, grasp_progress / max(1, grasp_total_steps - 1))
            action[2:] = grasp_start_value + grasp_alpha * (hold_close_value - grasp_start_value)
            grasp_progress += 1
            if grasp_progress >= grasp_total_steps:
                phase = "grasp_lock"
                grasp_lock_progress = 0

        elif phase == "grasp_lock":
            if prelift_y_target is None:
                prelift_y_target = float(base_pos[1])
            target = np.array([base_pos[0], prelift_y_target, grasp_target[2]], dtype=np.float64)
            action[0] = 0.0
            action[1] = 0.0
            action[2:] = hold_close_value
            grasp_lock_progress += 1
            if grasp_lock_progress >= grasp_lock_steps:
                phase = "hold"

        else:
            target = grasp_target
            action[0] = 0.0
            action[1] = 0.0
            action[2:] = hold_close_value
            hold_progress += 1

        # Keep can always upright (translation allowed, tipping forbidden)
        if chosen_obj in ("can", "milk"):
            obj_now_stable = get_object_pos_from_joint(env, obj_joint)
            set_object_pos(env, obj_joint, obj_now_stable, quat=obj_upright_quat)

        # Sticky contact mechanism: track object relative to palm and enforce offset
        if sticky_after_close:
            palm_pos = get_palm_pos(env)
            if palm_pos is not None:
                finger_mean = float(np.mean(action[2:]))
                obj_qpos_addr = env.sim.model.get_joint_qpos_addr(obj_joint)
                obj_now_for_sticky = env.sim.data.qpos[obj_qpos_addr[0]:obj_qpos_addr[0] + 3].copy()
                if sticky_attached and finger_mean < -0.4:
                    sticky_attached = False
                if (not sticky_attached) and phase in ("grasp_lock", "hold"):
                    near_palm = np.linalg.norm(obj_now_for_sticky - palm_pos) < 0.05
                    if robot_can_contact or near_palm:
                        sticky_offset = obj_now_for_sticky - palm_pos
                        sticky_attached = True
                        sticky_ever_attached = True
                if sticky_attached:
                    set_object_pos(env, obj_joint, palm_pos + sticky_offset, quat=obj_upright_quat)

        # Height lock: compensate wrist-control-induced lifting by lowering base target z when needed
        eef_now = get_eef_pos(obs, env)
        z_err = float(eef_now[2] - desired_eef_z)
        if z_err > 0.0 and phase in ("rotate", "forward", "grasp", "grasp_lock", "hold"):
            target = target.copy()
            target[2] -= min(0.03, 1.2 * z_err)

        # No-lift guard: during/after rotation and grasp, z is not allowed to increase.
        # Keep align phase fully bidirectional in z so it can truly converge to desired_align_z.
        if phase in ("rotate", "forward", "grasp", "grasp_lock"):
            target = target.copy()
            target[2] = min(target[2], base_pos[2], desired_align_z)

        # Explicit wrist yaw control with front-side hard gate:
        # rotate only after whole hand is in front side and no contact
        if phase == "rotate":
            arm_ry_now = float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr("robot0_arm_ry")])
            ry_rotated = abs(arm_ry_now - arm_ry_baseline)
            yaw_need = max(0.0, yaw_comp_total_rad - ry_rotated)
            yaw_target_dynamic = float(np.clip(yaw_initial + yaw_dir * yaw_need, -np.pi / 2.0 + yaw_margin, np.pi / 2.0 - yaw_margin))
            yaw_target = yaw_target_dynamic
            if front_latched:
                alpha = rotate_progress / max(1, rotate_total_steps)
                yaw_des = (1.0 - alpha) * yaw_initial + alpha * yaw_target
            else:
                yaw_des = yaw_initial
                action[1] = 0.0
        elif phase in ("forward", "grasp", "grasp_lock", "hold"):
            yaw_des = yaw_target
        else:
            yaw_des = yaw_initial

        if phase == "rotate":
            arm_rot_alpha = min(1.0, rotate_progress / max(1, rotate_total_steps))
            arm_rot_des = interpolate_joint_targets(arm_rot_initial, arm_rot_target, arm_rot_alpha)
        elif arm_rot_phase_override is not None:
            arm_rot_des = arm_rot_phase_override
        elif phase in ("forward", "grasp", "grasp_lock", "hold"):
            arm_rot_des = arm_rot_target.copy()
        else:
            # Keep latest arm rotation through align so ry does not get reset.
            arm_rot_des = arm_rot_hold_des.copy()
        arm_rot_hold_des = arm_rot_des.copy()

        step_speed = base_speed
        if phase == "approach_far":
            # Approach speed is controlled independently from other phases.
            step_speed = max(1e-6, float(pre_approach_speed))
        elif phase == "align":
            # Align speed is controlled independently from other phases.
            step_speed = max(1e-6, float(align_speed))
        elif phase == "forward":
            step_speed = forward_push_step_size

        if phase == "forward":
            next_base, base_delta = move_along_x_only(base_pos, forward_target_x, step_speed)
        elif phase in ("approach_far", "align", "rotate", "grasp", "grasp_lock"):
            next_base, base_delta = move_towards_xy_then_z(base_pos, target, step_speed)
        else:
            next_base, base_delta = move_towards(base_pos, target, step_speed)
        next_base[2] = max(next_base[2], base_z_floor)
        base_pos = next_base

        action[0] = float(pitch_initial)
        action[1] = float(yaw_des)

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
        set_joint_scalar(env, yaw_joint_name, float(yaw_des))
        env.sim.forward()

        states.append(get_compact_state(env, pitch_joint_name, yaw_joint_name, finger_joint_names))
        base_pos_seq.append(base_pos.copy())
        base_delta_seq.append(base_delta.astype(np.float32))
        arm_rot_seq.append(arm_rot_des.astype(np.float32))

        env_action = np.zeros((env.action_dim,), dtype=np.float32)
        env_action[-6:] = action
        obs, reward, done, _ = env.step(env_action)
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
        set_joint_scalar(env, yaw_joint_name, float(yaw_des))
        env.sim.forward()
        obs = env._get_observations(force_update=True)
        next_state_compact = get_compact_state(env, pitch_joint_name, yaw_joint_name, finger_joint_names)
        _g = float(next_state_compact[2])
        actions.append(np.array([float(next_state_compact[0]), float(next_state_compact[1]), _g, _g, _g, _g], dtype=np.float32))
        rewards.append(float(reward))
        dones.append(bool(done))

        if capture_images:
            # Update head-drift offset each frame before rendering.
            if arm_eye_cam_id is not None:
                if t % arm_eye_target_refresh_steps == 0:
                    arm_eye_drift_target = rng.uniform(np.zeros(3, dtype=np.float64), arm_eye_drift_max_delta)
                arm_eye_cam_offset += arm_eye_follow_alpha * (arm_eye_drift_target - arm_eye_cam_offset)
                arm_eye_cam_offset = np.clip(arm_eye_cam_offset, 0.0, arm_eye_drift_max_delta)
                walk_bob_z = 0.0
                if phase == "approach_far":
                    walk_bob_z = arm_eye_walk_bob_amp * np.sin(
                        (2.0 * np.pi * t / max(1, arm_eye_walk_bob_period_steps)) + arm_eye_walk_bob_phase
                    )
                cam_pos = arm_eye_cam_base_pos + arm_eye_cam_offset
                cam_pos[2] += walk_bob_z
                env.sim.model.cam_pos[arm_eye_cam_id] = cam_pos
            arm_eye = None
            if arm_eye_camera_name is not None:
                arm_eye = env.sim.render(height=image_height, width=image_width, camera_name=arm_eye_camera_name)
            if arm_eye is not None:
                if arm_eye.ndim == 3 and arm_eye.shape[2] >= 3:
                    arm_eye = arm_eye[:, :, :3]
                arm_eye = arm_eye[::-1]
                arm_eye = np.rot90(arm_eye, 2)
                eye_view_seq.append(np.asarray(arm_eye, dtype=np.uint8))
        if phase == "hold" and hold_progress >= 25:
            break


    obj_z_end = float(get_object_pos_from_joint(env, obj_joint)[2])
    sticky_success = int(sticky_ever_attached)
    lifted = int(sticky_ever_attached)

    result = {
        "actions": np.asarray(actions),
        "states": np.asarray(states),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "base_pos": np.asarray(base_pos_seq, dtype=np.float32),
        "base_delta": np.asarray(base_delta_seq, dtype=np.float32),
        "arm_rot": np.asarray(arm_rot_seq, dtype=np.float32),
        "chosen_object": chosen_obj,
        "sticky_success": sticky_success,
        "lift_success": lifted,
        "obj_z_start": obj_z_start,
        "obj_z_end": obj_z_end,
    }

    if capture_images and len(eye_view_seq) == len(actions):
        eye_view_arr = np.asarray(eye_view_seq, dtype=np.uint8)
        result["eye_view_images"] = eye_view_arr

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Lift")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument(
        "--horizon",
        type=int,
        default=None
    )
    parser.add_argument("--target-object", type=str, default="can")
    parser.add_argument(
        "--output-format",
        type=str, default="episode", choices=["episode", "autocruise"],
    )
    parser.add_argument(
        "--export-video",
        action="store_true",
        default=True
    )
    parser.add_argument("--far-start-distance", type=float, default=2.5)
    parser.add_argument("--far-align-distance", type=float, default=0.5)
    parser.add_argument("--approach-turn-noise-amp", type=float, default=0.28)
    parser.add_argument("--approach-turn-noise-period", type=int, default=60)
    parser.add_argument("--approach-turn-noise-steps", type=int, default=60)
    parser.add_argument("--approach-ry-rotate-deg", type=float, default=30.0)
    parser.add_argument("--yaw-comp-total-deg", type=float, default=90.0)
    parser.add_argument("--pre-approach-speed", type=float, default=0.018)
    parser.add_argument("--align-speed", type=float, default=0.008)
    parser.add_argument("--sticky-after-close", action="store_true", default=True)
    args = parser.parse_args()
    is_episode_output = args.output_format == "episode"
    capture_images = True

    os.makedirs("hannes_demonstrations", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_stem = "episode" if is_episode_output else "autocruise"
    output = args.output or os.path.join("hannes_demonstrations", f"hannes_{args.task}_{default_stem}_{timestamp}.hdf5")

    env_horizon = args.horizon if args.horizon is not None else 5000

    env = suite.make(
        args.task,
        robots="Hannes",
        has_renderer=False,
        has_offscreen_renderer=bool(capture_images),
        use_camera_obs=False,
        control_freq=args.control_freq,
        horizon=env_horizon,
        ignore_done=True,
        reward_shaping=True,
        use_object_obs=False,
    )

    env_meta = {
        "env_name": args.task,
        "type": 1,
        "env_kwargs": {
            "robots": "Hannes",
            "has_renderer": False,
            "has_offscreen_renderer": bool(capture_images),
            "use_camera_obs": False,
            "control_freq": args.control_freq,
            "horizon": env_horizon,
            "ignore_done": True,
            "reward_shaping": True,
            "use_object_obs": False,
        },
    }

    rng = np.random.default_rng(args.seed)

    # Discover available object joints
    joint_candidates = {
        "milk": "milk_joint0",
        "can": "can_joint0",
    }
    valid_joints = {}
    for name, joint in joint_candidates.items():
        try:
            _ = env.sim.model.get_joint_qpos_addr(joint)
            valid_joints[name] = joint
        except Exception:
            pass

    print("=== Hannes Auto Cruise Collection ===")
    print("Output:", output)
    print("Episodes:", args.episodes)
    print("Objects:", list(valid_joints.keys()))

    with h5py.File(output, "w") as f:
        if args.output_format == "autocruise":
            data = f.create_group("data")
            data.attrs["env_args"] = json.dumps(env_meta, indent=4)
        else:
            f.attrs["env_args"] = json.dumps(env_meta, indent=4)

        total_steps = 0
        successes = 0

        object_names = list(valid_joints.keys())

        for ep in range(args.episodes):
            if args.target_object:
                chosen = args.target_object
            else:
                chosen = object_names[ep % len(object_names)]
            ep_result = run_episode(
                env,
                rng,
                chosen,
                valid_joints,
                max_steps=args.horizon,
                capture_images=bool(capture_images),
                far_start_distance=args.far_start_distance,
                far_align_distance=args.far_align_distance,
                approach_turn_noise_amp=args.approach_turn_noise_amp,
                approach_turn_noise_period=args.approach_turn_noise_period,
                approach_turn_noise_steps=args.approach_turn_noise_steps,
                approach_ry_rotate_deg=args.approach_ry_rotate_deg if hasattr(args, "approach_ry_rotate_deg") else 30.0,
                yaw_comp_total_deg=args.yaw_comp_total_deg if hasattr(args, "yaw_comp_total_deg") else 90.0,
                pre_approach_speed=args.pre_approach_speed,
                align_speed=args.align_speed if hasattr(args, "align_speed") else 0.008,
                sticky_after_close=args.sticky_after_close if hasattr(args, "sticky_after_close") else False,
            )

            if args.output_format == "autocruise":
                demo_name = f"demo_{ep:06d}"
                grp = data.create_group(demo_name)
                grp.create_dataset("actions", data=ep_result["actions"], compression="gzip")
                grp.create_dataset("states", data=ep_result["states"], compression="gzip")
                grp.create_dataset("rewards", data=ep_result["rewards"], compression="gzip")
                grp.create_dataset("dones", data=ep_result["dones"], compression="gzip")
                grp.create_dataset("base_pos", data=ep_result["base_pos"], compression="gzip")
                grp.create_dataset("base_delta", data=ep_result["base_delta"], compression="gzip")
                grp.create_dataset("arm_rot", data=ep_result["arm_rot"], compression="gzip")
                if "eye_view_images" in ep_result:
                    grp.create_dataset("eye_view_images", data=ep_result["eye_view_images"], compression="gzip")
            else:
                ep_name = f"episode_{ep}"
                grp = f.create_group(ep_name)
                grp.create_dataset("actions", data=ep_result["actions"], compression="gzip")
                grp.create_dataset("states", data=ep_result["states"], compression="gzip")
                grp.create_dataset("rewards", data=ep_result["rewards"], compression="gzip")
                grp.create_dataset("dones", data=ep_result["dones"], compression="gzip")
                grp.create_dataset("base_pos", data=ep_result["base_pos"], compression="gzip")
                grp.create_dataset("base_delta", data=ep_result["base_delta"], compression="gzip")
                grp.create_dataset("arm_rot", data=ep_result["arm_rot"], compression="gzip")
                if "eye_view_images" in ep_result:
                    grp.create_dataset("eye_view_images", data=ep_result["eye_view_images"], compression="gzip")

            grp.attrs["num_samples"] = int(ep_result["actions"].shape[0])
            grp.attrs["model_file"] = env.sim.model.get_xml()
            grp.attrs["task"] = args.task
            grp.attrs["chosen_object"] = ep_result["chosen_object"]
            grp.attrs["state_fields"] = json.dumps(["wrist_pitch", "wrist_yaw", "grip_pos_mean"])
            grp.attrs["action_fields"] = json.dumps([
                "wrist_pitch_cmd",
                "wrist_yaw_cmd",
                "forefinger_cmd",
                "midfinger_cmd",
                "ringfinger_cmd",
                "littlefinger_cmd",
            ])
            grp.attrs["scripted_motion_fields"] = json.dumps([
                "base_x",
                "base_y",
                "base_z",
                "arm_rx",
                "arm_ry",
                "arm_rz",
            ])
            grp.attrs["sticky_success"] = int(ep_result["sticky_success"])
            grp.attrs["lift_success"] = int(ep_result["lift_success"])
            grp.attrs["obj_z_start"] = float(ep_result["obj_z_start"])
            grp.attrs["obj_z_end"] = float(ep_result["obj_z_end"])

            total_steps += int(ep_result["actions"].shape[0])
            successes += int(ep_result["sticky_success"])

            if (ep + 1) % 10 == 0:
                print(f"Episode {ep + 1}/{args.episodes}, success so far: {successes}/{ep + 1}")

        if args.output_format == "autocruise":
            data.attrs["total"] = total_steps
            data.attrs["num_success"] = successes
            data.attrs["num_episodes"] = args.episodes
        else:
            f.attrs["total"] = total_steps
            f.attrs["num_success"] = successes
            f.attrs["num_episodes"] = args.episodes

    env.close()

    if is_episode_output and args.export_video:
        exported_videos = export_episode_videos(output, fps=args.control_freq)
        for video_path in exported_videos:
            print("Exported video:", video_path)

    print("=== Done ===")
    print("Saved:", output)
    print(f"Success: {successes}/{args.episodes}")


if __name__ == "__main__":
    main()
