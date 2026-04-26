import argparse
import json
import os
import time
from datetime import datetime

import h5py
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


def get_palm_pos(env):
    for name in ("robot0_right_center", "robot0_grip_site", "gripper0_grip_site"):
        try:
            site_id = env.sim.model.site_name2id(name)
            return env.sim.data.site_xpos[site_id].copy()
        except Exception:
            continue
    return None


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


def set_joint_scalar_clipped(env, joint_name, value):
    addr = env.sim.model.get_joint_qpos_addr(joint_name)
    joint_id = env.sim.model.joint_name2id(joint_name)
    low, high = env.sim.model.jnt_range[joint_id]
    clipped = float(np.clip(value, low, high))
    env.sim.data.qpos[addr] = clipped
    vel_addr = env.sim.model.get_joint_qvel_addr(joint_name)
    env.sim.data.qvel[vel_addr] = 0.0
    return clipped


def run_episode(
    env,
    rng,
    chosen_obj,
    joints,
    base_speed=0.006,
    sticky_after_close=False,
    place_near_object=None,
    place_offset=0.08,
    capture_images=False,
    image_height=512,
    image_width=512,
    anomaly_type="wrist_pitch",
):
    obs = env.reset()

    base_quat = np.array([0.707, 0.0, 0.0, -0.707], dtype=np.float64)

    obj_joint = joints[chosen_obj]
    obj_qpos_addr = env.sim.model.get_joint_qpos_addr(obj_joint)
    hand_site_ids = get_hand_site_ids(env)
    obj_initial = get_object_pos_from_joint(env, obj_joint)
    obj_initial_quat = get_object_quat_from_joint(env, obj_joint)
    if obj_initial is None:
        raise RuntimeError(f"Cannot read object joint {obj_joint}")
    if obj_initial_quat is None:
        obj_initial_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    obj_upright_quat = np.array(obj_initial_quat, dtype=np.float64)

    # Randomize target object position on tabletop neighborhood
    obj_random = obj_initial.copy()
    obj_random[0] = np.clip(obj_initial[0] + rng.uniform(-0.06, 0.06), 0.02, 0.28)
    obj_random[1] = np.clip(obj_initial[1] + rng.uniform(-0.08, 0.08), -0.14, 0.14)
    set_object_pos(env, obj_joint, obj_random, quat=obj_upright_quat)

    # Keep non-target objects at their environment-sampled positions.

    # Random initial placement with safe z floor to avoid spawning below tabletop
    desired_align_z = float(obj_random[2] - 0.06)
    sampled_base_z = desired_align_z + rng.uniform(-0.08, 0.25)
    base_z_min = float(obj_random[2] + 0.12)
    base_pos = np.array(
        [
            obj_random[0] + rng.uniform(-0.35, 0.35),
            obj_random[1] + rng.uniform(-0.35, 0.35),
            max(sampled_base_z, base_z_min),
        ],
        dtype=np.float64,
    )

    set_base_pose(env, base_pos, base_quat)
    env.sim.forward()

    # Re-read target object position after randomization and forward
    obj_pos = get_object_pos_from_joint(env, obj_joint)
    front_sign = 1.0

    # Explicit staged routine:
    # 1) move to front of object and match height
    # 2) rotate wrist to face object
    # 3) move forward
    # 4) close fingers (grasp)
    # 5) lift along +Z
    # 6) move to beside a reference object
    # 7) descend along -Z (inverse of lift)
    # 8) release fingers
    front_clearance = 0.10
    pregrasp_clearance = 0.085
    grasp_clearance = 0.055
    target_y_offset = 0.02

    actions = []
    states = []
    rewards = []
    dones = []
    base_pos_seq = []
    base_delta_seq = []
    agentview_seq = []
    sideview_seq = []

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
    pitch_joint_id = env.sim.model.joint_name2id(pitch_joint_name)
    pitch_low, pitch_high = env.sim.model.jnt_range[pitch_joint_id]
    yaw_initial = float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(yaw_joint_name)])
    pitch_initial = float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(pitch_joint_name)])
    yaw_dir = -front_sign
    yaw_margin = 0.01
    yaw_target = float((np.pi / 2.0 - yaw_margin) if yaw_dir > 0 else (-np.pi / 2.0 + yaw_margin))
    rotate_total_steps = 60
    forward_push_steps = 16
    forward_push_distance = 0.084
    forward_push_step_size = forward_push_distance / max(1, forward_push_steps)
    palm_center_tol = 0.018
    grasp_total_steps = 45
    hold_close_value = 0.55
    grasp_lock_steps = 16
    release_open_value = -0.75
    lift_distance = float(rng.uniform(0.16, 0.21))
    lift_total_steps = 26
    move_total_steps = 32
    near_move_total_steps = 56
    lower_total_steps = 26
    release_total_steps = 18

    valid_anomaly_types = {"none", "wrist_pitch", "wrist_yaw", "base_z_high", "base_xy_shift", "grip_flip"}
    anomaly_type = str(anomaly_type)
    if anomaly_type not in valid_anomaly_types:
        raise ValueError(f"Unsupported anomaly_type={anomaly_type}, choose from {sorted(valid_anomaly_types)}")

    # Fixed (but randomized per episode) anomaly timing controls.
    anomaly_rotate_trigger_step = int(rng.integers(6, max(7, rotate_total_steps - 6)))

    anomaly_active = False
    anomaly_kind = "none"
    anomaly_injected_once = False
    anomaly_just_injected = False
    base_x_low, base_x_high = 0.02, 0.30
    base_y_low, base_y_high = -0.18, 0.18
    base_z_low = float(desired_align_z)
    base_z_high = float(max(base_pos[2] + 0.14, desired_align_z + 0.24))
    anomaly_anchor = {
        "base": None,
        "pitch": None,
        "yaw": None,
        "grip": None,
    }
    step_idx = 0

    near_target_joint = None
    near_target_name = None
    if place_near_object and place_near_object in joints and place_near_object != chosen_obj:
        near_target_joint = joints[place_near_object]
        near_target_name = place_near_object
    if near_target_joint is None:
        for name in ("milk", "can", "hammer", "lemon", "bread"):
            if name in joints and name != chosen_obj:
                near_target_joint = joints[name]
                near_target_name = name
                break
    if near_target_joint is None:
        raise RuntimeError("No valid reference object to place near. Provide --place-near-object.")
    active_move_total_steps = near_move_total_steps if near_target_joint is not None else move_total_steps

    phase = "align"
    rotate_progress = 0
    forward_progress = 0
    forward_anchor_pos = None
    forward_target_x = None
    prelift_y_target = None
    grasp_progress = 0
    grasp_lock_progress = 0
    lift_progress = 0
    move_progress = 0
    lower_progress = 0
    release_progress = 0
    lift_anchor_pos = None
    lift_target_pos = None
    move_target_pos = None
    lower_target_pos = None
    front_ready_count = 0
    rotate_anchor_x = None
    rotate_anchor_pos = None
    front_enter_clearance = front_clearance
    front_exit_clearance = max(0.07, front_clearance - 0.02)
    front_latched = False
    sticky_attached = False
    sticky_ever_attached = False
    sticky_offset = np.zeros(3, dtype=np.float64)

    while True:
        action = np.zeros(6, dtype=np.float32)

        # Inject exactly one anomaly at a random time inside rotate phase.
        random_rotate_hit = (phase == "rotate" and rotate_progress >= anomaly_rotate_trigger_step)
        if (
            (not anomaly_active)
            and (not anomaly_injected_once)
            and (anomaly_type != "none")
            and random_rotate_hit
        ):
            current_pitch = float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(pitch_joint_name)])
            current_yaw = float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(yaw_joint_name)])
            current_grip = float(get_grip_pos_mean(env, finger_joint_names))
            anomaly_anchor["base"] = base_pos.copy()
            anomaly_anchor["pitch"] = current_pitch
            anomaly_anchor["yaw"] = current_yaw
            anomaly_anchor["grip"] = current_grip

            # Use a fixed, user-specified anomaly type (no random type sampling).
            anomaly_kind = anomaly_type

            if anomaly_kind == "wrist_pitch":
                pitch_target = float(pitch_high if rng.random() < 0.5 else pitch_low)
                set_joint_scalar_clipped(env, pitch_joint_name, pitch_target)

            if anomaly_kind == "wrist_yaw":
                # One-direction yaw anomaly: always opposite to nominal task yaw direction.
                normal_is_high_side = yaw_target > 0.0
                yaw_anom_target = float((-np.pi / 2.0 + yaw_margin) if normal_is_high_side else (np.pi / 2.0 - yaw_margin))
                set_joint_scalar_clipped(env, yaw_joint_name, yaw_anom_target)

            if anomaly_kind == "base_z_high":
                base_pos = base_pos.copy()
                base_pos[2] = float(np.clip(base_z_high, base_z_low, base_z_high))
                set_base_pose(env, base_pos, base_quat)

            if anomaly_kind == "base_xy_shift":
                if rng.random() < 0.5:
                    base_pos = base_pos.copy()
                    base_pos[0] = float(base_x_high if rng.random() < 0.5 else base_x_low)
                else:
                    base_pos = base_pos.copy()
                    base_pos[1] = float(base_y_high if rng.random() < 0.5 else base_y_low)
                base_pos[0] = float(np.clip(base_pos[0], base_x_low, base_x_high))
                base_pos[1] = float(np.clip(base_pos[1], base_y_low, base_y_high))
                base_pos[2] = float(np.clip(base_pos[2], base_z_low, base_z_high))
                set_base_pose(env, base_pos, base_quat)

            if anomaly_kind == "grip_flip" and len(finger_joint_names) > 0:
                first_joint_id = env.sim.model.joint_name2id(finger_joint_names[0])
                grip_low, grip_high = env.sim.model.jnt_range[first_joint_id]
                grip_target = float(grip_high if rng.random() < 0.5 else grip_low)
                for finger_joint in finger_joint_names:
                    set_joint_scalar_clipped(env, finger_joint, grip_target)

            set_base_pose(env, base_pos, base_quat)
            env.sim.forward()

            anomaly_active = True
            anomaly_just_injected = True
            anomaly_injected_once = True
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
        ) and (not robot_can_contact)

        front_ready_exit = hand_is_on_front_side(
            hand_min_x=hand_min_x,
            hand_max_x=hand_max_x,
            obj_x=float(obj_now[0]),
            clearance=front_exit_clearance,
            front_sign=front_sign,
        ) and (not robot_can_contact)

        if front_latched:
            front_latched = front_ready_exit
        else:
            front_latched = front_ready_enter

        if anomaly_active:
            if anomaly_just_injected:
                # Keep the injected anomaly for exactly one step (instantaneous event), then recover gradually.
                zero_action = np.zeros(6, dtype=np.float32)
                states.append(get_compact_state(env, pitch_joint_name, yaw_joint_name, finger_joint_names))
                base_pos_seq.append(base_pos.copy())
                base_delta_seq.append(np.zeros(3, dtype=np.float32))

                env.sim.forward()

                # Critical: the anomaly frame itself should have no commanded motion.
                obs, reward, done, _ = env.step(zero_action)
                next_state_compact = get_compact_state(env, pitch_joint_name, yaw_joint_name, finger_joint_names)
                _g = float(next_state_compact[2])
                actions.append(np.array([float(next_state_compact[0]), float(next_state_compact[1]), _g, _g, _g, _g], dtype=np.float32))
                rewards.append(float(reward))
                dones.append(bool(done))

                if capture_images:
                    agent = env.sim.render(height=image_height, width=image_width, camera_name="agentview")
                    side = env.sim.render(height=image_height, width=image_width, camera_name="sideview")
                    if agent is not None:
                        if agent.ndim == 3 and agent.shape[2] >= 3:
                            agent = agent[:, :, :3]
                        agentview_seq.append(np.asarray(agent, dtype=np.uint8))
                    if side is not None:
                        if side.ndim == 3 and side.shape[2] >= 3:
                            side = side[:, :, :3]
                        sideview_seq.append(np.asarray(side, dtype=np.uint8))

                anomaly_just_injected = False
                step_idx += 1
                continue

            # Recovery-first mode: return to pre-anomaly anchor before resuming task logic.
            target = np.array(anomaly_anchor["base"], dtype=np.float64)
            desired_pitch = float(anomaly_anchor["pitch"])
            desired_yaw = float(anomaly_anchor["yaw"])
            wrist_recover_step = 0.035
            grip_recover_step = 0.06
            recovery_action = np.zeros(6, dtype=np.float32)

            # Gradual wrist recovery: move only a bounded step each control tick.
            pitch_cur = float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(pitch_joint_name)])
            yaw_cur = float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(yaw_joint_name)])
            pitch_next = pitch_cur + float(np.clip(desired_pitch - pitch_cur, -wrist_recover_step, wrist_recover_step))
            yaw_next = yaw_cur + float(np.clip(desired_yaw - yaw_cur, -wrist_recover_step, wrist_recover_step))
            recovery_action[0] = pitch_next
            recovery_action[1] = yaw_next

            recover_speed = base_speed
            next_base, base_delta = move_towards(base_pos, target, recover_speed)
            base_pos = next_base
            set_base_pose(env, base_pos, base_quat)
            set_joint_scalar_clipped(env, pitch_joint_name, pitch_next)
            set_joint_scalar_clipped(env, yaw_joint_name, yaw_next)

            # Gradual finger recovery, synchronized between simulator state and logged action.
            grip_cmds = []
            for finger_joint in finger_joint_names:
                cur = float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(finger_joint)])
                cmd = float(cur + np.clip(float(anomaly_anchor["grip"]) - cur, -grip_recover_step, grip_recover_step))
                set_joint_scalar_clipped(env, finger_joint, cmd)
                grip_cmds.append(cmd)
            recovery_action[2:] = float(np.mean(grip_cmds)) if len(grip_cmds) > 0 else 0.0

            env.sim.forward()

            states.append(get_compact_state(env, pitch_joint_name, yaw_joint_name, finger_joint_names))
            base_pos_seq.append(base_pos.copy())
            base_delta_seq.append(base_delta.astype(np.float32))

            obs, reward, done, _ = env.step(recovery_action)
            next_state_compact = get_compact_state(env, pitch_joint_name, yaw_joint_name, finger_joint_names)
            _g = float(next_state_compact[2])
            actions.append(np.array([float(next_state_compact[0]), float(next_state_compact[1]), _g, _g, _g, _g], dtype=np.float32))
            rewards.append(float(reward))
            dones.append(bool(done))

            if capture_images:
                agent = env.sim.render(height=image_height, width=image_width, camera_name="agentview")
                side = env.sim.render(height=image_height, width=image_width, camera_name="sideview")
                if agent is not None:
                    if agent.ndim == 3 and agent.shape[2] >= 3:
                        agent = agent[:, :, :3]
                    agentview_seq.append(np.asarray(agent, dtype=np.uint8))
                if side is not None:
                    if side.ndim == 3 and side.shape[2] >= 3:
                        side = side[:, :, :3]
                    sideview_seq.append(np.asarray(side, dtype=np.uint8))

            pitch_now = float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(pitch_joint_name)])
            yaw_now = float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(yaw_joint_name)])
            grip_now = float(get_grip_pos_mean(env, finger_joint_names))
            recovered = (
                np.linalg.norm(base_pos - anomaly_anchor["base"]) <= 0.006
                and abs(pitch_now - anomaly_anchor["pitch"]) <= 0.03
                and abs(yaw_now - anomaly_anchor["yaw"]) <= 0.03
                and abs(grip_now - anomaly_anchor["grip"]) <= 0.08
            )
            if recovered:
                anomaly_active = False
                anomaly_kind = "none"

            step_idx += 1
            continue

        if phase == "align":
            target = front_target
            action[0] = 0.0
            action[1] = 0.0
            if robot_can_contact or (not front_latched):
                if front_sign > 0:
                    retreat_x = max(base_pos[0] + 0.015, obj_now[0] + front_enter_clearance + 0.03)
                else:
                    retreat_x = min(base_pos[0] - 0.015, obj_now[0] - front_enter_clearance - 0.03)
                target = np.array([retreat_x, front_target[1], desired_align_z], dtype=np.float64)
            else:
                target = np.array([base_pos[0], front_target[1], desired_align_z], dtype=np.float64)

            if front_latched:
                front_ready_count += 1
            else:
                front_ready_count = 0
            if front_ready_count >= 8:
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
            if rotate_progress >= rotate_total_steps:
                phase = "forward"
                forward_progress = 0
                forward_anchor_pos = base_pos.copy()
                prelift_y_target = float(base_pos[1])
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
                prelift_y_target = float(base_pos[1])
            if forward_target_x is None:
                forward_target_x = float(obj_now[0] + front_sign * grasp_clearance)
                if front_sign > 0:
                    forward_target_x = min(forward_target_x, float(forward_anchor_pos[0] + forward_push_distance))
                else:
                    forward_target_x = max(forward_target_x, float(forward_anchor_pos[0] - forward_push_distance))
            target = np.array([forward_target_x, prelift_y_target, base_pos[2]], dtype=np.float64)
            action[0] = 0.0
            action[1] = 0.0
            palm_aligned = abs(float(obj_now[0]) - float(hand_center_x)) <= palm_center_tol
            near_forward_target = abs(base_pos[0] - forward_target_x) <= 0.0015
            if robot_can_contact or palm_aligned:
                target = np.array([base_pos[0], base_pos[1], desired_align_z], dtype=np.float64)
                action[2:] = hold_close_value
                phase = "grasp"
                grasp_progress = 0
            else:
                forward_progress += 1
                if near_forward_target or forward_progress >= forward_push_steps:
                    phase = "grasp"
                    grasp_progress = 0

        elif phase == "grasp":
            if prelift_y_target is None:
                prelift_y_target = float(base_pos[1])
            target = np.array([base_pos[0], prelift_y_target, grasp_target[2]], dtype=np.float64)
            action[0] = 0.0
            action[1] = 0.0
            grasp_alpha = min(1.0, grasp_progress / max(1, grasp_total_steps - 1))
            action[2:] = 0.22 + grasp_alpha * (hold_close_value - 0.22)
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
                phase = "lift"
                lift_progress = 0
                lift_anchor_pos = base_pos.copy()
                lift_target_pos = lift_anchor_pos.copy()
                lift_target_pos[2] = lift_anchor_pos[2] + lift_distance

        elif phase == "lift":
            if lift_anchor_pos is None:
                lift_anchor_pos = base_pos.copy()
            if lift_target_pos is None:
                lift_target_pos = lift_anchor_pos.copy()
                lift_target_pos[2] = lift_anchor_pos[2] + lift_distance
            target = lift_target_pos.copy()
            action[0] = 0.0
            action[1] = 0.0
            action[2:] = 0.0
            lift_progress += 1
            if np.linalg.norm(base_pos - lift_target_pos) <= 0.003 or lift_progress >= lift_total_steps:
                phase = "move"
                move_progress = 0
                move_target_pos = lift_target_pos.copy()
                near_obj_pos = get_object_pos_from_joint(env, near_target_joint)
                side_sign = -1.0 if float(base_pos[1] - near_obj_pos[1]) >= 0.0 else 1.0
                move_target_pos[0] = np.clip(float(near_obj_pos[0]), 0.02, 0.30)
                move_target_pos[1] = np.clip(float(near_obj_pos[1] + side_sign * place_offset), -0.18, 0.18)

        elif phase == "move":
            if move_target_pos is None:
                move_target_pos = base_pos.copy()
                near_obj_pos = get_object_pos_from_joint(env, near_target_joint)
                side_sign = -1.0 if float(base_pos[1] - near_obj_pos[1]) >= 0.0 else 1.0
                move_target_pos[0] = np.clip(float(near_obj_pos[0]), 0.02, 0.30)
                move_target_pos[1] = np.clip(float(near_obj_pos[1] + side_sign * place_offset), -0.18, 0.18)
            target = move_target_pos.copy()
            action[0] = 0.0
            action[1] = 0.0
            action[2:] = 0.0
            move_progress += 1
            if np.linalg.norm(base_pos[:2] - move_target_pos[:2]) <= 0.003 or move_progress >= active_move_total_steps:
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
            action[0] = 0.0
            action[1] = 0.0
            action[2:] = 0.0
            lower_progress += 1
            if abs(base_pos[2] - lower_target_pos[2]) <= 0.003 or lower_progress >= lower_total_steps:
                phase = "release"
                release_progress = 0

        elif phase == "release":
            if lower_target_pos is None:
                lower_target_pos = base_pos.copy()
            target = lower_target_pos.copy()
            action[0] = 0.0
            action[1] = 0.0
            action[2:] = release_open_value
            release_progress += 1

        else:
            target = grasp_target
            action[0] = 0.0
            action[1] = 0.0
            action[2:] = hold_close_value

        # Keep can always upright (translation allowed, tipping forbidden)
        if chosen_obj in ("can", "milk"):
            obj_now_stable = get_object_pos_from_joint(env, obj_joint)
            if phase in ("align", "rotate", "forward", "grasp", "grasp_lock"):
                pre_lift_z_cap = obj_z_start + 0.003
                obj_now_stable[2] = min(float(obj_now_stable[2]), pre_lift_z_cap)
            set_object_pos(env, obj_joint, obj_now_stable, quat=obj_upright_quat)

        if sticky_after_close:
            palm_pos = get_palm_pos(env)
            if palm_pos is not None:
                finger_mean = float(np.mean(action[2:]))
                obj_now_for_sticky = env.sim.data.qpos[obj_qpos_addr[0]:obj_qpos_addr[0] + 3].copy()
                if sticky_attached and finger_mean < -0.4:
                    sticky_attached = False
                if (not sticky_attached) and phase in ("grasp_lock", "lift", "move", "lower"):
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
        if z_err > 0.0:
            target = target.copy()
            target[2] -= min(0.03, 1.2 * z_err)

        # No-lift guard: before and during grasp, z is not allowed to increase
        if phase in ("align", "rotate", "forward", "grasp", "grasp_lock"):
            target = target.copy()
            target[2] = min(target[2], base_pos[2], desired_align_z)

        # Explicit wrist yaw control with front-side hard gate:
        # rotate only after whole hand is in front side and no contact
        if phase == "rotate":
            if front_latched:
                alpha = rotate_progress / max(1, rotate_total_steps)
                yaw_des = (1.0 - alpha) * yaw_initial + alpha * yaw_target
            else:
                yaw_des = yaw_initial
                action[1] = 0.0
        elif phase in ("forward", "grasp", "grasp_lock", "lift", "move", "lower", "release"):
            yaw_des = yaw_target
        else:
            yaw_des = yaw_initial

        step_speed = base_speed
        if phase == "forward":
            step_speed = forward_push_step_size

        if phase == "forward":
            next_base, base_delta = move_along_x_only(base_pos, forward_target_x, step_speed)
        elif phase in ("align", "rotate", "grasp", "grasp_lock"):
            next_base, base_delta = move_towards_xy_then_z(base_pos, target, step_speed)
        else:
            next_base, base_delta = move_towards(base_pos, target, step_speed)
        base_pos = next_base

        set_base_pose(env, base_pos, base_quat)
        set_joint_scalar(env, pitch_joint_name, pitch_initial)
        set_joint_scalar(env, yaw_joint_name, yaw_des)
        env.sim.forward()

        states.append(get_compact_state(env, pitch_joint_name, yaw_joint_name, finger_joint_names))
        base_pos_seq.append(base_pos.copy())
        base_delta_seq.append(base_delta.astype(np.float32))

        obs, reward, done, _ = env.step(action)
        next_state_compact = get_compact_state(env, pitch_joint_name, yaw_joint_name, finger_joint_names)
        _g = float(next_state_compact[2])
        actions.append(np.array([float(next_state_compact[0]), float(next_state_compact[1]), _g, _g, _g, _g], dtype=np.float32))
        rewards.append(float(reward))
        dones.append(bool(done))

        if capture_images:
            agent = env.sim.render(height=image_height, width=image_width, camera_name="agentview")
            side = env.sim.render(height=image_height, width=image_width, camera_name="sideview")
            if agent is not None:
                if agent.ndim == 3 and agent.shape[2] >= 3:
                    agent = agent[:, :, :3]
                agentview_seq.append(np.asarray(agent, dtype=np.uint8))
            if side is not None:
                if side.ndim == 3 and side.shape[2] >= 3:
                    side = side[:, :, :3]
                sideview_seq.append(np.asarray(side, dtype=np.uint8))

        # Stop once release phase is completed
        if phase == "release" and release_progress >= release_total_steps:
            break

        step_idx += 1

    obj_z_end = float(get_object_pos_from_joint(env, obj_joint)[2])
    # Success criterion: sticky grasp was activated at least once.
    lifted = int(sticky_ever_attached)

    result = {
        "actions": np.asarray(actions),
        "states": np.asarray(states),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "base_pos": np.asarray(base_pos_seq, dtype=np.float32),
        "base_delta": np.asarray(base_delta_seq, dtype=np.float32),
        "chosen_object": chosen_obj,
        "lift_success": lifted,
        "obj_z_start": obj_z_start,
        "obj_z_end": obj_z_end,
        "placed_near_object": near_target_name,
    }

    if capture_images and len(agentview_seq) == len(actions):
        result["agentview_images"] = np.asarray(agentview_seq, dtype=np.uint8)
    if capture_images and len(sideview_seq) == len(actions):
        result["sideview_images"] = np.asarray(sideview_seq, dtype=np.uint8)

    return result


def main():
    parser = argparse.ArgumentParser(description="Auto-cruise Hannes data collection with random object / base initialization")
    parser.add_argument("--task", type=str, default="Lift")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Random seed. If omitted, use current time-based seed")
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--target-object", type=str, default="can", help="Fixed target object to pick (default: can)")
    parser.add_argument("--place-near-object", type=str, default=None, help="Optional reference object name to place target beside")
    parser.add_argument("--place-offset", type=float, default=0.08, help="Side offset (meters) when placing beside reference object")
    parser.add_argument("--sticky-after-close", dest="sticky_after_close", action="store_true", help="Attach object to palm after close contact and release on opening")
    parser.add_argument("--no-sticky-after-close", dest="sticky_after_close", action="store_false", help="Disable sticky attachment behavior")
    parser.add_argument("--output-format", type=str, default="episode", choices=["episode", "autocruise"], help="Output file structure; rendered images are always captured and stored")
    parser.add_argument(
        "--anomaly-type",
        type=str,
        default="wrist_pitch",
        choices=["none", "wrist_pitch", "wrist_yaw", "base_z_high", "base_xy_shift", "grip_flip"],
        help="Fixed anomaly type to inject. Each injection applies this single anomaly only.",
    )
    parser.set_defaults(sticky_after_close=True)
    args = parser.parse_args()
    capture_images = True

    os.makedirs("hannes_demonstrations", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or os.path.join("hannes_demonstrations", f"hannes_{args.task}_autocruise_{timestamp}.hdf5")

    env = suite.make(
        args.task,
        robots="Hannes",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        control_freq=args.control_freq,
        horizon=args.horizon,
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
            "has_offscreen_renderer": False,
            "use_camera_obs": False,
            "control_freq": args.control_freq,
            "horizon": args.horizon,
            "ignore_done": True,
            "reward_shaping": True,
            "use_object_obs": False,
        },
    }

    run_seed = int(args.seed) if args.seed is not None else int(time.time_ns() % (2**32 - 1))
    rng = np.random.default_rng(run_seed)
    # Discover available object joints
    joint_candidates = {
        "milk": "milk_joint0",
        "can": "can_joint0",
        "lemon": "lemon_joint0",
        "bread": "bread_joint0",
        "hammer": "hammer_joint0",
        "potato": "potato_joint0",
    }
    valid_joints = {}
    for name, joint in joint_candidates.items():
        try:
            _ = env.sim.model.get_joint_qpos_addr(joint)
            valid_joints[name] = joint
        except Exception:
            pass

    if len(valid_joints) == 0:
        raise RuntimeError("No target object joints found (expected milk_joint0 or can_joint0).")

    print("=== Hannes Auto Cruise Collection ===")
    print("Output:", output)
    print("Episodes:", args.episodes)
    print("Objects:", list(valid_joints.keys()))
    print("Sticky after close:", bool(args.sticky_after_close))
    print("Place near object:", args.place_near_object)
    print("Output format:", args.output_format)
    print("Episode images:", True)
    print("Seed:", run_seed)
    print("Anomaly injection:", "exactly once per episode")
    print("Anomaly type:", args.anomaly_type)

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
                if args.target_object not in valid_joints:
                    raise ValueError(f"target object {args.target_object} not available, choose from {list(valid_joints.keys())}")
                chosen = args.target_object
            else:
                chosen = object_names[ep % len(object_names)]
            ep_result = run_episode(
                env,
                rng,
                chosen,
                valid_joints,
                sticky_after_close=args.sticky_after_close,
                place_near_object=args.place_near_object,
                place_offset=args.place_offset,
                capture_images=capture_images,
                anomaly_type=args.anomaly_type,
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
                if "agentview_images" in ep_result:
                    grp.create_dataset("agentview_images", data=ep_result["agentview_images"], compression="gzip")
                if "sideview_images" in ep_result:
                    grp.create_dataset("sideview_images", data=ep_result["sideview_images"], compression="gzip")
            else:
                ep_name = f"episode_{ep}"
                grp = f.create_group(ep_name)
                grp.create_dataset("actions", data=ep_result["actions"], compression="gzip")
                grp.create_dataset("rewards", data=ep_result["rewards"], compression="gzip")
                grp.create_dataset("dones", data=ep_result["dones"], compression="gzip")
                grp.create_dataset("states", data=ep_result["states"], compression="gzip")
                grp.create_dataset("base_pos", data=ep_result["base_pos"], compression="gzip")
                grp.create_dataset("base_delta", data=ep_result["base_delta"], compression="gzip")
                if "agentview_images" in ep_result:
                    grp.create_dataset("agentview_images", data=ep_result["agentview_images"], compression="gzip")
                if "sideview_images" in ep_result:
                    grp.create_dataset("sideview_images", data=ep_result["sideview_images"], compression="gzip")

            grp.attrs["num_samples"] = int(ep_result["actions"].shape[0])
            grp.attrs["model_file"] = env.sim.model.get_xml()
            grp.attrs["task"] = args.task
            grp.attrs["chosen_object"] = ep_result["chosen_object"]
            grp.attrs["placed_near_object"] = str(ep_result.get("placed_near_object", ""))
            grp.attrs["state_fields"] = json.dumps(["wrist_pitch", "wrist_yaw", "grip_pos_mean"])
            grp.attrs["action_fields"] = json.dumps([
                "wrist_pitch_cmd",
                "wrist_yaw_cmd",
                "forefinger_cmd",
                "midfinger_cmd",
                "ringfinger_cmd",
                "littlefinger_cmd",
            ])
            grp.attrs["lift_success"] = int(ep_result["lift_success"])
            grp.attrs["obj_z_start"] = float(ep_result["obj_z_start"])
            grp.attrs["obj_z_end"] = float(ep_result["obj_z_end"])

            total_steps += int(ep_result["actions"].shape[0])
            successes += int(ep_result["lift_success"])

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

    print("=== Done ===")
    print("Saved:", output)
    print(f"Success: {successes}/{args.episodes}")


if __name__ == "__main__":
    main()
