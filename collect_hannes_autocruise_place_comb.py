import argparse
import json
import os
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
    continue_from_current=False,
    pre_raise_distance=0.0,
    pre_raise_steps=0,
    min_target_near_init_dist=0.10,
    record_buffer=None,
):
    if continue_from_current:
        env.sim.forward()
        obs = {}
    else:
        obs = env.reset()

    base_quat = np.array([0.707, 0.0, 0.0, -0.707], dtype=np.float64)

    obj_joint = joints[chosen_obj]
    obj_qpos_addr = env.sim.model.get_joint_qpos_addr(obj_joint)
    hand_site_ids = get_hand_site_ids(env)

    near_target_joint_init = None
    near_target_name_init = None
    if place_near_object and place_near_object in joints and place_near_object != chosen_obj:
        near_target_joint_init = joints[place_near_object]
        near_target_name_init = place_near_object
    if near_target_joint_init is None:
        for name in ("milk", "can", "hammer", "lemon", "bread"):
            if name in joints and name != chosen_obj:
                near_target_joint_init = joints[name]
                near_target_name_init = name
                break

    obj_initial = get_object_pos_from_joint(env, obj_joint)
    obj_initial_quat = get_object_quat_from_joint(env, obj_joint)
    if obj_initial is None:
        raise RuntimeError(f"Cannot read object joint {obj_joint}")
    if obj_initial_quat is None:
        obj_initial_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    obj_upright_quat = np.array(obj_initial_quat, dtype=np.float64)

    # Randomize target object position on tabletop neighborhood while keeping it far from the near object.
    obj_random = obj_initial.copy()
    near_pos_init = None
    if near_target_joint_init is not None:
        near_pos_init = get_object_pos_from_joint(env, near_target_joint_init)

    sampled_ok = False
    for _ in range(80):
        cand_x = np.clip(obj_initial[0] + rng.uniform(-0.06, 0.06), 0.02, 0.28)
        cand_y = np.clip(obj_initial[1] + rng.uniform(-0.08, 0.08), -0.14, 0.14)
        if near_pos_init is not None:
            dxy = np.linalg.norm(np.array([cand_x, cand_y]) - near_pos_init[:2])
            if dxy < float(min_target_near_init_dist):
                continue
        obj_random[0] = cand_x
        obj_random[1] = cand_y
        sampled_ok = True
        break

    if (not sampled_ok) and near_pos_init is not None:
        push = float(min_target_near_init_dist)
        obj_random[0] = np.clip(float(near_pos_init[0]), 0.02, 0.28)
        if float(near_pos_init[1]) >= 0.0:
            obj_random[1] = np.clip(float(near_pos_init[1] - push), -0.14, 0.14)
        else:
            obj_random[1] = np.clip(float(near_pos_init[1] + push), -0.14, 0.14)

    set_object_pos(env, obj_joint, obj_random, quat=obj_upright_quat)

    # Keep non-target objects at their environment-sampled positions.

    # Random initial placement with safe z floor to avoid spawning below tabletop
    # Use a slightly higher align z for task2 to avoid over-lowering during second align.
    align_z_offset = 0.04 if continue_from_current else 0.06
    desired_align_z = float(obj_random[2] - align_z_offset)
    if continue_from_current:
        base_addr = env.sim.model.get_joint_qpos_addr("robot0_base_free")
        base_pos = env.sim.data.qpos[base_addr[0]:base_addr[0] + 3].copy()
    else:
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

    if record_buffer is None:
        actions = []
        states = []
        rewards = []
        dones = []
        base_pos_seq = []
        base_delta_seq = []
        agentview_seq = []
        sideview_seq = []
    else:
        actions = record_buffer["actions"]
        states = record_buffer["states"]
        rewards = record_buffer["rewards"]
        dones = record_buffer["dones"]
        base_pos_seq = record_buffer["base_pos"]
        base_delta_seq = record_buffer["base_delta"]
        agentview_seq = record_buffer["agentview_images"]
        sideview_seq = record_buffer["sideview_images"]
    step_start = len(actions)

    obj_z_start = float(obj_pos[2])
    base_initial_z = float(base_pos[2])
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
    yaw_dir = -front_sign
    yaw_margin = 0.01
    yaw_target = float((np.pi / 2.0 - yaw_margin) if yaw_dir > 0 else (-np.pi / 2.0 + yaw_margin))
    rotate_total_steps = 60
    forward_push_steps = 16
    forward_push_distance = 0.084
    forward_push_step_size = forward_push_distance / max(1, forward_push_steps)
    palm_center_tol = 0.018
    grasp_total_steps = 45
    hold_close_value = 0.70 if chosen_obj == "can" else 0.55
    grasp_lock_steps = 16
    release_open_value = -0.75
    lift_distance = float(rng.uniform(0.16, 0.21))
    lift_total_steps = 26
    move_total_steps = 32
    near_move_total_steps = 56
    lower_total_steps = 26
    release_total_steps = 18

    near_target_joint = near_target_joint_init
    near_target_name = near_target_name_init
    if near_target_joint is None:
        raise RuntimeError("No valid reference object to place near. Provide --place-near-object.")
    active_move_total_steps = near_move_total_steps if near_target_joint is not None else move_total_steps

    phase = "align"
    rotate_progress = 0
    forward_progress = 0
    forward_anchor_pos = None
    forward_target_x = None
    prelift_y_target = None
    grasp_hold_z = None
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

    # Optional transition: raise up before starting the next target task.
    if continue_from_current and pre_raise_distance > 0.0 and pre_raise_steps > 0:
        raise_target = base_pos.copy()
        raise_target[2] = base_pos[2] + float(pre_raise_distance)
        for _ in range(int(pre_raise_steps)):
            action = np.zeros(6, dtype=np.float32)
            action[0] = 0.0
            action[1] = 0.0
            action[2:] = release_open_value

            next_base, base_delta = move_towards(base_pos, raise_target, base_speed)
            base_pos = next_base
            set_base_pose(env, base_pos, base_quat)
            set_joint_scalar(env, pitch_joint_name, pitch_initial)
            set_joint_scalar(env, yaw_joint_name, yaw_initial)
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

            if np.linalg.norm(base_pos - raise_target) <= 0.003:
                break
        # Re-anchor height lock to post-raise EEF height for stable second-task grasping.
        desired_eef_z = float(get_eef_pos({}, env)[2])

    while True:
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
                grasp_hold_z = float(base_pos[2])
            else:
                forward_progress += 1
                if near_forward_target or forward_progress >= forward_push_steps:
                    phase = "grasp"
                    grasp_progress = 0
                    grasp_hold_z = float(base_pos[2])

        elif phase == "grasp":
            if prelift_y_target is None:
                prelift_y_target = float(base_pos[1])
            if grasp_hold_z is None:
                grasp_hold_z = float(base_pos[2])
            target = np.array([base_pos[0], prelift_y_target, grasp_hold_z], dtype=np.float64)
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
            if grasp_hold_z is None:
                grasp_hold_z = float(base_pos[2])
            target = np.array([base_pos[0], prelift_y_target, grasp_hold_z], dtype=np.float64)
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
                move_target_pos[0] = np.clip(float(near_obj_pos[0]), 0.02, 0.30)
                move_target_pos[1] = np.clip(float(near_obj_pos[1] + place_offset), -0.18, 0.18)

        elif phase == "move":
            if move_target_pos is None:
                move_target_pos = base_pos.copy()
                near_obj_pos = get_object_pos_from_joint(env, near_target_joint)
                move_target_pos[0] = np.clip(float(near_obj_pos[0]), 0.02, 0.30)
                move_target_pos[1] = np.clip(float(near_obj_pos[1] + place_offset), -0.18, 0.18)
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
        if z_err > 0.0 and phase in ("align", "rotate", "forward"):
            target = target.copy()
            target[2] -= min(0.03, 1.2 * z_err)

        # No-lift guard: before and during grasp, z is not allowed to increase
        if phase in ("align", "rotate"):
            target = target.copy()
            target[2] = min(target[2], base_pos[2], desired_align_z)
        elif phase in ("forward", "grasp", "grasp_lock"):
            target = target.copy()
            target[2] = min(target[2], base_pos[2])

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

    obj_z_end = float(get_object_pos_from_joint(env, obj_joint)[2])
    lifted = int(sticky_ever_attached)

    step_end = len(actions)
    result = {
        "actions": np.asarray(actions[step_start:step_end]),
        "states": np.asarray(states[step_start:step_end]),
        "rewards": np.asarray(rewards[step_start:step_end], dtype=np.float32),
        "dones": np.asarray(dones[step_start:step_end], dtype=np.bool_),
        "base_pos": np.asarray(base_pos_seq[step_start:step_end], dtype=np.float32),
        "base_delta": np.asarray(base_delta_seq[step_start:step_end], dtype=np.float32),
        "step_start": int(step_start),
        "step_end": int(step_end),
        "chosen_object": chosen_obj,
        "lift_success": lifted,
        "obj_z_start": obj_z_start,
        "obj_z_end": obj_z_end,
        "base_initial_z": base_initial_z,
        "placed_near_object": near_target_name,
    }

    if capture_images and len(agentview_seq) >= step_end:
        result["agentview_images"] = np.asarray(agentview_seq[step_start:step_end], dtype=np.uint8)
    if capture_images and len(sideview_seq) >= step_end:
        result["sideview_images"] = np.asarray(sideview_seq[step_start:step_end], dtype=np.uint8)

    return result


def main():
    parser = argparse.ArgumentParser(description="Auto-cruise Hannes data collection with random object / base initialization")
    parser.add_argument("--task", type=str, default="Lift")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--target-object", type=str, default="can", help="Deprecated in combo mode; kept for backward compatibility")
    parser.add_argument("--place-near-object", type=str, default=None, help="Optional reference object name to place target beside")
    parser.add_argument("--near-objects", type=str, default=None, help="Comma-separated near objects for two targets, e.g. bread,hammer")
    parser.add_argument("--target-order", type=str, default="milk,can", help="Two-target order: milk,can or can,milk")
    parser.add_argument("--inter-task-raise-distance", type=float, default=0.08, help="Raise distance between first and second target tasks")
    parser.add_argument("--inter-task-raise-steps", type=int, default=16, help="Raise steps between first and second target tasks")
    parser.add_argument("--target-near-init-min-dist", type=float, default=0.10, help="Minimum XY distance between target and near object at task initialization")
    parser.add_argument("--place-offset", type=float, default=0.005, help="Offset from near object (meters); near-overlap by default")
    parser.add_argument("--sticky-after-close", dest="sticky_after_close", action="store_true", help="Attach object to palm after close contact and release on opening")
    parser.add_argument("--no-sticky-after-close", dest="sticky_after_close", action="store_false", help="Disable sticky attachment behavior")
    parser.add_argument("--output-format", type=str, default="episode", choices=["episode", "autocruise"], help="Output file structure")
    parser.add_argument("--episode-images", dest="episode_images", action="store_true", help="Store agentview / sideview images in episode output")
    parser.add_argument("--no-episode-images", dest="episode_images", action="store_false", help="Do not store images in episode output")
    parser.set_defaults(sticky_after_close=True)
    parser.set_defaults(episode_images=True)
    args = parser.parse_args()

    os.makedirs("hannes_demonstrations", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or os.path.join("hannes_demonstrations", f"hannes_{args.task}_autocruise_{timestamp}.hdf5")

    env = suite.make(
        args.task,
        robots="Hannes",
        has_renderer=False,
        has_offscreen_renderer=bool(args.output_format == "episode" and args.episode_images),
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

    rng = np.random.default_rng(args.seed)
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
    print("Near objects:", args.near_objects)
    print("Target order:", args.target_order)
    print("Output format:", args.output_format)
    print("Episode images:", bool(args.episode_images))

    with h5py.File(output, "w") as f:
        if args.output_format == "autocruise":
            data = f.create_group("data")
            data.attrs["env_args"] = json.dumps(env_meta, indent=4)
        else:
            f.attrs["env_args"] = json.dumps(env_meta, indent=4)

        total_steps = 0
        successes = 0

        object_names = list(valid_joints.keys())

        raw_target_order = [x.strip() for x in str(args.target_order).split(",") if x.strip()]
        if len(raw_target_order) != 2:
            raise ValueError("--target-order must contain exactly two objects, e.g. milk,can")
        if set(raw_target_order) != {"milk", "can"}:
            raise ValueError("--target-order must be exactly milk and can, in either order")
        for t in raw_target_order:
            if t not in valid_joints:
                raise ValueError(f"target object {t} not available, choose from {list(valid_joints.keys())}")

        raw_near_objects = None
        if args.near_objects:
            raw_near_objects = [x.strip() for x in str(args.near_objects).split(",") if x.strip()]
            if len(raw_near_objects) != 2:
                raise ValueError("--near-objects must contain exactly two names, e.g. bread,hammer")

        for ep in range(args.episodes):
            target_seq = list(raw_target_order)
            near_seq = []
            if raw_near_objects is not None:
                near_seq = list(raw_near_objects)
            else:
                for t in target_seq:
                    if args.place_near_object and args.place_near_object in valid_joints and args.place_near_object != t:
                        near_seq.append(args.place_near_object)
                    else:
                        fallback = None
                        for name in ("milk", "can", "hammer", "lemon", "bread", "potato"):
                            if name in valid_joints and name != t:
                                fallback = name
                                break
                        near_seq.append(fallback)

            ep_buffer = {
                "actions": [],
                "states": [],
                "rewards": [],
                "dones": [],
                "base_pos": [],
                "base_delta": [],
                "agentview_images": [],
                "sideview_images": [],
            }

            first_result = run_episode(
                env,
                rng,
                target_seq[0],
                valid_joints,
                sticky_after_close=args.sticky_after_close,
                place_near_object=near_seq[0],
                place_offset=args.place_offset,
                min_target_near_init_dist=args.target_near_init_min_dist,
                capture_images=bool(args.output_format == "episode" and args.episode_images),
                record_buffer=ep_buffer,
            )

            # Bridge phase: raise to the initial base height of task0, then start task1.
            base_addr = env.sim.model.get_joint_qpos_addr("robot0_base_free")
            base_now = env.sim.data.qpos[base_addr[0]:base_addr[0] + 3].copy()
            raise_target_z = float(first_result["base_initial_z"])
            raise_distance = max(0.0, raise_target_z - float(base_now[2]))
            min_raise_steps = 0
            if raise_distance > 0.0:
                min_raise_steps = int(np.ceil(raise_distance / 0.006)) + 2
            raise_steps = max(int(args.inter_task_raise_steps), min_raise_steps)

            second_result = run_episode(
                env,
                rng,
                target_seq[1],
                valid_joints,
                sticky_after_close=args.sticky_after_close,
                place_near_object=near_seq[1],
                place_offset=args.place_offset,
                min_target_near_init_dist=args.target_near_init_min_dist,
                capture_images=bool(args.output_format == "episode" and args.episode_images),
                continue_from_current=True,
                pre_raise_distance=raise_distance,
                pre_raise_steps=raise_steps,
                record_buffer=ep_buffer,
            )

            ep_result = {
                "actions": np.asarray(ep_buffer["actions"], dtype=np.float32),
                "states": np.asarray(ep_buffer["states"], dtype=np.float32),
                "rewards": np.asarray(ep_buffer["rewards"], dtype=np.float32),
                "dones": np.asarray(ep_buffer["dones"], dtype=np.bool_),
                "base_pos": np.asarray(ep_buffer["base_pos"], dtype=np.float32),
                "base_delta": np.asarray(ep_buffer["base_delta"], dtype=np.float32),
                "chosen_object": ",".join(target_seq),
                "placed_near_object": ",".join([str(near_seq[0]), str(near_seq[1])]),
                "lift_success": int(first_result["lift_success"] and second_result["lift_success"]),
                "obj_z_start": float(first_result["obj_z_start"]),
                "obj_z_end": float(second_result["obj_z_end"]),
                "lift_success_task0": int(first_result["lift_success"]),
                "lift_success_task1": int(second_result["lift_success"]),
                "task0_step_end": int(first_result["step_end"]),
                "task1_step_start": int(second_result["step_start"]),
            }
            if bool(args.output_format == "episode" and args.episode_images):
                if len(ep_buffer["agentview_images"]) == len(ep_buffer["actions"]):
                    ep_result["agentview_images"] = np.asarray(ep_buffer["agentview_images"], dtype=np.uint8)
                if len(ep_buffer["sideview_images"]) == len(ep_buffer["actions"]):
                    ep_result["sideview_images"] = np.asarray(ep_buffer["sideview_images"], dtype=np.uint8)

            if args.output_format == "autocruise":
                demo_name = f"demo_{ep:06d}"
                grp = data.create_group(demo_name)
                grp.create_dataset("actions", data=ep_result["actions"], compression="gzip")
                grp.create_dataset("states", data=ep_result["states"], compression="gzip")
                grp.create_dataset("rewards", data=ep_result["rewards"], compression="gzip")
                grp.create_dataset("dones", data=ep_result["dones"], compression="gzip")
                grp.create_dataset("base_pos", data=ep_result["base_pos"], compression="gzip")
                grp.create_dataset("base_delta", data=ep_result["base_delta"], compression="gzip")
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
            grp.attrs["target_order"] = json.dumps(target_seq)
            grp.attrs["near_objects"] = json.dumps([str(near_seq[0]), str(near_seq[1])])
            grp.attrs["lift_success_task0"] = int(ep_result.get("lift_success_task0", 0))
            grp.attrs["lift_success_task1"] = int(ep_result.get("lift_success_task1", 0))
            grp.attrs["task0_step_end"] = int(ep_result.get("task0_step_end", 0))
            grp.attrs["task1_step_start"] = int(ep_result.get("task1_step_start", 0))
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
