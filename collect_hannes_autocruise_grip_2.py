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
    capture_images=False,
    image_height=512,
    image_width=512,
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

    # Explicit staged routine (top-down for robust multi-object coverage):
    # 1) route above target object
    # 2) descend to grasp height
    # 3) press wrist down
    # 4) close fingers (grasp)
    # 5) lift
    # 6) move to place target
    # 7) lower
    # 8) release
    hover_height = 0.13
    grasp_height_offset = -0.01
    # Palm-centric alignment offsets in XY; keep a deliberate lateral mismatch instead of exact centering.
    palm_align_bias_x = {
        "milk": 0.040,
        "can": 0.040,
        "lemon": 0.085,
        "bread": 0.075,
        "hammer": 0.085,
        "potato": 0.055,
    }
    palm_align_bias_y = {
        "milk": 0.020,
        "can": 0.020,
        "lemon": 0.045,
        "bread": 0.040,
        "hammer": 0.045,
        "potato": 0.030,
    }
    active_align_bias_x = float(palm_align_bias_x.get(chosen_obj, 0.050))
    active_align_bias_y = float(palm_align_bias_y.get(chosen_obj, 0.025))

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
    finger_close_joint_max = {}
    for joint_name in finger_joint_names:
        joint_id = env.sim.model.joint_name2id(joint_name)
        _joint_low, joint_high = env.sim.model.jnt_range[joint_id]
        finger_close_joint_max[joint_name] = float(joint_high)
    yaw_joint_id = env.sim.model.joint_name2id(yaw_joint_name)
    yaw_low, yaw_high = env.sim.model.jnt_range[yaw_joint_id]
    pitch_joint_id = env.sim.model.joint_name2id(pitch_joint_name)
    pitch_low, pitch_high = env.sim.model.jnt_range[pitch_joint_id]
    yaw_initial = -np.pi/2
    set_joint_scalar(env, yaw_joint_name, yaw_initial)
    pitch_initial = float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(pitch_joint_name)])
    yaw_target = yaw_initial
    # For this setup, increasing pitch corresponds to pressing wrist downward.
    # Use a larger downward press to make the motion clearly visible.
    pitch_press_target = float(np.clip(pitch_initial + 0.50, pitch_low + 0.01, pitch_high - 0.01))
    route_total_steps = 120
    route_min_steps = 45
    wrist_press_steps = 16
    wrist_hold_steps = 30
    micro_lower_distance = 0.050
    micro_lower_steps = 10
    finger_close_steps = 18
    finger_close_start_value = -0.10
    finger_close_end_value = 1.00
    finger_close_hold_steps = 12
    release_open_value = -0.75

    # Dumb non-tracking mode: once routing finishes, freeze targets and stop following object drift.
    freeze_targets_after_route = True

    phase = "route_above"
    route_progress = 0
    wrist_press_progress = 0
    wrist_hold_progress = 0
    micro_lower_progress = 0
    finger_close_progress = 0
    finger_close_hold_progress = 0
    frozen_above_target = None
    frozen_micro_lower_target = None
    sticky_attached = False
    sticky_offset = np.zeros(3, dtype=np.float64)

    while True:
        action = np.zeros(6, dtype=np.float32)
        obj_now = get_object_pos_from_joint(env, obj_joint)
        robot_can_contact = is_object_in_contact(env, chosen_obj)
        palm_pos = get_palm_pos(env)
        if palm_pos is None:
            palm_pos = get_eef_pos(obs, env)
        palm_xy_offset = np.asarray(palm_pos[:2], dtype=np.float64) - np.asarray(base_pos[:2], dtype=np.float64)
        # Convert desired palm-over-object point into base target XY.
        # Apply deliberate XY bias instead of exact palm/object centering.
        align_x = float(obj_now[0] - palm_xy_offset[0] + active_align_bias_x)
        align_y = float(obj_now[1] - palm_xy_offset[1] + active_align_bias_y)
        align_x = float(np.clip(align_x, 0.02, 0.30))
        align_y = float(np.clip(align_y, -0.18, 0.18))

        above_target = np.array([align_x, align_y, obj_now[2] + hover_height], dtype=np.float64)
        micro_lower_target = above_target.copy()
        micro_lower_target[2] = max(float(obj_now[2] + 0.02), float(above_target[2] - micro_lower_distance))
        grasp_target = np.array([align_x, align_y, obj_now[2] + grasp_height_offset], dtype=np.float64)

        pitch_des = pitch_initial
        yaw_des = yaw_target

        if phase == "route_above":
            target = above_target
            action[0] = 0.0
            action[1] = 0.0
            action[2:] = release_open_value
            route_progress += 1
            reached_above = np.linalg.norm(base_pos - above_target) <= 0.006
            if (reached_above and route_progress >= route_min_steps) or route_progress >= route_total_steps:
                if freeze_targets_after_route:
                    frozen_above_target = above_target.copy()
                    frozen_micro_lower_target = micro_lower_target.copy()
                phase = "wrist_press"
                wrist_press_progress = 0

        elif phase == "wrist_press":
            # Keep base at above-target and perform wrist-only press.
            target = frozen_above_target.copy() if frozen_above_target is not None else above_target.copy()
            press_alpha = min(1.0, wrist_press_progress / max(1, wrist_press_steps - 1))
            pitch_des = float((1.0 - press_alpha) * pitch_initial + press_alpha * pitch_press_target)
            action[0] = pitch_des
            action[1] = 0.0
            action[2:] = release_open_value
            wrist_press_progress += 1
            if wrist_press_progress >= wrist_press_steps:
                phase = "wrist_hold"
                wrist_hold_progress = 0

        elif phase == "wrist_hold":
            # Hold the pressed wrist pose for several frames so it is visible in playback.
            target = frozen_above_target.copy() if frozen_above_target is not None else above_target.copy()
            pitch_des = pitch_press_target
            action[0] = pitch_press_target
            action[1] = 0.0
            action[2:] = release_open_value
            wrist_hold_progress += 1
            if wrist_hold_progress >= wrist_hold_steps:
                phase = "micro_lower"
                micro_lower_progress = 0

        elif phase == "micro_lower":
            target = frozen_micro_lower_target.copy() if frozen_micro_lower_target is not None else micro_lower_target.copy()
            pitch_des = pitch_press_target
            action[0] = pitch_press_target
            action[1] = 0.0
            action[2:] = release_open_value
            micro_lower_progress += 1
            ref_micro_lower_target = frozen_micro_lower_target if frozen_micro_lower_target is not None else micro_lower_target
            if abs(base_pos[2] - ref_micro_lower_target[2]) <= 0.003 or micro_lower_progress >= micro_lower_steps:
                phase = "finger_close"
                finger_close_progress = 0

        elif phase == "finger_close":
            target = frozen_micro_lower_target.copy() if frozen_micro_lower_target is not None else micro_lower_target.copy()
            pitch_des = pitch_press_target
            close_alpha = min(1.0, finger_close_progress / max(1, finger_close_steps - 1))
            finger_close_value = float((1.0 - close_alpha) * finger_close_start_value + close_alpha * finger_close_end_value)
            action[0] = pitch_press_target
            action[1] = 0.0
            action[2:] = finger_close_value
            finger_close_progress += 1
            if finger_close_progress >= finger_close_steps:
                phase = "finger_close_hold"
                finger_close_hold_progress = 0

        elif phase == "finger_close_hold":
            target = frozen_micro_lower_target.copy() if frozen_micro_lower_target is not None else micro_lower_target.copy()
            pitch_des = pitch_press_target
            action[0] = pitch_press_target
            action[1] = 0.0
            action[2:] = finger_close_end_value
            finger_close_hold_progress += 1
            if finger_close_hold_progress >= finger_close_hold_steps:
                phase = "done"
        else:
            raise RuntimeError(f"Unexpected phase: {phase}")

        # Keep can always upright (translation allowed, tipping forbidden)
        if chosen_obj in ("can", "milk"):
            obj_now_stable = get_object_pos_from_joint(env, obj_joint)
            if phase in ("route_above", "wrist_press", "grasp", "grasp_lock"):
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
                if (not sticky_attached) and phase in ("finger_close", "finger_close_hold"):
                    near_palm = np.linalg.norm(obj_now_for_sticky - palm_pos) < 0.05
                    if robot_can_contact or near_palm:
                        sticky_offset = obj_now_for_sticky - palm_pos
                        sticky_attached = True
                if sticky_attached:
                    set_object_pos(env, obj_joint, palm_pos + sticky_offset, quat=obj_upright_quat)

        step_speed = base_speed
        if phase == "wrist_press":
            step_speed = base_speed * 0.85

        if phase == "route_above":
            next_base, base_delta = move_towards_xy_then_z(base_pos, target, step_speed)
        else:
            next_base, base_delta = move_towards(base_pos, target, step_speed)
        base_pos = next_base

        set_base_pose(env, base_pos, base_quat)
        set_joint_scalar(env, pitch_joint_name, pitch_des)
        set_joint_scalar(env, yaw_joint_name, yaw_des)
        env.sim.forward()

        states.append(get_compact_state(env, pitch_joint_name, yaw_joint_name, finger_joint_names))
        base_pos_seq.append(base_pos.copy())
        base_delta_seq.append(base_delta.astype(np.float32))

        obs, reward, done, _ = env.step(action)
        rewards.append(float(reward))
        dones.append(bool(done))

        if phase in ("finger_close", "finger_close_hold") and len(finger_close_joint_max) > 0:
            for joint_name, joint_high in finger_close_joint_max.items():
                set_joint_scalar(env, joint_name, joint_high)
            env.sim.forward()

        next_state_compact = get_compact_state(env, pitch_joint_name, yaw_joint_name, finger_joint_names)
        _g = float(next_state_compact[2])
        actions.append(np.array([float(next_state_compact[0]), float(next_state_compact[1]), _g, _g, _g, _g], dtype=np.float32))

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

        # Route + wrist-press only mode: stop immediately after requested stages complete.
        if phase == "done":
            break

    obj_z_end = float(get_object_pos_from_joint(env, obj_joint)[2])
    lifted = int((obj_z_end - obj_z_start) > 0.03)

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
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--target-object", type=str, default="can", help="Fixed target object to pick (default: can)")
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
                capture_images=bool(args.output_format == "episode" and args.episode_images),
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
