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
    max_steps=260,
    base_speed=0.006,
    capture_images=False,
    image_height=512,
    image_width=512,
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

    obj_random = obj_initial.copy()
    obj_random[0] = np.clip(obj_initial[0] + rng.uniform(-0.06, 0.06), 0.02, 0.28)
    obj_random[1] = np.clip(obj_initial[1] + rng.uniform(-0.08, 0.08), -0.14, 0.14)
    set_object_pos(env, obj_joint, obj_random, quat=obj_upright_quat)

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

    obj_pos = get_object_pos_from_joint(env, obj_joint)
    front_sign = 1.0

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
    yaw_initial = float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(yaw_joint_name)])
    pitch_initial = float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(pitch_joint_name)])
    yaw_dir = -front_sign
    yaw_margin = 0.01
    yaw_target = float((yaw_high - yaw_margin) if yaw_dir > 0 else (yaw_low + yaw_margin))
    rotate_total_steps = 60
    forward_push_steps = 16
    forward_push_distance = 0.084
    forward_push_step_size = forward_push_distance / max(1, forward_push_steps)
    palm_center_tol = 0.018
    grasp_total_steps = 45
    hold_close_value = 0.55
    grasp_lock_steps = 16

    phase = "align"
    rotate_progress = 0
    forward_progress = 0
    forward_anchor_pos = None
    forward_target_x = None
    prelift_y_target = None
    grasp_progress = 0
    grasp_lock_progress = 0
    hold_progress = 0
    front_ready_count = 0
    rotate_anchor_x = None
    rotate_anchor_pos = None
    front_enter_clearance = front_clearance
    front_exit_clearance = max(0.07, front_clearance - 0.02)
    front_latched = False

    for t in range(max_steps):
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
        elif phase in ("forward", "grasp", "grasp_lock", "hold"):
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
        action_log = action.copy()
        action_log[0] = float(pitch_initial)
        action_log[1] = float(yaw_des)
        actions.append(action_log)
        base_pos_seq.append(base_pos.copy())
        base_delta_seq.append(base_delta.astype(np.float32))

        obs, reward, done, _ = env.step(action)
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

        # Trim repetitive tail: once we have stayed in hold phase long enough, stop the demo early
        if phase == "hold" and hold_progress >= 25:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Lift")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--target-object", type=str, default="can")
    parser.add_argument("--output-format", type=str, default="episode", choices=["episode", "autocruise"], help="Output file structure; episode output stores rendered views by default")
    args = parser.parse_args()
    is_episode_output = args.output_format == "episode"

    os.makedirs("hannes_demonstrations", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or os.path.join("hannes_demonstrations", f"hannes_{args.task}_autocruise_{timestamp}.hdf5")

    env = suite.make(
        args.task,
        robots="Hannes",
        has_renderer=False,
        has_offscreen_renderer=bool(is_episode_output),
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
                capture_images=bool(is_episode_output),
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
                grp.create_dataset("states", data=ep_result["states"], compression="gzip")
                grp.create_dataset("rewards", data=ep_result["rewards"], compression="gzip")
                grp.create_dataset("dones", data=ep_result["dones"], compression="gzip")
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
