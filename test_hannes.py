import argparse
import os
from datetime import datetime

import imageio
import numpy as np
import robosuite as suite
import cv2

from openpi_client import websocket_client_policy as _websocket_client_policy


def set_base_pose(env, position, quaternion):
    qpos_addr = env.sim.model.get_joint_qpos_addr("robot0_base_free")
    qvel_addr = env.sim.model.get_joint_qvel_addr("robot0_base_free")
    env.sim.data.qpos[qpos_addr[0] : qpos_addr[0] + 3] = position
    env.sim.data.qpos[qpos_addr[0] + 3 : qpos_addr[0] + 7] = quaternion
    env.sim.data.qvel[qvel_addr[0] : qvel_addr[0] + 6] = 0.0


def get_flat_state(env):
    return np.array(env.sim.get_state().flatten(), dtype=np.float64)


def get_object_pos_from_joint(env, joint_name):
    qpos_addr = env.sim.model.get_joint_qpos_addr(joint_name)
    if not isinstance(qpos_addr, tuple):
        return None
    return env.sim.data.qpos[qpos_addr[0] : qpos_addr[0] + 3].copy()


def get_object_quat_from_joint(env, joint_name):
    qpos_addr = env.sim.model.get_joint_qpos_addr(joint_name)
    if not isinstance(qpos_addr, tuple):
        return None
    return env.sim.data.qpos[qpos_addr[0] + 3 : qpos_addr[0] + 7].copy()


def set_object_pos(env, joint_name, xyz, quat=None):
    qpos_addr = env.sim.model.get_joint_qpos_addr(joint_name)
    if not isinstance(qpos_addr, tuple):
        return
    env.sim.data.qpos[qpos_addr[0] : qpos_addr[0] + 3] = xyz
    if quat is not None:
        env.sim.data.qpos[qpos_addr[0] + 3 : qpos_addr[0] + 7] = quat
    qvel_addr = env.sim.model.get_joint_qvel_addr(joint_name)
    if isinstance(qvel_addr, tuple):
        env.sim.data.qvel[qvel_addr[0] : qvel_addr[0] + 6] = 0.0


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


def run_episode(
    env,
    rng,
    chosen_obj,
    joints,
    policy,
    prompt,
    max_steps=260,
    base_speed=0.006,
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
    # Always treat the "+x" side as the front side of the object.
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
    agent_frames = []

    obj_z_start = float(obj_pos[2])
    desired_eef_z = float(get_eef_pos(obs, env)[2])

    rotate_total_steps = 60
    forward_push_steps = 16
    forward_push_distance = 0.084
    forward_push_step_size = forward_push_distance / max(1, forward_push_steps)
    palm_center_tol = 0.018
    grasp_total_steps = 45
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
    rotate_anchor_pos = None
    front_enter_clearance = front_clearance
    front_exit_clearance = max(0.07, front_clearance - 0.02)
    front_latched = False

    for t in range(max_steps):
        # For visualization, flip agentview vertically so the saved video looks upright,
        # but keep the raw image for policy input below.
        agent_frames.append(obs["agentview_image"][::-1])

        # Match dominant auto-collected Hannes data: agentview as main image,
        # sideview as wrist image, both resized to 256x256.
        base_img = cv2.resize(
            obs["agentview_image"], (256, 256), interpolation=cv2.INTER_AREA
        )
        wrist_img = cv2.resize(
            obs["sideview_image"], (256, 256), interpolation=cv2.INTER_AREA
        )

        libero_obs = {
            "observation/state": np.zeros((8,), dtype=np.float32),
            "observation/image": base_img,
            "observation/wrist_image": wrist_img,
            "prompt": prompt,
        }

        out = policy.infer(libero_obs)
        actions_seq = np.asarray(out["actions"], dtype=np.float32)
        action = actions_seq[0, :6].copy()

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
                rotate_anchor_pos = base_pos.copy()

        elif phase == "rotate":
            if rotate_anchor_pos is None:
                rotate_anchor_pos = base_pos.copy()
            target = rotate_anchor_pos.copy()
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

            palm_aligned = abs(float(obj_now[0]) - float(hand_center_x)) <= palm_center_tol
            near_forward_target = abs(base_pos[0] - forward_target_x) <= 0.0015
            if robot_can_contact or palm_aligned:
                target = np.array([base_pos[0], base_pos[1], desired_align_z], dtype=np.float64)
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
                phase = "hold"

        else:
            target = grasp_target
            hold_progress += 1

        if chosen_obj in ("can", "milk"):
            obj_now_stable = get_object_pos_from_joint(env, obj_joint)
            set_object_pos(env, obj_joint, obj_now_stable, quat=obj_upright_quat)

        eef_now = get_eef_pos(obs, env)
        z_err = float(eef_now[2] - desired_eef_z)
        if z_err > 0.0:
            target = target.copy()
            target[2] -= min(0.03, 1.2 * z_err)

        if phase in ("align", "rotate", "forward", "grasp", "grasp_lock"):
            target = target.copy()
            target[2] = min(target[2], base_pos[2], desired_align_z)

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
        env.sim.forward()

        states.append(get_flat_state(env))
        actions.append(action.copy())
        base_pos_seq.append(base_pos.copy())
        base_delta_seq.append(base_delta.astype(np.float32))

        obs, reward, done, _ = env.step(action)
        rewards.append(float(reward))
        dones.append(bool(done))

        if phase == "hold" and hold_progress >= 25:
            break

    obj_z_end = float(get_object_pos_from_joint(env, obj_joint)[2])
    lifted = int((obj_z_end - obj_z_start) > 0.03)

    return {
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
        "agent_frames": np.asarray(agent_frames, dtype=np.uint8),
    }


def _prompt_for_object(name: str) -> str:
    if name == "can":
        return "Grip the can"
    if name == "milk":
        return "Grip the milk"
    if name == "bread":
        return "Grip the bread"
    if name == "lemon":
        return "Grip the lemon"
    if name == "hammer":
        return "Grip the hammer"
    return f"Hold the {name}"


def main():
    parser = argparse.ArgumentParser(
        description="Hannes policy-driven test script with collected_data-style trajectories"
    )
    parser.add_argument("--task", type=str, default="Lift")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=260)
    parser.add_argument(
        "--target-object",
        type=str,
        default="can",
        help="Fixed target object (default: can): one of milk, can, bread, lemon, hammer",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    os.makedirs("hannes_demonstrations_test", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = os.path.join("hannes_demonstrations_test", "videos")
    os.makedirs(video_dir, exist_ok=True)

    env = suite.make(
        args.task,
        robots="Hannes",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        # Match dominant auto-collected data: agentview + sideview.
        camera_names=["agentview", "sideview"],
        camera_heights=512,
        camera_widths=512,
        control_freq=args.control_freq,
        horizon=args.horizon,
        ignore_done=True,
        reward_shaping=True,
        use_object_obs=False,
    )

    rng = np.random.default_rng(args.seed)

    joint_candidates = {
        "milk": "milk_joint0",
        "can": "can_joint0",
        "bread": "bread_joint0",
        "lemon": "lemon_joint0",
        "hammer": "hammer_joint0",
    }
    valid_joints = {}
    for name, joint in joint_candidates.items():
        try:
            _ = env.sim.model.get_joint_qpos_addr(joint)
            valid_joints[name] = joint
        except Exception:
            continue

    if len(valid_joints) == 0:
        raise RuntimeError("No target object joints found (expected milk_joint0 or can_joint0).")

    print("=== Hannes Policy Test (auto trajectories) ===")
    print("Video dir:", video_dir)
    print("Episodes:", args.episodes)
    print("Objects:", list(valid_joints.keys()))

    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    print("Connected to policy server, server metadata:", policy.get_server_metadata())

    object_names = list(valid_joints.keys())

    for ep in range(args.episodes):
        if args.target_object:
            if args.target_object not in valid_joints:
                raise ValueError(
                    f"target object {args.target_object} not available, choose from {list(valid_joints.keys())}"
                )
            chosen = args.target_object
        else:
            chosen = object_names[ep % len(object_names)]

        prompt = _prompt_for_object(chosen)
        ep_result = run_episode(
            env,
            rng,
            chosen,
            valid_joints,
            policy=policy,
            prompt=prompt,
            max_steps=args.horizon,
            base_speed=0.006,
        )

        agent_frames = ep_result["agent_frames"]
        video_path = os.path.join(
            video_dir,
            f"hannes_{args.task}_policy_test_{timestamp}_ep{ep:02d}_agentview.mp4",
        )
        print(f"Saving episode {ep} agentview video to {video_path} ...")
        imageio.mimsave(video_path, agent_frames, fps=args.control_freq)
        print(f"Saved episode {ep} video: {video_path}")

    env.close()

    print("=== Done ===")


if __name__ == "__main__":
    main()
