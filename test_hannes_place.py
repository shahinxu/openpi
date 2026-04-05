import argparse
import os
from datetime import datetime

import imageio
import matplotlib.pyplot as plt
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
    policy,
    prompt,
    max_steps=260,
    base_speed=0.006,
    sticky_after_close=False,
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
    agent_frames = []

    obj_z_start = float(obj_pos[2])
    desired_eef_z = float(get_eef_pos(obs, env)[2])
    yaw_joint_name = "robot0_wrist_yaw"
    pitch_joint_name = "robot0_wrist_pitch"
    pitch_initial = float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(pitch_joint_name)])
    finger_joint_names = []
    for base_name in ("forefinger_joint", "midfinger_joint", "ringfinger_joint", "littlefinger_joint"):
        joint_name = resolve_joint_name(env, [f"robot0_{base_name}", base_name])
        if joint_name is not None:
            finger_joint_names.append(joint_name)

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
    sticky_attached = False
    sticky_offset = np.zeros(3, dtype=np.float64)

    for t in range(max_steps):
        # Hard-lock wrist pitch to its initial value for this episode.
        set_joint_scalar(env, pitch_joint_name, pitch_initial)
        env.sim.forward()

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
        compact_state = get_compact_state(env, pitch_joint_name, yaw_joint_name, finger_joint_names)
        policy_state = np.zeros((8,), dtype=np.float32)
        policy_state[: compact_state.shape[0]] = compact_state

        libero_obs = {
            "observation/state": policy_state,
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

        if sticky_after_close:
            palm_pos = get_palm_pos(env)
            if palm_pos is not None and isinstance(obj_qpos_addr, tuple):
                obj_now_for_sticky = env.sim.data.qpos[obj_qpos_addr[0] : obj_qpos_addr[0] + 3].copy()
                finger_mean = float(np.mean(action[2:]))
                if sticky_attached and finger_mean < -0.4:
                    sticky_attached = False
                if (not sticky_attached) and phase in ("grasp_lock", "hold"):
                    near_palm = np.linalg.norm(obj_now_for_sticky - palm_pos) < 0.05
                    if robot_can_contact or near_palm:
                        sticky_offset = obj_now_for_sticky - palm_pos
                        sticky_attached = True
                if sticky_attached:
                    set_object_pos(env, obj_joint, palm_pos + sticky_offset, quat=obj_upright_quat)

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
        set_joint_scalar(env, pitch_joint_name, pitch_initial)
        env.sim.forward()

        states.append(compact_state.copy())
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


def save_predicted_action_plot(actions: np.ndarray, out_path: str, title: str) -> None:
    if actions.size == 0:
        return

    t_axis = np.arange(actions.shape[0])
    plot_dims = min(3, actions.shape[1])
    dim_names = ["wrist_pitch", "wrist_yaw", "grip_mean"][:plot_dims]

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "grid.alpha": 0.25,
        }
    )

    fig, axes = plt.subplots(plot_dims, 1, figsize=(10, 6.8), sharex=True)
    if plot_dims == 1:
        axes = [axes]
    fig.suptitle(title, y=0.98, fontweight="bold")

    for i in range(plot_dims):
        ax = axes[i]
        ax.plot(t_axis, actions[:, i], color="#c84c09", linewidth=2.0, label="pred")
        ax.set_ylabel(dim_names[i])
        ax.set_ylim(-1.0, 1.0)
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


def save_state_plot(states: np.ndarray, out_path: str, title: str) -> None:
    if states.size == 0:
        return

    t_axis = np.arange(states.shape[0])
    plot_dims = min(3, states.shape[1])
    dim_names = ["wrist_pitch", "wrist_yaw", "grip_mean"][:plot_dims]

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "grid.alpha": 0.25,
        }
    )

    fig, axes = plt.subplots(plot_dims, 1, figsize=(10, 6.8), sharex=True)
    if plot_dims == 1:
        axes = [axes]
    fig.suptitle(title, y=0.98, fontweight="bold")

    for i in range(plot_dims):
        ax = axes[i]
        ax.plot(t_axis, states[:, i], color="#1f4e79", linewidth=2.0, label="state")
        ax.set_ylabel(dim_names[i])
        ax.set_ylim(-1.0, 1.0)
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
    parser.add_argument(
        "--sticky-after-close",
        dest="sticky_after_close",
        action="store_true",
        default=False,
        help="Attach object to palm after close contact and release on opening.",
    )
    args = parser.parse_args()

    os.makedirs("hannes_demonstrations_test", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = os.path.join("hannes_demonstrations_test", "videos")
    plot_dir = os.path.join("hannes_demonstrations_test", "plots")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

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
            sticky_after_close=args.sticky_after_close,
        )

        agent_frames = ep_result["agent_frames"]
        video_path = os.path.join(
            video_dir,
            f"hannes_{args.task}_policy_test_{timestamp}_ep{ep:02d}_agentview.mp4",
        )
        print(f"Saving episode {ep} agentview video to {video_path} ...")
        imageio.mimsave(video_path, agent_frames, fps=args.control_freq)
        print(f"Saved episode {ep} video: {video_path}")

        action_plot_path = os.path.join(
            plot_dir,
            f"hannes_{args.task}_policy_test_{timestamp}_ep{ep:02d}_pred_actions.png",
        )
        save_predicted_action_plot(
            actions=np.asarray(ep_result["actions"], dtype=np.float32),
            out_path=action_plot_path,
            title=f"Predicted Actions (First 3 dims) | {chosen}",
        )
        print(f"Saved episode {ep} predicted-action plot: {action_plot_path}")

        state_plot_path = os.path.join(
            plot_dir,
            f"hannes_{args.task}_policy_test_{timestamp}_ep{ep:02d}_states.png",
        )
        save_state_plot(
            states=np.asarray(ep_result["states"], dtype=np.float32),
            out_path=state_plot_path,
            title=f"States (First 3 dims) | {chosen}",
        )
        print(f"Saved episode {ep} state plot: {state_plot_path}")

    env.close()

    print("=== Done ===")


if __name__ == "__main__":
    main()
