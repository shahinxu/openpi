import os
import time
from datetime import datetime

import h5py
import numpy as np
import robosuite as suite
from openpi_client import websocket_client_policy as _websocket_client_policy


NUM_EPISODES = 1
ENV_TASK = "Lift"
PROMPT = "Hold the milk carton"
SAVE_DIR = "hannes_demonstrations_test"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

print(f"=== Starting Hannes Robot Data Collection (Policy-Controlled) ===")
print(f"Env Task: {ENV_TASK}")
print(f"Prompt: {PROMPT}")
print(f"Target Episodes: {NUM_EPISODES}")
print(f"Save Directory: {SAVE_DIR}")

policy = _websocket_client_policy.WebsocketClientPolicy(
    host="127.0.0.1",
    port=8000,
)
print("Connected to policy server, server metadata:", policy.get_server_metadata())


MAX_STEPS = 500

env = suite.make(
    ENV_TASK,
    robots="Hannes",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=["agentview", "sideview"],
    camera_heights=512,
    camera_widths=512,
    control_freq=20,
    horizon=MAX_STEPS,
    ignore_done=True,
    reward_shaping=True,
    use_object_obs=False,
)

try:
    _STICKY_PALM_SITE_ID = env.sim.model.site_name2id("robot0_right_center")
except Exception:
    _STICKY_PALM_SITE_ID = None

try:
    _STICKY_FINGER_SITE_IDS = [
        env.sim.model.site_name2id("robot0_forefinger_pivot"),
        env.sim.model.site_name2id("robot0_midfinger_pivot"),
        env.sim.model.site_name2id("robot0_ringfinger_pivot"),
        env.sim.model.site_name2id("robot0_littlefinger_pivot"),
    ]
except Exception:
    _STICKY_FINGER_SITE_IDS = None

_STICKY_HAND_GEOM_IDS = set()
_FINGER_BODY_NAMES = [
    "forefinger_link",
    "midfinger_link",
    "ringfinger_link",
    "littlefinger_link",
]
for base_body in _FINGER_BODY_NAMES:
    body_id = None
    for candidate in (f"robot0_{base_body}", base_body):
        try:
            body_id = env.sim.model.body_name2id(candidate)
            break
        except Exception:
            continue
    if body_id is None:
        continue
    geom_adr = env.sim.model.body_geomadr[body_id]
    geom_num = env.sim.model.body_geomnum[body_id]
    if geom_num <= 0:
        continue
    for gi in range(geom_adr, geom_adr + geom_num):
        _STICKY_HAND_GEOM_IDS.add(gi)
if not _STICKY_HAND_GEOM_IDS:
    _STICKY_HAND_GEOM_IDS = None
_STICKY_OBJECT_SPECS = [
    ("milk", ["milk_g0"]),
    ("lemon", ["lemon_g0"]),
    ("bread", ["bread_g0"]),
    ("can", ["can_g0"]),
    ("hammer", ["hammer_handle", "hammer_head"]),
]

_STICKY_OBJECTS = []
_STICKY_GEOM_TO_OBJECT = {}
for base_name, geom_names in _STICKY_OBJECT_SPECS:
    joint_name = f"{base_name}_joint0"
    try:
        qpos_addr = env.sim.model.get_joint_qpos_addr(joint_name)
        qvel_addr = env.sim.model.get_joint_qvel_addr(joint_name)
    except Exception:
        continue

    geom_ids = []
    for geom_name in geom_names:
        try:
            geom_ids.append(env.sim.model.geom_name2id(geom_name))
        except Exception:
            pass
    if not geom_ids:
        continue

    entry = {
        "name": base_name,
        "qpos_addr": qpos_addr,
        "qvel_addr": qvel_addr,
        "geom_ids": geom_ids,
        "active": False,
        "offset": np.zeros(3),
    }
    _STICKY_OBJECTS.append(entry)
    for gid in geom_ids:
        _STICKY_GEOM_TO_OBJECT[gid] = entry

_MILK_OBJECT = None
for obj in _STICKY_OBJECTS:
    if obj["name"] == "milk":
        _MILK_OBJECT = obj
        break

print("\nEnvironment created, ready to collect data (policy-controlled)...")


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(SAVE_DIR, f"hannes_{ENV_TASK}_{timestamp}_policy.hdf5")

collected_episodes = 0

with h5py.File(save_path, "w") as f:
    for ep in range(NUM_EPISODES):
        print(f"\n=== Starting Trajectory Collection {ep+1}/{NUM_EPISODES} (policy) ===")

        obs = env.reset()
        free_joint_qpos_addr = env.sim.model.get_joint_qpos_addr("robot0_base_free")
        target_position = env.sim.data.qpos[free_joint_qpos_addr[0] : free_joint_qpos_addr[0] + 3].copy()
        target_position[2] += 0.2
        target_quaternion = np.array([0.707, 0, 0, -0.707])
        env.sim.data.qpos[free_joint_qpos_addr[0] : free_joint_qpos_addr[0] + 3] = target_position
        env.sim.data.qpos[free_joint_qpos_addr[0] + 3 : free_joint_qpos_addr[0] + 7] = target_quaternion
        env.sim.forward()

        episode_actions: list[np.ndarray] = []
        episode_states: list[dict[str, np.ndarray]] = []
        episode_rewards: list[float] = []
        episode_dones: list[bool] = []
        episode_agentview: list[np.ndarray] = []
        episode_sideview: list[np.ndarray] = []

        step_count = 0
        total_reward = 0.0

        base_position = target_position.copy()

        base_target_position = base_position.copy()
        if _MILK_OBJECT is not None:
            milk_qpos_start = _MILK_OBJECT["qpos_addr"][0]
            milk_pos = env.sim.data.qpos[milk_qpos_start : milk_qpos_start + 3].copy()
            # 期望最终位置：在 milk 前方一点
            base_target_position[0] = milk_pos[0] + 0.2
            base_target_position[1] = milk_pos[1]
            base_target_position[2] = milk_pos[2]

        for obj in _STICKY_OBJECTS:
            obj["active"] = False
            obj["offset"] = np.zeros(3)

        done = False
        while not done and step_count < MAX_STEPS:
            start_time = time.time()

            libero_obs = {
                "observation/state": np.zeros((8,), dtype=np.float32),
                "observation/image": obs["agentview_image"],
                "observation/wrist_image": obs["sideview_image"],
                "prompt": PROMPT,
            }

            print(f"Step {step_count}: calling policy.infer ...", flush=True)
            out = policy.infer(libero_obs)
            print(f"Step {step_count}: got actions from policy", flush=True)
            actions_seq = np.asarray(out["actions"], dtype=np.float32)
            action = actions_seq[0, :6]

            episode_states.append(
                {
                    "robot_qpos": env.sim.data.qpos[:].copy(),
                    "robot_qvel": env.sim.data.qvel[:].copy(),
                }
            )
            episode_actions.append(action)
            episode_agentview.append(obs["agentview_image"])
            episode_sideview.append(obs["sideview_image"])

            obs, reward, done, info = env.step(action)
            # 让基座从初始位置逐步朝 base_target_position 逼近（固定目标点），移动得更快一些
            move_fraction = 0.05
            target_position = (1 - move_fraction) * target_position + move_fraction * base_target_position

            free_joint_qpos_addr = env.sim.model.get_joint_qpos_addr("robot0_base_free")
            free_joint_qvel_addr = env.sim.model.get_joint_qvel_addr("robot0_base_free")

            env.sim.data.qpos[free_joint_qpos_addr[0] : free_joint_qpos_addr[0] + 3] = target_position
            env.sim.data.qpos[free_joint_qpos_addr[0] + 3 : free_joint_qpos_addr[0] + 7] = target_quaternion
            env.sim.data.qvel[free_joint_qvel_addr[0] : free_joint_qvel_addr[0] + 6] = 0.0

            if _STICKY_OBJECTS and _STICKY_PALM_SITE_ID is not None:
                finger_mean = np.mean(action[2:]) if action.shape[0] >= 3 else 0.0
                release_on_open = finger_mean < -0.5
                palm_pos = env.sim.data.site_xpos[_STICKY_PALM_SITE_ID].copy()

                contact_hits = set()
                if _STICKY_GEOM_TO_OBJECT:
                    ncon = env.sim.data.ncon
                    for ci in range(ncon):
                        contact = env.sim.data.contact[ci]
                        g1, g2 = contact.geom1, contact.geom2

                        obj_entry = _STICKY_GEOM_TO_OBJECT.get(g1)
                        if obj_entry is not None:
                            other = g2
                            if _STICKY_HAND_GEOM_IDS is None or other in _STICKY_HAND_GEOM_IDS:
                                contact_hits.add(obj_entry["name"])
                        obj_entry = _STICKY_GEOM_TO_OBJECT.get(g2)
                        if obj_entry is not None:
                            other = g1
                            if _STICKY_HAND_GEOM_IDS is None or other in _STICKY_HAND_GEOM_IDS:
                                contact_hits.add(obj_entry["name"])

                for obj in _STICKY_OBJECTS:
                    if release_on_open and obj["active"]:
                        obj["active"] = False
                        print(f"[Sticky] Released {obj['name']} from hand")
                        continue

                    if obj["active"]:
                        env.sim.data.qpos[obj["qpos_addr"][0] : obj["qpos_addr"][0] + 3] = palm_pos + obj["offset"]
                        if obj["qvel_addr"] is not None:
                            env.sim.data.qvel[obj["qvel_addr"][0] : obj["qvel_addr"][0] + 6] = 0.0
                        continue

                    obj_pos = env.sim.data.qpos[obj["qpos_addr"][0] : obj["qpos_addr"][0] + 3].copy()

                    contact_triggered = obj["name"] in contact_hits

                    closest_finger = np.inf
                    if _STICKY_FINGER_SITE_IDS is not None:
                        for sid in _STICKY_FINGER_SITE_IDS:
                            finger_pos = env.sim.data.site_xpos[sid]
                            dist = np.linalg.norm(obj_pos - finger_pos)
                            if dist < closest_finger:
                                closest_finger = dist

                    near_hand = closest_finger < 0.02

                    manual_trigger = finger_mean > 0.5 and closest_finger < 0.025

                    if contact_triggered or near_hand or manual_trigger:
                        obj["offset"] = obj_pos - palm_pos
                        obj["active"] = True
                        reason = "contact" if contact_triggered else "manual" if manual_trigger else "proximity"
                        print(f"[Sticky] Attached {obj['name']} to hand ({reason})")

            env.sim.forward()

            episode_rewards.append(float(reward))
            episode_dones.append(bool(done))
            total_reward += float(reward)
            step_count += 1

            elapsed = time.time() - start_time
            diff = 1.0 / 20.0 - elapsed
            if diff > 0:
                time.sleep(diff)

        print(f"\nTrajectory complete: {step_count} steps, Total reward: {total_reward:.3f}")

        ep_group = f.create_group(f"episode_{ep}")
        ep_group.create_dataset("actions", data=np.array(episode_actions))
        ep_group.create_dataset("rewards", data=np.array(episode_rewards))
        ep_group.create_dataset("dones", data=np.array(episode_dones))
        ep_group.create_dataset("agentview_images", data=np.array(episode_agentview), compression="gzip")
        ep_group.create_dataset("sideview_images", data=np.array(episode_sideview), compression="gzip")

        ep_group.attrs["num_steps"] = step_count
        ep_group.attrs["total_reward"] = total_reward
        ep_group.attrs["task"] = PROMPT

        import imageio

        video_dir = os.path.join(SAVE_DIR, "videos")
        os.makedirs(video_dir, exist_ok=True)

        agentview_path = os.path.join(video_dir, f"{timestamp}_ep{ep}_agentview_policy.mp4")
        sideview_path = os.path.join(video_dir, f"{timestamp}_ep{ep}_sideview_policy.mp4")

        print("Saving videos...")
        imageio.mimsave(agentview_path, [img[::-1] for img in episode_agentview], fps=20)
        imageio.mimsave(sideview_path, [img[::-1] for img in episode_sideview], fps=20)
        print(f"  Videos saved: {agentview_path}")
        print(f"                {sideview_path}")

        collected_episodes += 1

env.close()

print(f"\n=== Data Collection Complete (Policy-Controlled) ===")
print(f"Save Location: {save_path}")
print(f"Total Trajectories: {collected_episodes}")
print("Data format is the same as collect_hannes_data.py, just with '_policy' suffix in filenames.")