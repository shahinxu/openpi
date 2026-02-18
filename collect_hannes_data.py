import numpy as np
import h5py
import robosuite as suite
from datetime import datetime
import os
import time

NUM_EPISODES = 1
TASK = "Lift"
SAVE_DIR = "hannes_demonstrations"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

print(f"=== Starting Hannes Robot Data Collection ===")
print(f"Task: {TASK}")
print(f"Target Episodes: {NUM_EPISODES}")
print(f"Save Directory: {SAVE_DIR}")
print(f"\nControl Method: Keyboard (MuJoCo-friendly keys)")
print("\n=== Keyboard Controls (avoiding MuJoCo shortcuts) ===")
print("  Arrow ↑/↓: Move forward/backward (X-axis)")
print("  Arrow ←/→: Move left/right (Y-axis)")
print("  R / F:     Move up/down (Z-axis)")
print("  Z / X:     Wrist Pitch increase/decrease")
print("  , / .:     Wrist Yaw increase/decrease")
print("  Enter:     Close all fingers")
print("  Space:     Open all fingers")
print("  ESC:       End and save trajectory")

from pynput import keyboard

class ControlState:
    def __init__(self):
        self.esc_pressed = False
        # 关节控制
        self.i_pressed = False
        self.k_pressed = False
        self.j_pressed = False
        self.l_pressed = False
        self.space_pressed = False
        self.v_pressed = False
        # 位置控制
        self.w_pressed = False
        self.s_pressed = False
        self.a_pressed = False
        self.d_pressed = False
        self.q_pressed = False
        self.e_pressed = False
        
state = ControlState()

def on_press(key):
    try:
        # Position control - up/down
        if key.char == 'r' or key.char == 'R':
            state.w_pressed = True  # Up (Z+)
        elif key.char == 'f' or key.char == 'F':
            state.s_pressed = True  # Down (Z-)
        # Joint control - wrist rotation
        elif key.char == 'z' or key.char == 'Z':
            state.i_pressed = True  # Pitch+
        elif key.char == 'x' or key.char == 'X':
            state.k_pressed = True  # Pitch-
        elif key.char == ',':
            state.j_pressed = True  # Yaw-
        elif key.char == '.':
            state.l_pressed = True  # Yaw+
    except AttributeError:
        # Special keys
        if key == keyboard.Key.enter:
            state.space_pressed = True  # Close fingers
        elif key == keyboard.Key.space:
            state.v_pressed = True  # Open fingers
        elif key == keyboard.Key.up:
            state.q_pressed = True  # Forward (Y+)
        elif key == keyboard.Key.down:
            state.e_pressed = True  # Backward (Y-)
        elif key == keyboard.Key.left:
            state.a_pressed = True  # Left (X-)
        elif key == keyboard.Key.right:
            state.d_pressed = True  # Right (X+)
        elif key == keyboard.Key.esc:
            state.esc_pressed = True
            print("ESC pressed - Ending episode")

def on_release(key):
    try:
        # Position control - up/down
        if key.char == 'r' or key.char == 'R':
            state.w_pressed = False
        elif key.char == 'f' or key.char == 'F':
            state.s_pressed = False
        # Joint control - wrist rotation
        elif key.char == 'z' or key.char == 'Z':
            state.i_pressed = False
        elif key.char == 'x' or key.char == 'X':
            state.k_pressed = False
        elif key.char == ',':
            state.j_pressed = False
        elif key.char == '.':
            state.l_pressed = False
    except AttributeError:
        # Special keys
        if key == keyboard.Key.enter:
            state.space_pressed = False
        elif key == keyboard.Key.space:
            state.v_pressed = False
        elif key == keyboard.Key.up:
            state.q_pressed = False
        elif key == keyboard.Key.down:
            state.e_pressed = False
        elif key == keyboard.Key.left:
            state.a_pressed = False
        elif key == keyboard.Key.right:
            state.d_pressed = False

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
print("Keyboard listener started (pynput)")

env = suite.make(
    TASK,
    robots="Hannes",
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=["frontview", "agentview"],
    camera_heights=512,
    camera_widths=512,
    control_freq=20,
    horizon=1000,
    ignore_done=True,
    reward_shaping=True,
    use_object_obs=False,  # Simpler observation
)

# === 粘手逻辑相关：预先拿到手掌 site、物体的自由关节索引以及与手相关的 geom ===
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

print("\nEnvironment created, ready to collect data...")
print("Tip: Use mouse to drag camera view in render window")


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_path = os.path.join(SAVE_DIR, f"hannes_{TASK}_{timestamp}.hdf5")

collected_episodes = 0

with h5py.File(save_path, 'w') as f:
    for ep in range(NUM_EPISODES):
        print(f"\n=== Starting Trajectory Collection {ep+1}/{NUM_EPISODES} ===")
        print("Press ESC to end current trajectory and save")
        
        obs = env.reset()
        free_joint_qpos_addr = env.sim.model.get_joint_qpos_addr('robot0_base_free')
        current_qpos = env.sim.data.qpos[free_joint_qpos_addr[0]:free_joint_qpos_addr[0]+7].copy()
        current_qpos[2] = 1.2
        env.sim.data.qpos[free_joint_qpos_addr[0]:free_joint_qpos_addr[0]+7] = current_qpos
        env.sim.forward()
        
        env.render()
        
        # Initialize target position (will only change when WASD/QE pressed)
        target_position = np.array([0.125, 0.0, 1.0])  # Lowered from 1.2 to 1.0
        target_quaternion = np.array([0.707, 0, 0, -0.707])

        episode_actions = []
        episode_states = []
        episode_rewards = []
        episode_dones = []
        episode_frontview = []
        episode_agentview = []
        
        step_count = 0
        total_reward = 0
        
        state.esc_pressed = False
        state.joint_action = np.zeros(6)

        for obj in _STICKY_OBJECTS:
            obj["active"] = False
            obj["offset"] = np.zeros(3)
        
        print(f"\nRobot initial position: {env.sim.data.qpos[:3]}")
        print("Start control! Press ESC to end and save trajectory...")
        
        while not state.esc_pressed:
            start_time = time.time()
            
            action = np.zeros(6)
            
            if state.i_pressed:
                action[0] = 1.0  # Pitch up
            if state.k_pressed:
                action[0] = -1.0  # Pitch down
            if state.j_pressed:
                action[1] = -1.0  # Yaw left
            if state.l_pressed:
                action[1] = 1.0  # Yaw right
            if state.space_pressed:
                action[2:] = 1.0  # Close fingers
            if state.v_pressed:
                action[2:] = -1.0  # Open fingers
            
            if np.any(action != 0):
                print(f"  Action: {action}, Keys: Z={state.i_pressed}, X={state.k_pressed}, ,={state.j_pressed}, .={state.l_pressed}, Enter={state.space_pressed}, Space={state.v_pressed}")
            
            episode_states.append({
                'robot_qpos': env.sim.data.qpos[:].copy(),
                'robot_qvel': env.sim.data.qvel[:].copy(),
            })
            episode_actions.append(action)
            episode_frontview.append(obs['frontview_image'])
            episode_agentview.append(obs['agentview_image'])

            obs, reward, done, info = env.step(action)
            
            # Update target position based on keyboard input
            move_speed = 0.01
            pos_changed = False
            
            if state.w_pressed:
                target_position[2] += move_speed  # R key: Up (Z+)
                pos_changed = True
            if state.s_pressed:
                target_position[2] -= move_speed  # F key: Down (Z-)
                pos_changed = True
            if state.a_pressed:
                target_position[1] -= move_speed  # Left arrow: Left (Y-)
                pos_changed = True
            if state.d_pressed:
                target_position[1] += move_speed  # Right arrow: Right (Y+)
                pos_changed = True
            if state.q_pressed:
                target_position[0] -= move_speed  # Up arrow: Forward (X+)
                pos_changed = True
            if state.e_pressed:
                target_position[0] += move_speed  # Down arrow: Backward (X-)
                pos_changed = True
            
            # Force free joint to target position every frame (prevents gravity drift)
            free_joint_qpos_addr = env.sim.model.get_joint_qpos_addr('robot0_base_free')
            free_joint_qvel_addr = env.sim.model.get_joint_qvel_addr('robot0_base_free')
            
            env.sim.data.qpos[free_joint_qpos_addr[0]:free_joint_qpos_addr[0]+3] = target_position
            env.sim.data.qpos[free_joint_qpos_addr[0]+3:free_joint_qpos_addr[0]+7] = target_quaternion
            env.sim.data.qvel[free_joint_qvel_addr[0]:free_joint_qvel_addr[0]+6] = 0.0

            # === 粘手逻辑：一旦与手部发生接触即吸附到手掌 ===
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
                            if (
                                _STICKY_HAND_GEOM_IDS is None
                                or other in _STICKY_HAND_GEOM_IDS
                            ):
                                contact_hits.add(obj_entry["name"])
                        obj_entry = _STICKY_GEOM_TO_OBJECT.get(g2)
                        if obj_entry is not None:
                            other = g1
                            if (
                                _STICKY_HAND_GEOM_IDS is None
                                or other in _STICKY_HAND_GEOM_IDS
                            ):
                                contact_hits.add(obj_entry["name"])

                for obj in _STICKY_OBJECTS:
                    if release_on_open and obj["active"]:
                        obj["active"] = False
                        print(f"[Sticky] Released {obj['name']} from hand")
                        continue

                    if obj["active"]:
                        env.sim.data.qpos[
                            obj["qpos_addr"][0]:obj["qpos_addr"][0]+3
                        ] = palm_pos + obj["offset"]
                        if obj["qvel_addr"] is not None:
                            env.sim.data.qvel[
                                obj["qvel_addr"][0]:obj["qvel_addr"][0]+6
                            ] = 0.0
                        continue

                    obj_pos = env.sim.data.qpos[
                        obj["qpos_addr"][0]:obj["qpos_addr"][0]+3
                    ].copy()

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
                        reason = (
                            "contact"
                            if contact_triggered
                            else "manual" if manual_trigger
                            else "proximity"
                        )
                        print(f"[Sticky] Attached {obj['name']} to hand ({reason})")

            env.sim.forward()
            
            if pos_changed and step_count % 20 == 0:
                pos_keys = f"R={state.w_pressed} F={state.s_pressed} ←={state.a_pressed} →={state.d_pressed} ↑={state.q_pressed} ↓={state.e_pressed}"
                print(f"  Position control: {pos_keys}, Target Position: {target_position}")
            
            env.render()
            
            episode_rewards.append(reward)
            episode_dones.append(done)
            total_reward += reward
            step_count += 1
            
            if step_count % 100 == 0:
                print(f"  Steps: {step_count}, Total Reward: {total_reward:.3f}, Hand Position: {env.sim.data.qpos[:3]}")
            
            elapsed = time.time() - start_time
            diff = 1.0 / 20.0 - elapsed
            if diff > 0:
                time.sleep(diff)
        
        print(f"\nTrajectory complete: {step_count} steps, Total reward: {total_reward:.3f}")
        
        ep_group = f.create_group(f'episode_{ep}')
        ep_group.create_dataset('actions', data=np.array(episode_actions))
        ep_group.create_dataset('rewards', data=np.array(episode_rewards))
        ep_group.create_dataset('dones', data=np.array(episode_dones))
        ep_group.create_dataset('frontview_images', data=np.array(episode_frontview), compression='gzip')
        ep_group.create_dataset('agentview_images', data=np.array(episode_agentview), compression='gzip')
        
        # 保存元信息
        ep_group.attrs['num_steps'] = step_count
        ep_group.attrs['total_reward'] = total_reward
        ep_group.attrs['task'] = TASK
        
        import imageio
        video_dir = os.path.join(SAVE_DIR, 'videos')
        os.makedirs(video_dir, exist_ok=True)
        
        frontview_path = os.path.join(video_dir, f'{timestamp}_ep{ep}_frontview.mp4')
        agentview_path = os.path.join(video_dir, f'{timestamp}_ep{ep}_agentview.mp4')
        
        print(f"Saving videos...")
        imageio.mimsave(frontview_path, [img[::-1] for img in episode_frontview], fps=20)
        imageio.mimsave(agentview_path, [img[::-1] for img in episode_agentview], fps=20)
        print(f"  Videos saved: {frontview_path}")
        print(f"                {agentview_path}")
        
        collected_episodes += 1

env.close()
listener.stop()

print(f"\n=== Data Collection Complete ===")
print(f"Save Location: {save_path}")
print(f"Total Trajectories: {collected_episodes}")
print(f"\nData Contains:")
print(f"  - actions: Action sequence (6 joints)")
print(f"  - rewards: Rewards")
print(f"  - frontview_images: Front view (512x512)")
print(f"  - agentview_images: Agent view (512x512)")
print(f"  - videos: Video files saved in {os.path.join(SAVE_DIR, 'videos')}")
