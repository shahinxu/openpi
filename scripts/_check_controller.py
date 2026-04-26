import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "robosuite"))
import robosuite as suite
env = suite.make(
    "Lift", robots="Hannes",
    has_renderer=False, has_offscreen_renderer=False,
    use_camera_obs=False, control_freq=20, horizon=10,
    ignore_done=True, reward_shaping=True, use_object_obs=False,
)
obs = env.reset()
print("action_dim:", env.action_dim)
robot = env.robots[0]
ctrl = robot.composite_controller
print("composite_controller type:", type(ctrl).__name__)
for part_name, part_ctrl in ctrl.part_controllers.items():
    print(
        f"  part={part_name!r}"
        f"  type={type(part_ctrl).__name__}"
        f"  output_type={getattr(part_ctrl, 'output_type', '?')}"
        f"  input_type={getattr(part_ctrl, 'input_type', '?')}"
        f"  control_dim={getattr(part_ctrl, 'control_dim', '?')}"
    )
# Check the raw default controller config
import json, pathlib
cfg_path = pathlib.Path("robosuite/controllers/config/robots/default_hannes.json")
if cfg_path.exists():
    cfg = json.loads(cfg_path.read_text())
    print("\ndefault_hannes.json:")
    print(json.dumps(cfg, indent=2))
env.close()
