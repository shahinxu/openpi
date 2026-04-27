# Real Robot Control Pipeline

Production-ready VLA inference system for real robot control.

## Architecture

```
                    ┌─────────────────┐
                    │   Gradio UI     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ RobotControl    │
                    │ Pipeline        │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
   ┌────────────┐     ┌────────────┐     ┌────────────┐
   │  Camera    │     │  Model     │     │  Robot     │
   │ Interface  │     │ Interface  │     │ Interface  │
   └────┬───────┘     └────┬───────┘     └────┬───────┘
        │ (Gradio)         │ (JAX)            │ (Serial)
        │                  │                  │
        └──────────────────┴──────────────────┘
                            │
                    ┌───────▼───────┐
                    │  Normalizer   │
                    │  (DataRange)  │
                    └───────────────┘
```

## File Structure

```
real_robot_pipeline/
├── config.yaml              # Configuration (YAML)
├── main.py                  # Main entry point + Gradio UI
├── interfaces.py            # Abstract base classes
├── cameras.py               # Camera implementations
├── models.py                # Model inference implementations
├── robots.py                # Robot controller implementations
├── normalizers.py           # Action normalization
└── README.md                # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install gradio pyyaml numpy pillow
# For JAX/LeRobot inference:
pip install lerobot
# For Serial robot control:
pip install pyserial
# For ROS:
pip install rospy geometry_msgs
```

### 2. Configure (config.yaml)

Edit `config.yaml` to set:
- Model checkpoint path
- Camera type (Gradio/OpenCV)
- Robot type (Dummy/Serial/ROS)
- Action normalization ranges
- UI settings

### 3. Run

```bash
cd /mnt/unites2/home/zhx/openpi/real_robot_pipeline
python main.py config.yaml
```

Then open browser to `http://localhost:7860`

## Extending the Pipeline

### Adding a New Camera Type

1. **Implement the interface** in `cameras.py`:

```python
class MyCamera(CameraInterface):
    def connect(self) -> bool:
        # Your connection logic
        return True
    
    def capture_frame(self) -> Optional[np.ndarray]:
        # Return (H, W, 3) uint8 RGB
        return frame
    
    def close(self):
        pass
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
```

2. **Register in factory** (`cameras.py`):

```python
def create_camera(config: dict) -> CameraInterface:
    camera_type = config.get("type", "gradio_webcam")
    
    if camera_type == "my_camera":
        return MyCamera(**config.get("my_camera_params", {}))
    # ... rest
```

3. **Configure in config.yaml**:

```yaml
camera:
  type: "my_camera"
  my_camera_params:
    param1: value1
```

### Adding a New Model Backend

Same pattern in `models.py`:

```python
class MyModelInference(InferenceEngineInterface):
    def load_model(self, checkpoint_path: str) -> bool:
        # Load from your format
        return True
    
    def infer(self, image, instruction, prev_state) -> np.ndarray:
        # Return [pitch, yaw, grip, ...]
        return action
    
    # ... rest
```

### Adding a New Robot Type

Same pattern in `robots.py`:

```python
class MyRobot(RobotControllerInterface):
    def connect(self) -> bool:
        # Connect to robot hardware
        return True
    
    def send_command(self, action_cmd: Dict[str, int]) -> bool:
        # Send {"pitch": int, "yaw": int, "grip": int}
        return True
    
    def get_state(self) -> Optional[np.ndarray]:
        # Return [pitch, yaw, grip] or None
        return state
    
    # ... rest
```

### Custom Action Normalization

Edit `normalizers.py`:

```python
class MyNormalizer(ActionNormalizerInterface):
    def normalize(self, raw_action: np.ndarray) -> Dict[str, int]:
        # Map raw model output to command range
        return {"pitch": int, "yaw": int, "grip": int}
```

## Configuration Reference

### Model

```yaml
model:
  type: "jax_checkpoint"  # "jax_checkpoint", "huggingface", "dummy"
  checkpoint_path: "/path/to/checkpoint"
  device: "cuda"          # "cuda" or "cpu"
  inference_frequency: 10 # Hz

**VLASH (PyTorch + LoRA)**:
```yaml
model:
  type: "vlash_checkpoint"
  checkpoint_path: "/mnt/unites2/home/zhx/openpi/outputs/train/pi05_hannes_all_async_lora/checkpoints/010000"
  device: "cuda"
```

**JAX with Egomotion Encoding**:
```yaml
model:
  type: "jax_checkpoint_with_egomotion"
  checkpoint_path: "/mnt/unites2/home/zhx/openpi/checkpoints/pi05_hannes_dataset_new/exp_egomotion_z_v1/20000"
  device: "cuda"
```

Egomotion is computed in real time from state delta in the control loop.
```

### Camera

```yaml
camera:
  type: "gradio_webcam"   # "gradio_webcam", "opencv"
  width: 640
  height: 480
  mirror: false
  facing_mode: "environment"  # "environment" for rear camera
```

### Robot

```yaml
robot:
  type: "dummy"           # "dummy", "serial", "ros"
  # For serial:
  # port: "/dev/ttyUSB0"
  # baudrate: 115200
```

### Action Normalization

```yaml
action_normalization:
  pitch:
    data_min: 0.48      # Data range from dataset
    data_max: 0.52
    cmd_min: 0          # Command range (0-100)
    cmd_max: 100
  yaw:
    data_min: -1.02     # Yaw range from dataset
    data_max: 0.69
    cmd_min: -100       # Yaw range (-100 to 100)
    cmd_max: 100
  grip:
    data_min: 0.0
    data_max: 1.0
    cmd_min: 0
    cmd_max: 100
```

## Data Range Reference

From `dataset_new` analysis:
- **pitch**: [0.48, 0.52] rad (~28°-30°, limited range)
- **yaw**: [-1.02, 0.69] rad (~-58° to +40°)
- **grip**: [0, 1] (normalized aperture)

**Initial state**: [0.5 pitch, -1.0 yaw, 0.0 grip] (start position)

## Debugging

### Enable Verbose Logging

In config.yaml:
```yaml
logging:
  level: "DEBUG"
```

Or in code:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Test with Dummy Mode

```yaml
model:
  type: "dummy"
camera:
  type: "gradio_webcam"
robot:
  type: "dummy"
```

This runs the full UI without real model/robot, useful for testing.

### Save Action History

```yaml
logging:
  history_file: "/tmp/robot_actions.jsonl"
```

Each line is JSON record with `frame`, `timestamp`, `raw_action`, `command`, `state`, `instruction`.

## Production Tips

1. **Use persistent logging**: Set `history_file` in config to record all actions
2. **Validate ranges**: Test normalization with sample data before deployment
3. **Monitor latency**: Check "Status" column for inference/command latency
4. **Graceful shutdown**: Use `Ctrl+C` to save history and close connections
5. **Separate configs**: Keep different `config.yaml` for test vs production

## Common Issues

### "Model not loaded"
- Check checkpoint path exists
- Verify JAX/LeRobot installed: `pip install lerobot`
- Check GPU availability: `nvidia-smi`

### "Robot not connected"
- For Serial: Check port with `ls /dev/ttyUSB*`
- For ROS: Verify ROS node is running
- For testing: Use `type: "dummy"` in config

### Slow inference
- Check GPU utilization: `nvidia-smi`
- Reduce image resolution in config
- Increase `batch_size` in model config (if supported)

### Gradio port already in use
- Change `server_port` in config (default 7860)
- Or kill existing process: `lsof -i :7860`

## License

[Your license here]

## Contact

For questions or issues, contact the development team.
