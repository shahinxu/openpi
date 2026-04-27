"""
Real Robot Control Pipeline - Main Entry Point

High-level orchestration of camera, model, normalization, and robot control.
"""
import logging
import json
import time
import socket
from pathlib import Path
from typing import Optional, Dict, Tuple
from collections import deque
from datetime import datetime

import numpy as np
import yaml

# Import components
from interfaces import (
    CameraInterface,
    InferenceEngineInterface,
    RobotControllerInterface,
    ActionNormalizerInterface,
)
from cameras import create_camera
from models import create_inference_engine
from robots import create_robot_controller
from normalizers import create_normalizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class RobotControlPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize pipeline from config file."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize components
        logger.info("Initializing components...")
        self.camera: Optional[CameraInterface] = create_camera(self.config["camera"])
        self.model: Optional[InferenceEngineInterface] = create_inference_engine(self.config["model"])
        self.robot: Optional[RobotControllerInterface] = create_robot_controller(self.config["robot"])
        self.normalizer: Optional[ActionNormalizerInterface] = create_normalizer(self.config)
        
        # State management
        self.current_state = np.array(
            [
                self.config["initial_state"]["pitch"],
                self.config["initial_state"]["yaw"],
                self.config["initial_state"]["grip"],
            ],
            dtype=np.float32,
        )
        self.last_instruction = ""
        self.action_history = deque(maxlen=100)
        self.frame_count = 0
        self.start_time = None
        
        # Setup frame save directory
        save_frames = self.config.get("ui", {}).get("save_frames", False)
        if save_frames:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_dir = Path("runs") / run_id / "frames"
            self.save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving frames to {self.save_dir}")
        else:
            self.save_dir = None

        # Setup low-latency action streaming (TCP/UDP)
        self.action_stream_cfg = self.config.get("action_stream", {})
        self.action_stream_enabled = False
        self.action_stream_protocol = "udp"
        self.action_stream_addr = None
        self.action_stream_sock = None
        self._action_stream_last_connect_try = 0.0
        self._init_action_stream()

        # Connect to hardware
        self._connect_hardware()
        
        logger.info("Pipeline initialized")

        # Egomotion-enabled model works in real time from state delta.
        if self.config["model"].get("type") == "jax_checkpoint_with_egomotion":
            logger.info("Egomotion model detected, using real-time state-delta egomotion")
    
    def _load_config(self) -> dict:
        """Load and validate config file."""
        if not self.config_path.exists():
            logger.error(f"Config file not found: {self.config_path}")
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded config from {self.config_path}")
        return config
    
    def _connect_hardware(self):
        """Connect to all hardware components."""
        # Camera
        if not self.camera.connect():
            logger.warning("Failed to connect camera - UI will still work")
        
        # Model (already loaded in factory)
        if not self.model.is_loaded:
            logger.warning("Model not loaded - using dummy inference")
        
        # Robot
        if not self.robot.connect():
            logger.warning("Failed to connect robot - using dummy mode")

    def _init_action_stream(self):
        """Initialize action streaming to local computer."""
        if not self.action_stream_cfg.get("enabled", False):
            return

        protocol = str(self.action_stream_cfg.get("protocol", "udp")).lower()
        if protocol not in ("udp", "tcp"):
            logger.warning(f"Unsupported action_stream protocol '{protocol}', fallback to udp")
            protocol = "udp"

        host = self.action_stream_cfg.get("host", "127.0.0.1")
        port = int(self.action_stream_cfg.get("port", 18080))

        try:
            self.action_stream_protocol = protocol
            self.action_stream_addr = (host, port)

            if protocol == "udp":
                self.action_stream_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.action_stream_sock.setblocking(False)
                self.action_stream_enabled = True
                logger.info(f"Action UDP stream enabled: {host}:{port}")
            else:
                self.action_stream_sock = None
                self.action_stream_enabled = True
                logger.info(f"Action TCP stream configured: {host}:{port}")
        except Exception as e:
            self.action_stream_enabled = False
            self.action_stream_sock = None
            self.action_stream_addr = None
            logger.warning(f"Failed to initialize action stream: {e}")

    def _ensure_tcp_action_stream_connected(self):
        """Best-effort TCP connect with throttled retries."""
        if self.action_stream_protocol != "tcp":
            return False
        if self.action_stream_addr is None:
            return False
        if self.action_stream_sock is not None:
            return True

        now = time.time()
        if now - self._action_stream_last_connect_try < 1.0:
            return False
        self._action_stream_last_connect_try = now

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.2)
            sock.connect(self.action_stream_addr)
            sock.settimeout(0.0)
            self.action_stream_sock = sock
            logger.info(
                f"Action TCP stream connected: {self.action_stream_addr[0]}:{self.action_stream_addr[1]}"
            )
            return True
        except Exception:
            return False

    def _stream_action(self, raw_action: np.ndarray, action_cmd: Dict, infer_ms: float):
        """Send action packet via UDP/TCP (best-effort, low latency)."""
        if not self.action_stream_enabled or self.action_stream_addr is None:
            return

        packet = {
            "frame": self.frame_count,
            "ts_unix": time.time(),
            "infer_ms": round(infer_ms, 3),
            "raw_action": [float(x) for x in raw_action[:3]],
            "command": {
                "pitch": int(action_cmd.get("pitch", 0)),
                "yaw": int(action_cmd.get("yaw", 0)),
                "grip": int(action_cmd.get("grip", 0)),
            },
        }
        try:
            payload = json.dumps(packet, separators=(",", ":")).encode("utf-8")

            if self.action_stream_protocol == "udp":
                if self.action_stream_sock is None:
                    return
                self.action_stream_sock.sendto(payload, self.action_stream_addr)
            else:
                if not self._ensure_tcp_action_stream_connected():
                    return
                assert self.action_stream_sock is not None
                # Newline-delimited JSON over TCP.
                self.action_stream_sock.sendall(payload + b"\n")
        except Exception:
            # Keep inference path robust; do not fail pipeline for stream errors.
            if self.action_stream_protocol == "tcp" and self.action_stream_sock is not None:
                try:
                    self.action_stream_sock.close()
                except Exception:
                    pass
                self.action_stream_sock = None

    def _clip_command_value(self, key: str, value: float) -> int:
        """Clip manual command value to configured robot command range."""
        cfg = self.config.get("action_normalization", {}).get(key, {})
        cmd_min = int(cfg.get("cmd_min", -100))
        cmd_max = int(cfg.get("cmd_max", 100))
        return int(np.clip(int(round(value)), cmd_min, cmd_max))

    def process_manual_command(
        self,
        pitch: float,
        yaw: float,
        grip: float,
        instruction: str = "manual_override",
    ) -> Tuple[str, str, str, str]:
        """Send manual command (bypassing model inference)."""
        t_start = time.time()

        if self.start_time is None:
            self.start_time = t_start
        self.frame_count += 1
        self.last_instruction = instruction if instruction else "manual_override"

        action_cmd = {
            "pitch": self._clip_command_value("pitch", pitch),
            "yaw": self._clip_command_value("yaw", yaw),
            "grip": self._clip_command_value("grip", grip),
        }

        # Keep raw_action structure for downstream logging/streaming consumers.
        raw_action = self.current_state.copy()
        self._stream_action(raw_action, action_cmd, infer_ms=0.0)

        success = self.robot.send_command(action_cmd)

        robot_state = self.robot.get_state()
        if robot_state is not None:
            self.current_state = robot_state

        self.action_history.append({
            "frame": self.frame_count,
            "timestamp": datetime.now().isoformat(),
            "raw_action": raw_action.tolist(),
            "command": action_cmd,
            "state": self.current_state.tolist(),
            "instruction": self.last_instruction,
            "success": success,
            "mode": "manual",
        })

        latency_ms = (time.time() - t_start) * 1000
        raw_str = "[manual override: VLA bypassed]"
        cmd_str = (
            f"[pitch={action_cmd['pitch']:3d}, yaw={action_cmd['yaw']:4d}, grip={action_cmd['grip']:3d}]"
        )
        state_str = (
            f"[pitch={self.current_state[0]:.4f}, yaw={self.current_state[1]:.4f}, "
            f"grip={self.current_state[2]:.4f}]"
        )
        status = f"✓ manual {latency_ms:.1f}ms" if success else f"✗ manual {latency_ms:.1f}ms (send failed)"
        return raw_str, cmd_str, state_str, status
    
    def process_frame(self, image: np.ndarray, instruction: str) -> Tuple[Dict, Dict, str]:
        """
        Process single frame: infer action and control robot.
        
        Args:
            image: (H, W, 3) uint8 RGB frame
            instruction: Task instruction text
            
        Returns:
            (raw_action_dict, command_dict, status_message)
        """
        t_start = time.time()
        
        if self.start_time is None:
            self.start_time = t_start
        self.frame_count += 1
        self.last_instruction = instruction
        
        # Run inference
        logger.debug(f"Frame {self.frame_count}: inferring action...")
        t_infer_start = time.time()
        raw_action = self.model.infer(image, instruction, self.current_state)
        t_infer_end = time.time()
        infer_ms = (t_infer_end - t_infer_start) * 1000
        logger.info(f"[TIMING] frame={self.frame_count} infer={infer_ms:.0f}ms")
        
        # Save frame if enabled
        if self.save_dir is not None:
            from PIL import Image as PILImage
            img_path = self.save_dir / f"frame_{self.frame_count:06d}.jpg"
            PILImage.fromarray(image).save(img_path, quality=90)

        # Normalize to command format
        action_cmd = self.normalizer.normalize(raw_action)

        # Low-latency action forwarding to local machine
        self._stream_action(raw_action, action_cmd, infer_ms)
        
        # Send to robot
        success = self.robot.send_command(action_cmd)
        
        # Update state (use command or read from robot)
        robot_state = self.robot.get_state()
        if robot_state is not None:
            self.current_state = robot_state
        else:
            # Use raw action as predicted state
            self.current_state = raw_action[:3].copy()
        
        # Record history
        self.action_history.append({
            "frame": self.frame_count,
            "timestamp": datetime.now().isoformat(),
            "raw_action": raw_action.tolist(),
            "command": action_cmd,
            "state": self.current_state.tolist(),
            "instruction": instruction,
            "success": success,
        })
        
        # Format outputs
        t_end = time.time()
        latency_ms = (t_end - t_start) * 1000
        
        raw_str = (
            f"[pitch={raw_action[0]:.4f}, yaw={raw_action[1]:.4f}, grip={raw_action[2]:.4f}]"
        )
        cmd_str = (
            f"[pitch={action_cmd['pitch']:3d}, yaw={action_cmd['yaw']:4d}, grip={action_cmd['grip']:3d}]"
        )
        state_str = (
            f"[pitch={self.current_state[0]:.4f}, yaw={self.current_state[1]:.4f}, "
            f"grip={self.current_state[2]:.4f}]"
        )
        status = (
            f"✓ {latency_ms:.1f}ms" if success else f"✗ {latency_ms:.1f}ms (send failed)"
        )
        
        return (
            raw_str,
            cmd_str,
            state_str,
            status,
        )
    
    def get_stats(self) -> str:
        """Get pipeline statistics."""
        if self.frame_count == 0:
            return "No frames processed yet"
        
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        return (
            f"Frames: {self.frame_count} | "
            f"FPS: {fps:.1f} | "
            f"Elapsed: {elapsed:.1f}s | "
            f"Last instruction: '{self.last_instruction}' | "
            f"Current state: [{self.current_state[0]:.3f}, {self.current_state[1]:.3f}, {self.current_state[2]:.3f}]"
        )
    
    def save_history(self, output_file: str):
        """Save action history to JSONL file."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w") as f:
            for record in self.action_history:
                f.write(json.dumps(record) + "\n")
        
        logger.info(f"Saved {len(self.action_history)} records to {output_file}")
    
    def close(self):
        """Close all connections."""
        logger.info("Closing pipeline...")
        
        if hasattr(self.camera, "close"):
            self.camera.close()
        if hasattr(self.model, "unload_model"):
            self.model.unload_model()
        if hasattr(self.robot, "close"):
            self.robot.close()
        if self.action_stream_sock is not None:
            self.action_stream_sock.close()
        
        # Save history
        history_file = self.config.get("logging", {}).get("history_file")
        if history_file and self.action_history:
            self.save_history(history_file)
        
        logger.info("Pipeline closed")


# ==============================================================================
# Gradio UI
# ==============================================================================
def build_ui(pipeline: RobotControlPipeline):
    """Build Gradio interface."""
    import gradio as gr
    from gradio.components.image_editor import WebcamOptions
    
    with gr.Blocks(title="Real Robot Control Pipeline") as demo:
        gr.Markdown("# Real Robot Control Pipeline")
        gr.Markdown("Frame-by-frame VLA inference + robot control")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                
                image_input = gr.Image(
                    sources=["webcam"],
                    label="Webcam Stream",
                    type="numpy",
                    streaming=True,
                    webcam_options=WebcamOptions(
                        mirror=pipeline.config["camera"].get("mirror", False),
                        constraints={
                            "video": {
                                "facingMode": {"ideal": "environment"},
                                "width": {"ideal": 640},
                                "height": {"ideal": 480},
                            }
                        },
                    ),
                )
                
                instruction_input = gr.Textbox(
                    label="Task Instruction",
                    placeholder="e.g., 'Grasp the can and lift it'",
                    lines=2,
                )

                gr.Markdown("### Manual Control (Override VLA)")
                manual_mode = gr.Checkbox(
                    label="Manual Mode (use buttons instead of VLA output)",
                    value=False,
                )
                manual_step = gr.Number(label="Step", value=10, precision=0)

                pitch_cfg = pipeline.config.get("action_normalization", {}).get("pitch", {})
                yaw_cfg = pipeline.config.get("action_normalization", {}).get("yaw", {})
                grip_cfg = pipeline.config.get("action_normalization", {}).get("grip", {})

                pitch_min = int(pitch_cfg.get("cmd_min", 0))
                pitch_max = int(pitch_cfg.get("cmd_max", 100))
                yaw_min = int(yaw_cfg.get("cmd_min", -100))
                yaw_max = int(yaw_cfg.get("cmd_max", 100))
                grip_min = int(grip_cfg.get("cmd_min", 0))
                grip_max = int(grip_cfg.get("cmd_max", 100))

                manual_pitch = gr.Number(label="Manual Pitch", value=50, precision=0)
                with gr.Row():
                    pitch_minus_btn = gr.Button("Pitch -", scale=1)
                    pitch_plus_btn = gr.Button("Pitch +", scale=1)

                manual_yaw = gr.Number(label="Manual Yaw", value=0, precision=0)
                with gr.Row():
                    yaw_minus_btn = gr.Button("Yaw -", scale=1)
                    yaw_plus_btn = gr.Button("Yaw +", scale=1)

                manual_grip = gr.Number(label="Manual Grip", value=0, precision=0)
                with gr.Row():
                    grip_minus_btn = gr.Button("Grip -", scale=1)
                    grip_plus_btn = gr.Button("Grip +", scale=1)

                send_manual_btn = gr.Button("Send Manual Once", variant="secondary")
                
                auto_infer = gr.Checkbox(label="Auto Infer (streaming)", value=False)
                run_btn = gr.Button("Infer & Execute", variant="primary", scale=2)
            
            with gr.Column(scale=1):
                gr.Markdown("### Output")
                
                raw_action_output = gr.Textbox(
                    label="Raw Model Output",
                    interactive=False,
                )
                cmd_output = gr.Textbox(
                    label="Standardized Command",
                    interactive=False,
                )
                state_output = gr.Textbox(
                    label="Predicted State",
                    interactive=False,
                )
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                )
        
        with gr.Row():
            stats_display = gr.Textbox(
                label="Pipeline Statistics",
                interactive=False,
                scale=4,
            )
            refresh_btn = gr.Button("Refresh Stats", scale=1)
        
        # Callbacks
        def adjust_value(value, step, lower, upper, direction):
            base = 0 if value is None else float(value)
            step_val = 1 if step is None else float(step)
            new_v = base + direction * step_val
            return int(np.clip(int(round(new_v)), lower, upper))

        def on_run(image, instruction, manual_enabled, m_pitch, m_yaw, m_grip):
            if manual_enabled:
                return pipeline.process_manual_command(m_pitch, m_yaw, m_grip, instruction)

            if image is None:
                return "", "", "", "Error: No frame"

            raw_str, cmd_str, state_str, status = pipeline.process_frame(image, instruction)
            return raw_str, cmd_str, state_str, status

        def on_send_manual(instruction, m_pitch, m_yaw, m_grip):
            return pipeline.process_manual_command(m_pitch, m_yaw, m_grip, instruction)
        
        def on_refresh_stats():
            return pipeline.get_stats()
        
        run_btn.click(
            fn=on_run,
            inputs=[image_input, instruction_input, manual_mode, manual_pitch, manual_yaw, manual_grip],
            outputs=[raw_action_output, cmd_output, state_output, status_output],
        )

        send_manual_btn.click(
            fn=on_send_manual,
            inputs=[instruction_input, manual_pitch, manual_yaw, manual_grip],
            outputs=[raw_action_output, cmd_output, state_output, status_output],
        )

        def adjust_and_send(pitch, yaw, grip, step, dim, direction, instruction):
            new_p = adjust_value(pitch, step, pitch_min, pitch_max, direction) if dim == "pitch" else int(pitch or 0)
            new_y = adjust_value(yaw, step, yaw_min, yaw_max, direction) if dim == "yaw" else int(yaw or 0)
            new_g = adjust_value(grip, step, grip_min, grip_max, direction) if dim == "grip" else int(grip or 0)
            raw_str, cmd_str, state_str, status = pipeline.process_manual_command(new_p, new_y, new_g, instruction)
            return new_p, new_y, new_g, raw_str, cmd_str, state_str, status

        _btn_outputs = [manual_pitch, manual_yaw, manual_grip, raw_action_output, cmd_output, state_output, status_output]
        _btn_inputs = [manual_pitch, manual_yaw, manual_grip, manual_step, instruction_input]

        pitch_minus_btn.click(
            fn=lambda p, y, g, s, inst: adjust_and_send(p, y, g, s, "pitch", -1, inst),
            inputs=_btn_inputs, outputs=_btn_outputs,
        )
        pitch_plus_btn.click(
            fn=lambda p, y, g, s, inst: adjust_and_send(p, y, g, s, "pitch", 1, inst),
            inputs=_btn_inputs, outputs=_btn_outputs,
        )
        yaw_minus_btn.click(
            fn=lambda p, y, g, s, inst: adjust_and_send(p, y, g, s, "yaw", -1, inst),
            inputs=_btn_inputs, outputs=_btn_outputs,
        )
        yaw_plus_btn.click(
            fn=lambda p, y, g, s, inst: adjust_and_send(p, y, g, s, "yaw", 1, inst),
            inputs=_btn_inputs, outputs=_btn_outputs,
        )
        grip_minus_btn.click(
            fn=lambda p, y, g, s, inst: adjust_and_send(p, y, g, s, "grip", -1, inst),
            inputs=_btn_inputs, outputs=_btn_outputs,
        )
        grip_plus_btn.click(
            fn=lambda p, y, g, s, inst: adjust_and_send(p, y, g, s, "grip", 1, inst),
            inputs=_btn_inputs, outputs=_btn_outputs,
        )

        # Streaming auto-infer: triggers on every webcam frame when checkbox is on
        image_input.stream(
            fn=lambda img, inst, auto, manual, mp, my, mg: on_run(img, inst, manual, mp, my, mg)
            if auto else ("", "", "", "Auto infer off"),
            inputs=[image_input, instruction_input, auto_infer, manual_mode, manual_pitch, manual_yaw, manual_grip],
            outputs=[raw_action_output, cmd_output, state_output, status_output],
            concurrency_limit=1,
        )
        
        refresh_btn.click(
            fn=on_refresh_stats,
            outputs=stats_display,
        )
        
        # Auto-refresh stats (gr.Timer for newer Gradio versions)
        try:
            timer = gr.Timer(2)
            timer.tick(on_refresh_stats, outputs=stats_display)
        except AttributeError:
            demo.load(on_refresh_stats, outputs=stats_display)
        
        return demo


def main():
    """Main entry point."""
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    
    try:
        pipeline = RobotControlPipeline(config_path)
        
        demo = build_ui(pipeline)
        
        # Configure cloudflare tunnel if enabled
        ui_config = pipeline.config.get("ui", {})
        if ui_config.get("enable_cloudflare", False):
            import subprocess
            import threading
            
            def start_tunnel():
                cloudflare_path = ui_config.get("cloudflare_path", "/home/zhx/bin/cloudflared")
                port = ui_config.get("server_port", 7860)
                subprocess.Popen(
                    [cloudflare_path, "tunnel", "--url", f"http://localhost:{port}"],
                )
            
            threading.Thread(target=start_tunnel, daemon=True).start()
            logger.info("Cloudflare tunnel starting...")
        
        # Launch UI
        demo.launch(
            server_name=ui_config.get("server_name", "0.0.0.0"),
            server_port=ui_config.get("server_port", 7860),
            share=ui_config.get("share", False),
            max_threads=2,
        )
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        if "pipeline" in locals():
            pipeline.close()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
