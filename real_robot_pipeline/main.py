"""
Real Robot Control Pipeline - Main Entry Point

High-level orchestration of camera, model, normalization, and robot control.
"""
import logging
import json
import time
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
        raw_action = self.model.infer(image, instruction, self.current_state)
        
        # Normalize to command format
        action_cmd = self.normalizer.normalize(raw_action)
        
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
        def on_run(image, instruction):
            if image is None:
                return "", "", "", "Error: No frame"
            
            raw_str, cmd_str, state_str, status = pipeline.process_frame(image, instruction)
            return raw_str, cmd_str, state_str, status
        
        def on_refresh_stats():
            return pipeline.get_stats()
        
        run_btn.click(
            fn=on_run,
            inputs=[image_input, instruction_input],
            outputs=[raw_action_output, cmd_output, state_output, status_output],
        )
        
        refresh_btn.click(
            fn=on_refresh_stats,
            outputs=stats_display,
        )
        
        # Auto-refresh stats
        demo.load(on_refresh_stats, outputs=stats_display, every=2)
        
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
