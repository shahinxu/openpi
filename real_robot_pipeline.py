"""
Real robot inference pipeline for Hannes robot.
- Captures frame-by-frame input + instruction
- Runs inference using trained VLA model
- Outputs standardized action commands
- Controls real robot and updates state
"""
import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import gradio as gr
import torch
from PIL import Image

# ==============================================================================
# Configuration
# ==============================================================================
MODEL_REPO_ID = "hannes/dataset_new"  # Or your fine-tuned checkpoint
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INFERENCE_FREQ = 10  # Hz

# Action normalization ranges
ACTION_RANGES = {
    "wrist_pitch": {"min": 0, "max": 100},      # [0, 100]
    "wrist_yaw": {"min": -100, "max": 100},     # [-100, 100]
    "grip": {"min": 0, "max": 100},             # [0, 100]
}

# ==============================================================================
# Model Loading
# ==============================================================================
def load_model():
    """Load trained VLA model from HuggingFace or local checkpoint."""
    try:
        # Try loading from huggingface
        from transformers import AutoModelForCausalLM, AutoProcessor
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO_ID, 
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to(DEVICE)
        processor = AutoProcessor.from_pretrained(MODEL_REPO_ID, trust_remote_code=True)
        return model, processor
    except Exception as e:
        print(f"Warning: Could not load from HF: {e}")
        print("Please ensure model checkpoint is available.")
        return None, None


# ==============================================================================
# Action Inference
# ==============================================================================
def infer_action(
    model,
    processor,
    image: np.ndarray,
    instruction: str,
    prev_state: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Run inference on current frame + instruction.
    
    Args:
        model: Loaded VLA model
        processor: Model processor/tokenizer
        image: Current frame (H, W, 3) uint8
        instruction: Text instruction
        prev_state: Previous state [pitch, yaw, grip] for context
        
    Returns:
        action: Raw action [pitch, yaw, grip, ...] from model
    """
    if model is None:
        # Dummy fallback
        return np.array([0.5, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    # Prepare input
    image_pil = Image.fromarray(image)
    
    # Tokenize instruction
    text_input = f"Task: {instruction}"
    if prev_state is not None:
        text_input += f"\nCurrent state: pitch={prev_state[0]:.3f}, yaw={prev_state[1]:.3f}, grip={prev_state[2]:.3f}"
    
    # Run inference
    with torch.no_grad():
        inputs = processor(
            text=text_input,
            images=[image_pil],
            return_tensors="pt"
        ).to(DEVICE)
        
        outputs = model.generate(**inputs, max_new_tokens=128)
        action_str = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Parse action from output (format may vary - adjust as needed)
    # For now, use dummy parsing
    try:
        # Try to extract numbers from output
        action = np.array([0.5, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    except:
        action = np.array([0.5, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    return action


# ==============================================================================
# Action Post-processing
# ==============================================================================
def standardize_action(action: np.ndarray) -> Dict[str, int]:
    """
    Convert raw action to standardized integer commands.
    
    Args:
        action: [pitch, yaw, grip, ...] in original range
        
    Returns:
        Standardized commands: {pitch, yaw, grip} as integers
    """
    # Denormalize from [-1, 1] or [0, 1] to actual ranges
    # Assuming model outputs in [-1, 1] or [0, 1]
    
    pitch_raw = action[0]  # Assumed in [0, 0.5] from data
    yaw_raw = action[1]    # Assumed in [-1, 0.7] from data
    grip_raw = action[2]   # Assumed in [0, 1] from data
    
    # Map to standardized ranges and scale by 100
    # pitch: [0.48, 0.52] -> [0, 100]
    pitch_min, pitch_max = 0.48, 0.52
    pitch_normalized = (pitch_raw - pitch_min) / (pitch_max - pitch_min)
    pitch_cmd = int(np.clip(pitch_normalized * 100, 0, 100))
    
    # yaw: [-1.02, 0.69] -> [-100, 100]
    yaw_min, yaw_max = -1.02, 0.69
    yaw_normalized = (yaw_raw - yaw_min) / (yaw_max - yaw_min) * 2 - 1
    yaw_cmd = int(np.clip(yaw_normalized * 100, -100, 100))
    
    # grip: [0, 1] -> [0, 100]
    grip_cmd = int(np.clip(grip_raw * 100, 0, 100))
    
    return {
        "pitch": pitch_cmd,
        "yaw": yaw_cmd,
        "grip": grip_cmd,
    }


# ==============================================================================
# Robot Control Interface (Placeholder)
# ==============================================================================
class RobotController:
    """Interface to real robot. Implement based on your hardware."""
    
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200):
        """
        Initialize robot controller.
        
        Args:
            port: Serial port (e.g., "/dev/ttyUSB0") or ROS topic name
            baudrate: Serial communication speed
        """
        self.port = port
        self.connected = False
        self._connect()
    
    def _connect(self):
        """Connect to robot (implement based on your interface)."""
        # TODO: Implement actual connection
        # Examples:
        # - Serial port: serial.Serial(self.port, self.baudrate)
        # - ROS: rospy.Publisher(...), rospy.Subscriber(...)
        # - REST API: requests.post(...)
        self.connected = True
        print(f"[INFO] Robot controller connected (simulated)")
    
    def send_command(self, action_cmd: Dict[str, int]):
        """
        Send standardized action command to robot.
        
        Args:
            action_cmd: {pitch, yaw, grip} with integer values
        """
        if not self.connected:
            print("[ERROR] Robot not connected")
            return False
        
        # TODO: Implement actual command sending
        # Example serial format: "P{pitch},Y{yaw},G{grip}\n"
        command = f"P{action_cmd['pitch']},Y{action_cmd['yaw']},G{action_cmd['grip']}\n"
        
        print(f"[CMD] {command.strip()}")
        
        # Uncomment when ready:
        # self.serial_port.write(command.encode())
        
        return True
    
    def get_state(self) -> Optional[np.ndarray]:
        """
        Read current robot state (or use predicted state).
        
        Returns:
            [pitch, yaw, grip] or None if unavailable
        """
        # TODO: Implement actual state reading from robot
        # For now, return None to use predicted state
        return None
    
    def close(self):
        """Close robot connection."""
        # TODO: Implement cleanup
        self.connected = False


# ==============================================================================
# Gradio UI
# ==============================================================================
class RealRobotPipeline:
    """Unified pipeline combining model + UI + robot control."""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.robot = RobotController()
        self.current_state = np.array([0.5, -1.0, 0.0], dtype=np.float32)  # Initial state
        self.instruction = ""
        self.action_history = []
    
    def load_models(self):
        """Load model in background."""
        print("[INFO] Loading model...")
        self.model, self.processor = load_model()
        if self.model:
            print("[INFO] Model loaded successfully")
        else:
            print("[WARNING] Model loading failed - using dummy inference")
    
    def process_frame(
        self,
        image: np.ndarray,
        instruction: str,
    ) -> Tuple[Dict, Dict, str]:
        """
        Process single frame: infer action + control robot.
        
        Args:
            image: Current frame from camera
            instruction: Task instruction
            
        Returns:
            (raw_action, standardized_cmd, status_message)
        """
        self.instruction = instruction
        
        # Run inference
        raw_action = infer_action(
            self.model, self.processor, image, instruction, self.current_state
        )
        
        # Standardize action
        action_cmd = standardize_action(raw_action)
        
        # Send to robot
        success = self.robot.send_command(action_cmd)
        
        # Update state (use action as next state or read from robot)
        robot_state = self.robot.get_state()
        if robot_state is not None:
            self.current_state = robot_state
        else:
            # Use action as predicted state
            self.current_state = raw_action[:3]
        
        # Record history
        self.action_history.append({
            "raw_action": raw_action.tolist(),
            "cmd": action_cmd,
            "state": self.current_state.tolist(),
        })
        
        # Format output
        raw_str = f"pitch={raw_action[0]:.4f}, yaw={raw_action[1]:.4f}, grip={raw_action[2]:.4f}"
        cmd_str = f"pitch={action_cmd['pitch']}, yaw={action_cmd['yaw']}, grip={action_cmd['grip']}"
        status = f"✓ Command sent" if success else "✗ Command failed"
        
        return raw_str, cmd_str, status
    
    def build_ui(self):
        """Build Gradio interface."""
        with gr.Blocks(title="Real Robot Control Pipeline") as demo:
            gr.Markdown("# Real Robot Inference Pipeline")
            gr.Markdown("Frame-by-frame action prediction and robot control")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Input")
                    image_input = gr.Image(label="Camera Frame", type="numpy")
                    instruction_input = gr.Textbox(
                        label="Task Instruction",
                        placeholder="e.g., 'Grasp the can and lift it'"
                    )
                    run_btn = gr.Button("Infer & Execute", variant="primary")
                
                with gr.Column():
                    gr.Markdown("### Output")
                    raw_action_output = gr.Textbox(label="Raw Action", interactive=False)
                    cmd_output = gr.Textbox(label="Standardized Command", interactive=False)
                    status_output = gr.Textbox(label="Status", interactive=False)
            
            with gr.Row():
                gr.Markdown("### State Tracking")
                state_display = gr.Textbox(
                    label="Current State",
                    value=f"[{self.current_state[0]:.4f}, {self.current_state[1]:.4f}, {self.current_state[2]:.4f}]",
                    interactive=False
                )
                action_count = gr.Number(
                    label="Actions Executed",
                    value=0,
                    interactive=False
                )
            
            # Connect buttons
            def on_run(image, instruction):
                if image is None:
                    return "", "", "Error: No image provided", state_display.value, len(self.action_history)
                
                raw_str, cmd_str, status = self.process_frame(image, instruction)
                
                # Update state display
                state_str = f"[{self.current_state[0]:.4f}, {self.current_state[1]:.4f}, {self.current_state[2]:.4f}]"
                
                return raw_str, cmd_str, status, state_str, len(self.action_history)
            
            run_btn.click(
                fn=on_run,
                inputs=[image_input, instruction_input],
                outputs=[raw_action_output, cmd_output, status_output, state_display, action_count]
            )
            
            return demo
    
    def launch(self, share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7860):
        """Launch Gradio UI."""
        print("[INFO] Loading models...")
        self.load_models()
        
        print("[INFO] Building UI...")
        demo = self.build_ui()
        
        print(f"[INFO] Launching at http://{server_name}:{server_port}")
        demo.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            show_error=True
        )


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    pipeline = RealRobotPipeline()
    pipeline.launch(share=False, server_port=7860)
