import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import websockets.sync.client

from openpi_client import msgpack_numpy

# ---------------------------------------------------------------------------
# EgoMotion encoder (optional – only loaded when --egomotion-ckpt is set)
# ---------------------------------------------------------------------------
_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORKSPACE_ROOT / "EgoMotion"))

_EGOMOTION_SPAN = 16
_EGOMOTION_IMAGE_SIZE = 224
_MOTION_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_MOTION_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def _load_egomotion_encoder(ckpt_path: str, device: torch.device):
    from src.models.encoder import EncoderConfig, VideoEncoder  # noqa: E402

    cfg = EncoderConfig(
        backbone="resnet18",
        latent_dim=256,
        temporal_layers=4,
        temporal_heads=8,
        dropout=0.0,
        pretrained=False,
        freeze_backbone=False,
        use_motion_branch=True,
        aggregate_last_k=4,
    )
    encoder = VideoEncoder(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.to(device)
    encoder.eval()
    return encoder


def _precompute_egomotion_z(images_uint8: np.ndarray, encoder, device: torch.device) -> np.ndarray:
    from src.data.motion_features import apply_motion_feature  # noqa: E402

    T = len(images_uint8)
    # Resize to 224x224 and normalise to [0, 1]
    images_01 = np.empty((T, _EGOMOTION_IMAGE_SIZE, _EGOMOTION_IMAGE_SIZE, 3), dtype=np.float32)
    for i, img in enumerate(images_uint8):
        resized = cv2.resize(img, (_EGOMOTION_IMAGE_SIZE, _EGOMOTION_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        images_01[i] = resized.astype(np.float32) / 255.0

    z_all = np.empty((T, 256), dtype=np.float32)
    batch_size = 32  # windows per forward pass

    for batch_start in range(0, T, batch_size):
        batch_end = min(batch_start + batch_size, T)
        windows = []
        for t in range(batch_start, batch_end):
            start_idx = max(0, t - _EGOMOTION_SPAN + 1)
            window = images_01[start_idx: t + 1]  # [<=16, H, W, 3]
            missing = _EGOMOTION_SPAN - len(window)
            if missing > 0:
                pad = np.repeat(window[:1], missing, axis=0)
                window = np.concatenate([pad, window], axis=0)
            # Apply frame_diff motion feature and normalise
            clip = apply_motion_feature(window, mode="frame_diff")  # [16, H, W, 3] in [0,1]
            clip = (clip - _MOTION_MEAN) / _MOTION_STD
            clip = clip.transpose(0, 3, 1, 2).astype(np.float32)  # [16, 3, H, W]
            windows.append(clip)
        batch_tensor = torch.from_numpy(np.stack(windows, axis=0)).to(device)
        with torch.no_grad():
            z = encoder(batch_tensor)  # [B, 256]
        z_all[batch_start:batch_end] = z.cpu().numpy()

    return z_all


class _SimpleWebsocketClient:
    def __init__(self, host: str = "0.0.0.0", port: int | None = None):
        if host.startswith("ws"):
            uri = host
        else:
            uri = f"ws://{host}"
        if port is not None:
            uri += f":{port}"

        self._packer = msgpack_numpy.Packer()
        self._ws, self._metadata = self._wait_for_server(uri)

    def _wait_for_server(self, uri: str):
        while True:
            try:
                conn = websockets.sync.client.connect(
                    uri,
                    compression=None,
                    max_size=None,
                    ping_interval=None,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                time.sleep(5)

    def get_server_metadata(self) -> dict:
        return self._metadata

    def infer(self, obs: dict) -> dict:
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass


def load_episode(hdf5_path: str, episode: int = 0):
    with h5py.File(hdf5_path, "r") as f:
        ep_key = f"episode_{episode}"
        if ep_key not in f:
            raise KeyError(f"Episode group {ep_key} not found in {hdf5_path}")
        g = f[ep_key]
        actions = np.asarray(g["actions"], dtype=np.float32)
        states = np.asarray(g["states"], dtype=np.float32) if "states" in g else None
        wrist = None
        # Support legacy two-camera layouts and the current single-agentview layout.
        if "frontview_images" in g and "agentview_images" in g:
            image = np.asarray(g["frontview_images"], dtype=np.uint8)
            wrist = np.asarray(g["agentview_images"], dtype=np.uint8)
        elif "agentview_images" in g and "sideview_images" in g:
            image = np.asarray(g["agentview_images"], dtype=np.uint8)
            wrist = np.asarray(g["sideview_images"], dtype=np.uint8)
        elif "agentview_images" in g:
            image = np.asarray(g["agentview_images"], dtype=np.uint8)
        elif "agent_view_images" in g:
            image = np.asarray(g["agent_view_images"], dtype=np.uint8)
        elif "eye_view_images" in g:
            image = np.asarray(g["eye_view_images"], dtype=np.uint8)
        else:
            available = list(g.keys())
            raise KeyError(
                f"Unsupported image layout in {hdf5_path} under {ep_key}: "
                f"expected frontview+agentview, agentview+sideview, or agentview-only, got keys: {available}"
            )
        task = g.attrs.get("task", None)
    return actions, states, image, wrist, task


def style_eval_plot():
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "axes.linewidth": 1.1,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "legend.frameon": False,
            "lines.linewidth": 2.2,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.22,
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hannes policy vs demo actions on a single HDF5 file.")
    parser.add_argument(
        "--hdf5",
        type=str,
        default="dataset_hannes_total/Hold the milk carton_002.hdf5"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000
    )
    parser.add_argument(
        "--prompt",
        type=str
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="plots/eval_Hold_the_milk_carton_v2.png"
    )
    parser.add_argument(
        "--egomotion-ckpt",
        type=str,
        default=str(_WORKSPACE_ROOT / "EgoMotion" / "checkpoints" / "best_encoder_frame_diff.pt"),
        help="Path to EgoMotion encoder checkpoint.  Set to 'none' to disable z_t conditioning.",
    )
    parser.add_argument(
        "--egomotion-npy",
        type=str,
        default=None,
        help="Path to pre-computed egomotion z .npy file [T, 256]. Skips encoder.",
    )
    args = parser.parse_args()

    print(f"Loading demo from {args.hdf5}")
    gt_actions, gt_states, image_imgs, wrist_imgs, task_attr = load_episode(args.hdf5, episode=0)

    if args.max_steps is not None:
        T = min(args.max_steps, gt_actions.shape[0], image_imgs.shape[0])
    else:
        T = min(gt_actions.shape[0], image_imgs.shape[0])

    if wrist_imgs is not None:
        T = min(T, wrist_imgs.shape[0])

    if gt_states is not None:
        T = min(T, gt_states.shape[0])
        if gt_states.ndim == 1:
            gt_states = gt_states.reshape(T, -1)
        if gt_states.shape[1] < 8:
            pad = np.zeros((gt_states.shape[0], 8 - gt_states.shape[1]), dtype=np.float32)
            gt_states = np.concatenate([gt_states, pad], axis=1)
        elif gt_states.shape[1] > 8:
            gt_states = gt_states[:, :8]
    else:
        gt_states = np.zeros((T, 8), dtype=np.float32)

    # Decide prompt
    prompt = (
        args.prompt
        or (task_attr.decode("utf-8") if isinstance(task_attr, (bytes, bytearray)) else task_attr)
        or os.path.basename(args.hdf5).split(".")[0]
    )
    print(f"Using prompt: {prompt!r}")
    print(f"Episode length (frames used): {T}")

    # ------------------------------------------------------------------
    # Precompute EgoMotion z_t for all frames
    # ------------------------------------------------------------------
    ego_motion_z_all: np.ndarray | None = None
    if args.egomotion_npy is not None and os.path.exists(args.egomotion_npy):
        print(f"Loading pre-computed egomotion z from {args.egomotion_npy} ...")
        ego_motion_z_all = np.load(args.egomotion_npy)[:T].astype(np.float32)
        print(f"z_t shape (from npy): {ego_motion_z_all.shape}")
    else:
        use_egomotion = args.egomotion_ckpt.lower() != "none" and os.path.exists(args.egomotion_ckpt)
        if use_egomotion:
            print(f"Loading EgoMotion encoder from {args.egomotion_ckpt} ...")
            ego_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ego_encoder = _load_egomotion_encoder(args.egomotion_ckpt, ego_device)
            print("Precomputing EgoMotion z_t for all frames ...")
            ego_motion_z_all = _precompute_egomotion_z(image_imgs[:T], ego_encoder, ego_device)
            print(f"z_t shape: {ego_motion_z_all.shape}")
        else:
            print("EgoMotion encoder not loaded – skipping z_t conditioning.")

    pred_actions = np.zeros((T, 6), dtype=np.float32)

    chunk_size = args.chunk_size if args.chunk_size and args.chunk_size > 0 else T

    first_connect = True
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        print(f"Connecting to policy server at {args.host}:{args.port} for steps [{start}, {end}) ...")
        policy = _SimpleWebsocketClient(host=args.host, port=args.port)
        if first_connect:
            print("Connected. Server metadata:", policy.get_server_metadata())
            first_connect = False

        for t in range(start, end):
            obs = {
                "observation/state": gt_states[t],
                "observation/image": image_imgs[t],
                "prompt": prompt,
            }
            if wrist_imgs is not None:
                obs["observation/wrist_image"] = wrist_imgs[t]
            if ego_motion_z_all is not None:
                obs["ego_motion_z"] = ego_motion_z_all[t]
            out = policy.infer(obs)
            actions_seq = np.asarray(out["actions"], dtype=np.float32)
            pred = actions_seq[0, :6]
            pred_actions[t] = pred

            if t < 5:
                print(f"t={t}: gt={gt_actions[t]}, pred={pred}")

        # 显式关闭连接，避免长时间空闲被 keepalive 认为超时
        try:
            if hasattr(policy, "_ws"):
                policy._ws.close()  # type: ignore[attr-defined]
        except Exception:
            pass
    diff = pred_actions - gt_actions[:T]

    l2_per_step = np.linalg.norm(diff, axis=-1)
    mae_per_dim = np.mean(np.abs(diff), axis=0)
    print("\n=== Evaluation summary ===")
    print(f"Mean L2 error per step: {l2_per_step.mean():.4f}")
    print(f"Std  L2 error per step: {l2_per_step.std():.4f}")
    print("Mean absolute error per dim (6 dims):", mae_per_dim)

    # Plot / save per-dimension trajectories: gt vs pred
    if not args.no_show or args.save_path is not None:
        style_eval_plot()
        t_axis = np.arange(T)
        plot_dims = 3
        dim_names = ["wrist_pitch", "wrist_yaw", "grip_mean"]
        gt_color = "#1f4e79"
        pred_color = "#c84c09"

        fig, axes = plt.subplots(plot_dims, 1, figsize=(11.5, 7.8), sharex=True)
        fig.suptitle(
            "Ground Truth vs Predicted Actions\n"
            f"{os.path.basename(args.hdf5)} | mean L2={l2_per_step.mean():.4f}",
            y=0.985,
            fontweight="bold",
        )

        for i in range(plot_dims):
            ax = axes[i]
            ax.plot(t_axis, gt_actions[:T, i], label="Ground truth", color=gt_color)
            ax.plot(
                t_axis,
                pred_actions[:, i],
                label="Prediction",
                color=pred_color,
                linestyle="--",
                linewidth=2.0,
            )
            ax.set_ylabel(dim_names[i])
            ax.set_ylim(-1.0, 1.0)
            ax.set_xlim(0, max(T - 1, 1))
            ax.grid(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if i == 0:
                ax.legend(loc="upper right", ncol=2, handlelength=2.8)

        axes[-1].set_xlabel("Time step")
        fig.align_ylabels(axes)
        plt.tight_layout(rect=(0.03, 0.03, 1, 0.95))

        if args.save_path is not None:
            out_path = args.save_path
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            fig.savefig(out_path, dpi=150)
            print(f"Saved plot to {out_path}")

        if not args.no_show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
