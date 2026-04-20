#!/usr/bin/env python
"""Offline z_t caching: run frozen EgoMotion encoder on HDF5 episode datasets.

For each HDF5 file, reads episode images, builds 16-frame sliding windows for
every timestep (with earliest-frame left-padding), runs the frozen encoder in
batch, and saves a [T, 256] float32 .npy file alongside the original HDF5.

Usage (from workspace root):
    conda activate openpi311
    python backups/pi05_egomotion_z_patch_20260420/cache_egomotion_z.py

Default behaviour processes dataset_auto_grip_5/ and dataset_auto_grip_6/.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Resolve workspace root so that ``EgoMotion`` is importable.
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]  # -> openpi/
sys.path.insert(0, str(WORKSPACE_ROOT / "EgoMotion"))

from src.models.encoder import EncoderConfig, VideoEncoder  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SPAN = 16  # 16-frame window expected by VideoEncoder
IMAGE_SIZE = 224
ENCODER_CKPT = str(WORKSPACE_ROOT / "EgoMotion" / "checkpoints" / "best_encoder.pt")

# Candidate image dataset names (tried in order)
IMAGE_KEYS = ("eye_view_images", "agent_view_images", "agentview_images")

DEFAULT_DIRS = [
    str(WORKSPACE_ROOT / "dataset_auto_grip_5"),
    str(WORKSPACE_ROOT / "dataset_auto_grip_6"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_image_key(episode_group: h5py.Group) -> str:
    """Return the first matching image dataset key in *episode_group*."""
    for key in IMAGE_KEYS:
        if key in episode_group:
            return key
    raise KeyError(
        f"No known image key found in episode group. "
        f"Available keys: {list(episode_group.keys())}"
    )


def _preprocess_images(images: np.ndarray) -> np.ndarray:
    """Resize & normalise a batch of uint8 RGB images.

    Args:
        images: [N, H_orig, W_orig, 3] uint8

    Returns:
        [N, 3, 224, 224] float32 (ImageNet-normalised)
    """
    out = np.empty((len(images), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    for i, img in enumerate(images):
        resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        out[i] = resized.astype(np.float32) / 255.0
    # Normalise
    out = (out - IMAGENET_MEAN) / IMAGENET_STD
    # HWC -> CHW
    out = out.transpose(0, 3, 1, 2)  # [N, 3, 224, 224]
    return out


def _build_window(frames: np.ndarray, end_idx: int, span: int = SPAN) -> np.ndarray:
    """Return a *span*-length window ending at *end_idx* with left-padding."""
    start = max(0, end_idx - span + 1)
    window = frames[start: end_idx + 1]
    missing = span - len(window)
    if missing > 0:
        pad = np.repeat(window[:1], missing, axis=0)
        window = np.concatenate([pad, window], axis=0)
    return window


def _load_encoder(ckpt_path: str, device: torch.device) -> VideoEncoder:
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


@torch.no_grad()
def _encode_episode(
    encoder: VideoEncoder,
    images_preprocessed: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Run encoder on all 16-frame windows for one episode.

    Args:
        images_preprocessed: [T, 3, 224, 224] float32 (already normalised).
        batch_size: Number of windows to forward in parallel.

    Returns:
        [T, 256] float32 numpy array.
    """
    T = len(images_preprocessed)
    z_all = np.empty((T, 256), dtype=np.float32)

    for batch_start in range(0, T, batch_size):
        batch_end = min(batch_start + batch_size, T)
        windows = []
        for t in range(batch_start, batch_end):
            win = _build_window(images_preprocessed, t, SPAN)  # [16, 3, 224, 224]
            windows.append(win)
        # [B, 16, 3, 224, 224]
        batch_tensor = torch.from_numpy(np.stack(windows, axis=0)).to(device)
        z = encoder(batch_tensor)  # [B, 256]
        z_all[batch_start:batch_end] = z.cpu().numpy()

    return z_all


def _output_path(hdf5_path: str, output_dir: str | None) -> str:
    """Derive the .npy output path from the source HDF5 path."""
    base = Path(hdf5_path).stem  # e.g. "place_can_near_bread_001"
    if output_dir is None:
        # Save next to the original HDF5
        return str(Path(hdf5_path).with_suffix(".egomotion_z.npy"))
    else:
        # Preserve per-directory structure inside output_dir
        dataset_dir_name = Path(hdf5_path).parent.name  # e.g. "dataset_auto_grip_5"
        dest = Path(output_dir) / dataset_dir_name
        dest.mkdir(parents=True, exist_ok=True)
        return str(dest / f"{base}.egomotion_z.npy")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_file(
    hdf5_path: str,
    encoder: VideoEncoder,
    device: torch.device,
    batch_size: int,
    output_dir: str | None,
    overwrite: bool,
) -> bool:
    """Process one HDF5 file.  Returns True if z_t was written."""
    npy_path = _output_path(hdf5_path, output_dir)
    if not overwrite and os.path.exists(npy_path):
        return False

    with h5py.File(hdf5_path, "r") as f:
        ep = f["episode_0"]
        img_key = _find_image_key(ep)
        images_raw = ep[img_key][:]  # [T, H, W, 3] uint8

    images_pp = _preprocess_images(images_raw)  # [T, 3, 224, 224]
    z_t = _encode_episode(encoder, images_pp, device, batch_size)  # [T, 256]

    os.makedirs(os.path.dirname(npy_path) or ".", exist_ok=True)
    np.save(npy_path, z_t)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache EgoMotion z_t for HDF5 episodes")
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=DEFAULT_DIRS,
        help="Dataset directories containing .hdf5 files",
    )
    parser.add_argument(
        "--encoder-ckpt",
        default=ENCODER_CKPT,
        help="Path to best_encoder.pt",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="If set, write .npy files into this directory (preserving dataset sub-dirs). "
             "Default: save .npy alongside each .hdf5.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true", help="Re-compute even if .npy exists")
    args = parser.parse_args()

    # Collect all HDF5 files
    hdf5_files: list[str] = []
    for d in args.dirs:
        found = sorted(glob.glob(os.path.join(d, "*.hdf5")))
        if not found:
            print(f"WARNING: no .hdf5 files found in {d}")
        hdf5_files.extend(found)

    if not hdf5_files:
        print("No HDF5 files to process.")
        return

    print(f"Found {len(hdf5_files)} HDF5 files across {len(args.dirs)} directories.")
    print(f"Encoder checkpoint: {args.encoder_ckpt}")
    print(f"Device: {args.device}  |  Batch size: {args.batch_size}")
    print(f"Output: {'alongside HDF5' if args.output_dir is None else args.output_dir}")
    print()

    device = torch.device(args.device)
    encoder = _load_encoder(args.encoder_ckpt, device)

    written = 0
    skipped = 0
    t0 = time.time()

    for i, path in enumerate(hdf5_files, 1):
        name = os.path.basename(path)
        try:
            did_write = process_file(path, encoder, device, args.batch_size, args.output_dir, args.overwrite)
            if did_write:
                written += 1
                elapsed = time.time() - t0
                print(f"[{i}/{len(hdf5_files)}] {name}  ->  saved  ({elapsed:.1f}s)")
            else:
                skipped += 1
                print(f"[{i}/{len(hdf5_files)}] {name}  ->  skipped (exists)")
        except Exception as exc:
            print(f"[{i}/{len(hdf5_files)}] {name}  ->  ERROR: {exc}")

    elapsed = time.time() - t0
    print(f"\nDone. written={written}  skipped={skipped}  total_time={elapsed:.1f}s")


if __name__ == "__main__":
    main()
