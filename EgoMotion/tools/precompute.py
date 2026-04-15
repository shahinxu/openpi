from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline precompute for three-decoder training (video-cache layout)."
        )
    )
    parser.add_argument("--manifest", type=str, required=True, help="Input manifest with source videos/npz labels")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for window sample files")
    parser.add_argument(
        "--out-manifest",
        type=str,
        default="",
        help="Optional output manifest path. Defaults to <out-dir>/precomputed_manifest.json",
    )
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--save-video-uint8",
        action="store_true",
        help="Save video as uint8 [T,H,W,C] (smaller disk, normalize at train-time).",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=0,
        help="Optional limit for source samples (0 = all). Useful for smoke generation.",
    )
    return parser.parse_args()


def _load_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a dict: {path}")
    samples = payload.get("samples", [])
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Manifest has empty/non-list samples: {path}")
    return payload


def _prepare_frame(frame: np.ndarray, image_size: int) -> np.ndarray:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return frame


def _decode_video(video_path: str, image_size: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(_prepare_frame(frame, image_size))
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return np.stack(frames, axis=0)


def main() -> None:
    args = parse_args()

    payload = _load_manifest(args.manifest)
    source_samples = payload.get("samples", [])
    if args.limit_samples > 0:
        source_samples = source_samples[: args.limit_samples]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    if args.out_manifest:
        out_manifest = Path(args.out_manifest)
    else:
        out_manifest = out_dir / "precomputed_manifest.json"
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    out_samples = []
    skipped = 0
    total_windows = 0

    for sample_idx, sample in enumerate(source_samples):
        npz_path = sample.get("npz", "")
        video_path = sample.get("rgb_video", "")
        sequence = sample.get("sequence", "unknown")

        if not npz_path or not os.path.exists(npz_path):
            skipped += 1
            continue
        if not video_path or not os.path.exists(video_path):
            skipped += 1
            continue

        with np.load(npz_path) as data:
            trunk6 = data["trunk6"].astype(np.float32)
            skeleton153 = data["skeleton_vel153"].astype(np.float32)
            occupancy = data["occupancy3d"].astype(np.float32)

        frames = _decode_video(video_path, args.image_size)
        n_label_frames = int(trunk6.shape[0])
        if n_label_frames < args.window_size:
            skipped += 1
            continue
        n_video_frames = int(frames.shape[0])

        stem = f"s{sample_idx:05d}"
        cache_path = videos_dir / f"{stem}.npy"
        if args.save_video_uint8:
            cache_arr = frames.astype(np.uint8)
        else:
            video_f = frames.astype(np.float32) / 255.0
            video_f = (video_f - mean) / std
            cache_arr = np.transpose(video_f, (0, 3, 1, 2)).astype(np.float16)
        np.save(cache_path, cache_arr)

        out_samples.append(
            {
                "sequence": sequence,
                "npz": npz_path,
                "rgb_video": video_path,
                "video_cache": str(cache_path),
                "n_label_frames": n_label_frames,
                "n_video_frames": n_video_frames,
            }
        )
        total_windows += max(0, (n_label_frames - args.window_size) // args.stride + 1)

        print(
            f"[precompute] sample={sample_idx + 1}/{len(source_samples)} sequence={sequence} "
            f"windows_so_far={total_windows}",
            flush=True,
        )

    out_payload = {
        "source_manifest": args.manifest,
        "mode": "precomputed_video_cache",
        "window": {
            "size": args.window_size,
            "stride": args.stride,
            "image_size": args.image_size,
            "video_encoding": "uint8_thwc" if args.save_video_uint8 else "float16_tchw_normalized",
            "layout": "video-cache",
        },
        "num_samples": len(out_samples),
        "skipped_source_samples": skipped,
        "samples": out_samples,
    }
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    print(f"[precompute] done out_manifest={out_manifest}")
    print(f"[precompute] num_window_samples={len(out_samples)} skipped_source_samples={skipped}")


if __name__ == "__main__":
    main()
