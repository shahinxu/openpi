from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure `src` is importable regardless of invocation cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import yaml

from src.models.encoder import EncoderConfig
from src.decoders.heads import DecoderConfig
from src.models.three_decoder_model import ThreeDecoderModel, ThreeDecoderModelConfig


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run inference for three-decoder model")
	parser.add_argument("--model-config", type=str, default="configs/model.yaml")
	parser.add_argument("--checkpoint", type=str, required=True)
	parser.add_argument("--video", type=str, required=True)
	parser.add_argument("--start", type=int, default=0)
	parser.add_argument("--window", type=int, default=16)
	parser.add_argument("--output", type=str, default="outputs/infer")
	return parser.parse_args()

def _read_video_clip(video_path: str, start: int, window: int, image_size: int = 224) -> torch.Tensor:
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise FileNotFoundError(video_path)

	n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	start = max(0, min(start, max(n - 1, 0)))
	end = min(n, start + window)
	idx = list(range(start, end))
	if not idx:
		idx = [0]

	frames = []
	mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
	std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
	for i in idx:
		cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
		ok, frame = cap.read()
		if not ok or frame is None:
			continue
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
		frame = frame.astype(np.float32) / 255.0
		frame = (frame - mean) / std
		frames.append(frame)
	cap.release()

	if not frames:
		raise RuntimeError("No frames decoded from video")

	arr = np.stack(frames, axis=0)
	arr = np.transpose(arr, (0, 3, 1, 2))
	return torch.from_numpy(arr).unsqueeze(0)



def _load_yaml(path: str) -> dict:
	with open(path, "r", encoding="utf-8") as f:
		data = yaml.safe_load(f) or {}
	if not isinstance(data, dict):
		raise ValueError(f"Config must be mapping: {path}")
	return data


def main() -> None:
	args = parse_args()
	if not os.path.exists(args.checkpoint):
		raise FileNotFoundError(args.checkpoint)
	if not os.path.exists(args.video):
		raise FileNotFoundError(args.video)

	model_cfg = _load_yaml(args.model_config)
	enc_block = model_cfg.get("model", {})
	dec_block = model_cfg.get("decoder", {})
	model = ThreeDecoderModel(
		ThreeDecoderModelConfig(
			encoder=EncoderConfig(
				backbone=str(enc_block.get("backbone", "resnet18")),
				latent_dim=int(enc_block.get("latent_dim", 256)),
				temporal_layers=int(enc_block.get("temporal_layers", 4)),
				temporal_heads=int(enc_block.get("temporal_heads", 8)),
				dropout=float(enc_block.get("dropout", 0.1)),
				pretrained=bool(enc_block.get("pretrained", True)),
				freeze_backbone=bool(enc_block.get("freeze_backbone", False)),
				use_motion_branch=bool(enc_block.get("use_motion_branch", True)),
				aggregate_last_k=int(enc_block.get("aggregate_last_k", 4)),
			),
			decoder=DecoderConfig(
				latent_dim=int(enc_block.get("latent_dim", 256)),
				hidden_dim=int(dec_block.get("hidden_dim", 512)),
				occ_shape=tuple(int(v) for v in dec_block.get("occ_shape", [7, 15, 15])),
			),
		)
	)

	ckpt = torch.load(args.checkpoint, map_location="cpu")
	model.encoder.load_state_dict(ckpt["encoder_state_dict"])
	model.eval()
	x = _read_video_clip(args.video, args.start, args.window)
	with torch.no_grad():
		preds = model(x)

	os.makedirs(args.output, exist_ok=True)
	out_json = os.path.join(args.output, "prediction.json")
	payload = {
		"input_video": args.video,
		"start": int(args.start),
		"window": int(args.window),
		"trunk6": preds["trunk6"][0].cpu().tolist(),
		"skeleton153": preds["skeleton153"][0].cpu().tolist(),
		"occupancy_shape": list(preds["occupancy"][0].shape),
	}
	with open(out_json, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)

	print(out_json)


if __name__ == "__main__":
	main()
