from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure `src` is importable regardless of invocation cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from src.data.datamodule import DataModuleConfig, ThreeDecoderDataModule
from src.engine.trainer import Trainer, TrainerConfig
from src.losses.multitask_loss import LossConfig, MultiTaskLoss
from src.models.encoder import EncoderConfig
from src.decoders.heads import DecoderConfig
from src.models.three_decoder_model import ThreeDecoderModel, ThreeDecoderModelConfig


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate three-decoder model")
	parser.add_argument("--train-config", type=str, default="configs/train.yaml")
	parser.add_argument("--data-config", type=str, default="configs/data.yaml")
	parser.add_argument("--model-config", type=str, default="configs/model.yaml")
	parser.add_argument("--checkpoint", type=str, required=True)
	parser.add_argument("--split", type=str, default="val", choices=["val"])
	return parser.parse_args()


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

	data_cfg = _load_yaml(args.data_config)
	model_cfg = _load_yaml(args.model_config)
	train_cfg = _load_yaml(args.train_config)

	window_cfg = data_cfg.get("window", {})
	loader_cfg = data_cfg.get("loader", {})
	window_span = int(window_cfg.get("span", window_cfg.get("size", 16)))
	dm_cfg = DataModuleConfig(
		train_manifest="",
		val_manifest=str(data_cfg.get("dataset", {}).get("val_manifest", "outputs/splits/val_manifest.json")),
		batch_size=int(loader_cfg.get("batch_size", 8)),
		num_workers=int(loader_cfg.get("num_workers", 4)),
		window_span=window_span,
		stride=int(window_cfg.get("stride", 1)),
		image_size=int(window_cfg.get("image_size", 224)),
	)
	datamodule = ThreeDecoderDataModule(dm_cfg)

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

	loss_block = train_cfg.get("loss", {})
	loss_fn = MultiTaskLoss(
		LossConfig(
			lambda_trunk=float(loss_block.get("lambda_trunk", 1.0)),
			lambda_skeleton=float(loss_block.get("lambda_skeleton", 1.0)),
			lambda_occupancy=float(loss_block.get("lambda_occupancy", 2.0)),
			occupancy_pos_weight_clip_min=float(loss_block.get("occupancy_pos_weight_clip_min", 1.0)),
			occupancy_pos_weight_clip_max=float(loss_block.get("occupancy_pos_weight_clip_max", 200.0)),
		)
	)

	trainer_block = train_cfg.get("trainer", {})
	trainer = Trainer(
		model=model,
		loss_fn=loss_fn,
		datamodule=datamodule,
		cfg=TrainerConfig(
			epochs=1,
			lr=float(trainer_block.get("lr", 1e-4)),
			weight_decay=float(trainer_block.get("weight_decay", 1e-4)),
			grad_clip_norm=float(trainer_block.get("grad_clip_norm", 1.0)),
			device=str(trainer_block.get("device", "cuda")),
			checkpoint_dir=str(trainer_block.get("checkpoint_dir", "checkpoints")),
			monitor_metric="val_loss",
			monitor_mode="min",
			best_ckpt_name="best_encoder.pt",
		),
	)

	datamodule.setup()
	metrics = trainer.validate_one_epoch(0)
	print(metrics)


if __name__ == "__main__":
	main()
