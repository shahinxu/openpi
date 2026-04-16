from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml

from src.data.datamodule import DataModuleConfig, ThreeDecoderDataModule
from src.data.cache import PrecomputeRequest, prepare_precompute_manifest
from src.data.splits import SplitConfig, build_split_manifests
from src.engine.trainer import Trainer, TrainerConfig
from src.losses.multitask_loss import LossConfig, MultiTaskLoss
from src.models.three_decoder_model import ThreeDecoderModel, ThreeDecoderModelConfig
from src.models.encoder import EncoderConfig
from src.decoders.heads import DecoderConfig


def _log_stage(message: str) -> None:
	print(f"[train] {message}", flush=True)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train three-decoder model")
	parser.add_argument("--train-config", type=str, default="configs/train.yaml")
	parser.add_argument("--data-config", type=str, default="configs/data.yaml")
	parser.add_argument("--model-config", type=str, default="configs/model.yaml")
	parser.add_argument("--manifest", type=str, default="adt_pipeline_data/processed/manifest.json")
	parser.add_argument("--split-dir", type=str, default="outputs/splits")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument(
		"--precompute-before-train",
		action="store_true",
		help="Decode videos offline into cached per-video arrays before training.",
	)
	parser.add_argument(
		"--precompute-out-dir",
		type=str,
		default="",
		help="Directory for precomputed samples. Default: outputs/precompute_cache_<config-hash>",
	)
	parser.add_argument(
		"--precompute-out-manifest",
		type=str,
		default="",
		help="Optional output manifest path for precomputed cache metadata.",
	)
	parser.add_argument(
		"--precompute-video-format",
		type=str,
		choices=["uint8", "float16"],
		default="uint8",
		help="Storage format for precomputed clips.",
	)
	parser.add_argument(
		"--precompute-limit-samples",
		type=int,
		default=0,
		help="Optional source sample limit for precompute (0 means all).",
	)
	parser.add_argument(
		"--cleanup-precompute",
		action="store_true",
		help="Remove the precomputed output directory after training finishes.",
	)
	return parser.parse_args()


def _load_yaml(path: str) -> dict:
	with open(path, "r", encoding="utf-8") as f:
		data = yaml.safe_load(f) or {}
	if not isinstance(data, dict):
		raise ValueError(f"Config must be mapping: {path}")
	return data


def _set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	# Avoid eager CUDA probing at startup, which can hang on some driver/runtime setups.
	# Set TRAIN_ENABLE_CUDA_SEED=1 to opt in when CUDA init is known healthy.
	if os.environ.get("TRAIN_ENABLE_CUDA_SEED", "0") == "1":
		torch.cuda.manual_seed_all(seed)


def main() -> None:
	args = parse_args()
	_log_stage("starting")
	_set_seed(args.seed)
	_log_stage(f"seed set to {args.seed}")

	_log_stage("loading configs")
	data_cfg = _load_yaml(args.data_config)
	model_cfg = _load_yaml(args.model_config)
	train_cfg = _load_yaml(args.train_config)

	manifest_for_split = args.manifest
	cleanup_precompute_dir = ""
	if args.precompute_before_train:
		window_block = data_cfg.get("window", {})
		window_span = int(window_block.get("span", window_block.get("size", 16)))
		precompute_result = prepare_precompute_manifest(
			PrecomputeRequest(
				enabled=True,
				source_manifest=args.manifest,
				data_config_path=args.data_config,
				window_size=window_span,
				stride=int(window_block.get("stride", 1)),
				image_size=int(window_block.get("image_size", 224)),
				precompute_out_dir=args.precompute_out_dir,
				precompute_out_manifest=args.precompute_out_manifest,
				precompute_video_format=args.precompute_video_format,
				precompute_limit_samples=args.precompute_limit_samples,
			),
			logger=_log_stage,
		)
		manifest_for_split = precompute_result.manifest_path
		_log_stage(f"precompute manifest ready at {manifest_for_split}")
		if args.cleanup_precompute:
			cleanup_precompute_dir = precompute_result.cache_dir

	split_block = data_cfg.get("split", {})
	split_cfg = SplitConfig(
		train_ratio=float(split_block.get("train_ratio", 0.9)),
		val_ratio=float(split_block.get("val_ratio", 0.1)),
		seed=int(split_block.get("seed", args.seed)),
		group_key=str(data_cfg.get("dataset", {}).get("split_group_key", "sequence")),
	)
	out_split_dir = args.split_dir
	os.makedirs(out_split_dir, exist_ok=True)
	_log_stage(f"building split manifests from {manifest_for_split}")
	split_paths = build_split_manifests(manifest_for_split, out_split_dir, split_cfg)
	_log_stage(
		f"split ready train={split_paths['train']} val={split_paths['val']}"
	)

	window_cfg = data_cfg.get("window", {})
	loader_cfg = data_cfg.get("loader", {})
	window_span = int(window_cfg.get("span", window_cfg.get("size", 16)))
	dm_cfg = DataModuleConfig(
		train_manifest=split_paths["train"],
		val_manifest=split_paths["val"],
		batch_size=int(loader_cfg.get("batch_size", 8)),
		num_workers=int(loader_cfg.get("num_workers", 4)),
		window_span=window_span,
		stride=int(window_cfg.get("stride", 1)),
		image_size=int(window_cfg.get("image_size", 224)),
		persistent_workers=bool(loader_cfg.get("persistent_workers", True)),
		prefetch_factor=int(loader_cfg.get("prefetch_factor", 2)),
	)
	_log_stage("constructing datamodule")
	datamodule = ThreeDecoderDataModule(dm_cfg)

	enc_block = model_cfg.get("model", {})
	dec_block = model_cfg.get("decoder", {})
	_log_stage("constructing model")
	m_cfg = ThreeDecoderModelConfig(
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
	model = ThreeDecoderModel(m_cfg)

	loss_block = train_cfg.get("loss", {})
	_log_stage("constructing loss")
	loss_fn = MultiTaskLoss(
		LossConfig(
			lambda_trunk=float(loss_block.get("lambda_trunk", 1.0)),
			lambda_skeleton=float(loss_block.get("lambda_skeleton", 1.0)),
			lambda_occupancy=float(loss_block.get("lambda_occupancy", 0.0)),
			occupancy_pos_weight_clip_min=float(loss_block.get("occupancy_pos_weight_clip_min", 1.0)),
			occupancy_pos_weight_clip_max=float(loss_block.get("occupancy_pos_weight_clip_max", 200.0)),
		)
	)

	trainer_block = train_cfg.get("trainer", {})
	_log_stage("constructing trainer config")
	t_cfg = TrainerConfig(
		epochs=int(trainer_block.get("epochs", 30)),
		lr=float(trainer_block.get("lr", 1e-4)),
		weight_decay=float(trainer_block.get("weight_decay", 1e-4)),
		grad_clip_norm=float(trainer_block.get("grad_clip_norm", 1.0)),
		device=str(trainer_block.get("device", "cuda")),
		checkpoint_dir=str(trainer_block.get("checkpoint_dir", "checkpoints")),
		monitor_metric=str(trainer_block.get("monitor_metric", "val_loss")),
		monitor_mode=str(trainer_block.get("monitor_mode", "min")),
		best_ckpt_name=str(trainer_block.get("best_ckpt_name", "best_encoder.pt")),
		log_every_steps=int(trainer_block.get("log_every_steps", 20)),
	)

	trainer = Trainer(model=model, loss_fn=loss_fn, datamodule=datamodule, cfg=t_cfg)
	try:
		_log_stage("starting fit")
		trainer.fit()
		_log_stage("training finished")
	finally:
		if cleanup_precompute_dir and os.path.isdir(cleanup_precompute_dir):
			_log_stage(f"cleaning precomputed samples at {cleanup_precompute_dir}")
			shutil.rmtree(cleanup_precompute_dir)


if __name__ == "__main__":
	main()
