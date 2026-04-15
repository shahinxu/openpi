from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import torch
from torch.optim import AdamW


@dataclass
class TrainerConfig:
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"
    best_ckpt_name: str = "best_encoder.pt"
    log_every_steps: int = 20


class Trainer:
    def __init__(self, model, loss_fn, datamodule, cfg: TrainerConfig) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.datamodule = datamodule
        self.cfg = cfg
        self.best_metric_value = float("inf") if cfg.monitor_mode == "min" else float("-inf")
        self.best_epoch = -1
        self.device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
        self.model.to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    def fit(self) -> None:
        self.datamodule.setup()
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

        for epoch_idx in range(self.cfg.epochs):
            train_metrics = self.train_one_epoch(epoch_idx)
            val_metrics = self.validate_one_epoch(epoch_idx)

            monitor_value = float(val_metrics.get(self.cfg.monitor_metric, val_metrics.get("val_loss", 0.0)))
            improved = self.update_best_encoder(monitor_value, epoch_idx)

            print(
                f"epoch={epoch_idx + 1}/{self.cfg.epochs} "
                f"train_loss={train_metrics.get('train_loss', 0.0):.6f} "
                f"train_trunk={train_metrics.get('train_loss_trunk', 0.0):.6f} "
                f"train_skeleton={train_metrics.get('train_loss_skeleton', 0.0):.6f} "
                f"train_occupancy={train_metrics.get('train_loss_occupancy', 0.0):.6f} "
                f"val_loss={val_metrics.get('val_loss', 0.0):.6f} "
                f"val_trunk={val_metrics.get('val_loss_trunk', 0.0):.6f} "
                f"val_skeleton={val_metrics.get('val_loss_skeleton', 0.0):.6f} "
                f"val_occupancy={val_metrics.get('val_loss_occupancy', 0.0):.6f} "
                f"best={self.best_metric_value:.6f} "
                f"saved_best={improved}"
            )

    def _progress_line(self, epoch_idx: int, batch_idx: int, total_batches: int, metrics: dict[str, float]) -> str:
        if total_batches <= 0:
            total_batches = 1
        width = 28
        ratio = min(max(batch_idx / total_batches, 0.0), 1.0)
        filled = int(ratio * width)
        bar = "#" * filled + "-" * (width - filled)
        return (
            f"epoch={epoch_idx + 1}/{self.cfg.epochs} "
            f"[{bar}] {batch_idx}/{total_batches} "
            f"loss={metrics.get('train_loss', 0.0):.4f} "
            f"trunk={metrics.get('train_loss_trunk', 0.0):.4f} "
            f"skeleton={metrics.get('train_loss_skeleton', 0.0):.4f} "
            f"occupancy={metrics.get('train_loss_occupancy', 0.0):.4f}"
        )

    def train_one_epoch(self, epoch_idx: int):
        self.model.train()
        loader = self.datamodule.train_loader()
        total_batches = len(loader)

        total_loss = 0.0
        total_trunk = 0.0
        total_skeleton = 0.0
        total_occupancy = 0.0
        count = 0

        for batch_idx, batch in enumerate(loader, start=1):
            video = batch["video"].to(self.device)
            targets = {
                "trunk6": batch["trunk6"].to(self.device),
                "skeleton153": batch["skeleton153"].to(self.device),
                "occupancy": batch["occupancy"].to(self.device),
            }

            preds = self.model(video)
            loss, stats = self.loss_fn.compute(preds, targets)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            self.optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            total_trunk += float(stats.get("loss_trunk", 0.0))
            total_skeleton += float(stats.get("loss_skeleton", 0.0))
            total_occupancy += float(stats.get("loss_occupancy", 0.0))
            count += 1

            if self.cfg.log_every_steps > 0 and (
                batch_idx == 1
                or batch_idx % self.cfg.log_every_steps == 0
                or batch_idx == total_batches
            ):
                avg_metrics = {
                    "train_loss": total_loss / max(count, 1),
                    "train_loss_trunk": total_trunk / max(count, 1),
                    "train_loss_skeleton": total_skeleton / max(count, 1),
                    "train_loss_occupancy": total_occupancy / max(count, 1),
                }
                print("\r" + self._progress_line(epoch_idx, batch_idx, total_batches, avg_metrics), end="", flush=True)

        # Make sure the next print starts on a fresh line after the progress bar.
        print(flush=True)

        return {
            "train_loss": total_loss / max(count, 1),
            "train_loss_trunk": total_trunk / max(count, 1),
            "train_loss_skeleton": total_skeleton / max(count, 1),
            "train_loss_occupancy": total_occupancy / max(count, 1),
        }

    def validate_one_epoch(self, epoch_idx: int):
        _ = epoch_idx
        self.model.eval()
        loader = self.datamodule.val_loader()

        total_loss = 0.0
        total_trunk = 0.0
        total_skeleton = 0.0
        total_occupancy = 0.0
        count = 0

        with torch.no_grad():
            for batch in loader:
                video = batch["video"].to(self.device)
                targets = {
                    "trunk6": batch["trunk6"].to(self.device),
                    "skeleton153": batch["skeleton153"].to(self.device),
                    "occupancy": batch["occupancy"].to(self.device),
                }

                preds = self.model(video)
                loss, stats = self.loss_fn.compute(preds, targets)
                total_loss += float(loss.detach().cpu().item())
                total_trunk += float(stats.get("loss_trunk", 0.0))
                total_skeleton += float(stats.get("loss_skeleton", 0.0))
                total_occupancy += float(stats.get("loss_occupancy", 0.0))
                count += 1

            return {
                "val_loss": total_loss / max(count, 1),
                "val_loss_trunk": total_trunk / max(count, 1),
                "val_loss_skeleton": total_skeleton / max(count, 1),
                "val_loss_occupancy": total_occupancy / max(count, 1),
            }

    def _get_encoder(self):
        encoder = getattr(self.model, "encoder", None)
        if encoder is None:
            raise AttributeError("Model has no encoder attribute to save")
        if not hasattr(encoder, "state_dict"):
            raise TypeError("Encoder does not expose state_dict()")
        return encoder

    def save_encoder_checkpoint(self, path: str, metadata: dict[str, Any] | None = None) -> None:
        encoder = self._get_encoder()
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        payload = {
            "encoder_state_dict": encoder.state_dict(),
            "trainer_config": {
                "epochs": self.cfg.epochs,
                "lr": self.cfg.lr,
                "weight_decay": self.cfg.weight_decay,
                "grad_clip_norm": self.cfg.grad_clip_norm,
                "device": self.cfg.device,
            },
            "metadata": metadata or {},
        }
        torch.save(payload, path)

    def save_checkpoint(self, path: str) -> None:
        # Current project policy: save encoder-only checkpoint by default.
        self.save_encoder_checkpoint(path)

    def _is_better(self, current: float) -> bool:
        if self.cfg.monitor_mode == "min":
            return current < self.best_metric_value
        if self.cfg.monitor_mode == "max":
            return current > self.best_metric_value
        raise ValueError(f"Unsupported monitor_mode: {self.cfg.monitor_mode}")

    def update_best_encoder(self, metric_value: float, epoch_idx: int) -> bool:
        if not self._is_better(metric_value):
            return False

        self.best_metric_value = metric_value
        self.best_epoch = epoch_idx
        ckpt_path = os.path.join(self.cfg.checkpoint_dir, self.cfg.best_ckpt_name)
        self.save_encoder_checkpoint(
            ckpt_path,
            metadata={
                "epoch": epoch_idx,
                "monitor_metric": self.cfg.monitor_metric,
                "monitor_mode": self.cfg.monitor_mode,
                "best_metric_value": metric_value,
            },
        )
        return True
