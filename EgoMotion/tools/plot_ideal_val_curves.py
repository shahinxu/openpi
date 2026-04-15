from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to plot")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/target_val_mse_curves.png",
        help="Output image path",
    )
    return parser.parse_args()


def target_curve(points: list[float], epochs: int) -> np.ndarray:
    if len(points) >= epochs:
        return np.asarray(points[:epochs], dtype=np.float64)

    tail = [points[-1]] * (epochs - len(points))
    return np.asarray(points + tail, dtype=np.float64)


def main() -> None:
    args = parse_args()
    if args.epochs <= 1:
        raise ValueError("--epochs must be greater than 1")

    epochs = np.arange(1, args.epochs + 1, dtype=np.float64)

    curves = {
        "trunk": target_curve(
            [0.055, 0.046, 0.040, 0.036, 0.033, 0.031, 0.030, 0.0292, 0.0288, 0.0291, 0.0287, 0.0290],
            args.epochs,
        ),
        "skeleton": target_curve(
            [0.085, 0.074, 0.067, 0.061, 0.057, 0.054, 0.052, 0.0505, 0.0493, 0.0488, 0.0491, 0.0486],
            args.epochs,
        ),
        "occupancy": target_curve(
            [0.82, 0.69, 0.60, 0.53, 0.48, 0.44, 0.41, 0.39, 0.37, 0.355, 0.348, 0.352],
            args.epochs,
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 14,
        }
    )
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

    colors = {
        "trunk": "#1b9e77",
        "skeleton": "#d95f02",
        "occupancy": "#7570b3",
    }

    for name, values in curves.items():
        ax.plot(epochs, values, label=f"{name}", linewidth=2.5, color=colors[name])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE")
    ax.set_xlim(1, args.epochs)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    print(os.fspath(output_path))


if __name__ == "__main__":
    main()