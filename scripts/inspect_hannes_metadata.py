import argparse
from pathlib import Path

import h5py
import numpy as np


def inspect_file(path: Path):
    info = {
        "file": str(path.name),
        "task": None,
        "num_steps": None,
        "total_reward": None,
        "T": None,
        "action_dim": None,
        "action_norm_mean": None,
        "action_norm_std": None,
        "nonzero_frac": None,
    }
    with h5py.File(path, "r") as f:
        if "episode_0" not in f:
            return info
        g = f["episode_0"]
        info["task"] = g.attrs.get("task", None)
        info["num_steps"] = int(g.attrs.get("num_steps", -1))
        info["total_reward"] = float(g.attrs.get("total_reward", 0.0))

        if "actions" in g:
            actions = np.asarray(g["actions"], dtype=np.float32)
            T, D = actions.shape
            info["T"] = T
            info["action_dim"] = D
            norms = np.linalg.norm(actions, axis=-1)
            info["action_norm_mean"] = float(norms.mean())
            info["action_norm_std"] = float(norms.std())
            info["nonzero_frac"] = float(np.count_nonzero(norms > 1e-5) / T)
        else:
            info["T"] = 0
            info["action_dim"] = 0
            info["nonzero_frac"] = 0.0

    return info


def main():
    parser = argparse.ArgumentParser(description="Inspect Hannes demonstration HDF5 metadata and basic stats.")
    parser.add_argument(
        "--root",
        type=str,
        default="hannes_demonstrations",
        help="Directory containing raw HDF5 files.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)

    files = sorted(root.glob("*.hdf5"))
    if not files:
        print("No .hdf5 files found under", root)
        return

    print(f"Found {len(files)} HDF5 files under {root}.")

    per_file = []
    tasks = {}
    all_lengths = []
    all_norms = []
    all_nonzero_fracs = []

    for fp in files:
        info = inspect_file(fp)
        per_file.append(info)

        task = info["task"]
        if isinstance(task, bytes):
            task = task.decode("utf-8")
        if task is None:
            task = "<unknown>"
        tasks[task] = tasks.get(task, 0) + 1

        if info["T"]:
            all_lengths.append(info["T"])
        if info["action_norm_mean"] is not None:
            all_norms.append(info["action_norm_mean"])
        if info["nonzero_frac"] is not None:
            all_nonzero_fracs.append(info["nonzero_frac"])

    # Print per-file brief summary (first few files)
    print("\nPer-file summary (first 10):")
    for info in per_file[:10]:
        print(
            f"- {info['file']}: task={info['task']!r}, T={info['T']}, "
            f"nonzero_frac={info['nonzero_frac']:.3f}"
        )

    # Overall task distribution
    print("\nTask distribution:")
    for task, count in sorted(tasks.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {task!r}: {count} files")

    # Global stats
    if all_lengths:
        lengths = np.asarray(all_lengths, dtype=np.int32)
        print("\nEpisode length stats (in steps):")
        print(f"  mean={lengths.mean():.1f}, std={lengths.std():.1f}, min={lengths.min()}, max={lengths.max()}")

    if all_norms:
        norms = np.asarray(all_norms, dtype=np.float32)
        print("\nPer-episode mean action-norm stats:")
        print(f"  mean={norms.mean():.4f}, std={norms.std():.4f}, min={norms.min():.4f}, max={norms.max():.4f}")

    if all_nonzero_fracs:
        nz = np.asarray(all_nonzero_fracs, dtype=np.float32)
        print("\nPer-episode nonzero action fraction:")
        print(f"  mean={nz.mean():.3f}, std={nz.std():.3f}, min={nz.min():.3f}, max={nz.max():.3f}")


if __name__ == "__main__":
    main()
