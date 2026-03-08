import argparse
import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_STATE_FIELDS = ["wrist_pitch", "wrist_yaw", "grip_pos_mean"]


def parse_action_fields(group):
    raw = group.attrs.get("action_fields", None)
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            return None
    if isinstance(raw, (list, tuple, np.ndarray)):
        return [str(x) for x in raw]
    return None


def resolve_action_layout(group, action_dim):
    fields = parse_action_fields(group)
    if fields is None:
        # Fallback convention used in this workspace:
        # [wrist_pitch_cmd, wrist_yaw_cmd, finger0..]
        return 0, 1, list(range(2, action_dim))

    name_to_idx = {name: i for i, name in enumerate(fields)}

    pitch_idx = name_to_idx.get("wrist_pitch_cmd", 0)
    yaw_idx = name_to_idx.get("wrist_yaw_cmd", 1)

    finger_candidates = [
        "forefinger_cmd",
        "midfinger_cmd",
        "ringfinger_cmd",
        "littlefinger_cmd",
    ]
    finger_idx = [name_to_idx[n] for n in finger_candidates if n in name_to_idx]

    if not finger_idx:
        finger_idx = list(range(2, action_dim))

    return pitch_idx, yaw_idx, finger_idx


def reconstruct_states(actions, pitch_idx, yaw_idx, finger_idx, mode, alpha):
    n = actions.shape[0]
    states = np.zeros((n, 3), dtype=np.float32)

    if mode == "command":
        states[:, 0] = actions[:, pitch_idx]
        states[:, 1] = actions[:, yaw_idx]
        states[:, 2] = np.mean(actions[:, finger_idx], axis=1)
        return states

    # "smoothed" mode: first-order lag from command to state.
    p = float(actions[0, pitch_idx])
    y = float(actions[0, yaw_idx])
    g = float(np.mean(actions[0, finger_idx]))

    for t in range(n):
        p_cmd = float(actions[t, pitch_idx])
        y_cmd = float(actions[t, yaw_idx])
        g_cmd = float(np.mean(actions[t, finger_idx]))

        p += alpha * (p_cmd - p)
        y += alpha * (y_cmd - y)
        g += alpha * (g_cmd - g)

        states[t, 0] = p
        states[t, 1] = y
        states[t, 2] = g

    return states


def iter_episode_groups(h5file):
    groups = []

    for name, node in h5file.items():
        if isinstance(node, h5py.Group) and name.startswith("episode_"):
            groups.append(node)

    if "data" in h5file and isinstance(h5file["data"], h5py.Group):
        data_group = h5file["data"]
        for _, node in data_group.items():
            if isinstance(node, h5py.Group) and "actions" in node:
                groups.append(node)

    return groups


def process_file(file_path, overwrite, mode, alpha):
    updated = 0
    skipped = 0
    missing_actions = 0

    with h5py.File(file_path, "r+") as f:
        episode_groups = iter_episode_groups(f)

        if not episode_groups:
            return {
                "updated": 0,
                "skipped": 0,
                "missing_actions": 0,
                "episodes": 0,
            }

        for grp in episode_groups:
            if "actions" not in grp:
                missing_actions += 1
                continue

            if "states" in grp and not overwrite:
                skipped += 1
                continue

            actions = np.asarray(grp["actions"])
            if actions.ndim != 2 or actions.shape[1] < 3:
                skipped += 1
                continue

            pitch_idx, yaw_idx, finger_idx = resolve_action_layout(grp, actions.shape[1])
            states = reconstruct_states(
                actions=actions,
                pitch_idx=pitch_idx,
                yaw_idx=yaw_idx,
                finger_idx=finger_idx,
                mode=mode,
                alpha=alpha,
            )

            if "states" in grp:
                del grp["states"]
            grp.create_dataset("states", data=states, compression="gzip")

            grp.attrs["state_fields"] = json.dumps(DEFAULT_STATE_FIELDS)
            grp.attrs["state_source"] = (
                f"reconstructed_from_actions(mode={mode},alpha={alpha})"
            )
            updated += 1

    return {
        "updated": updated,
        "skipped": skipped,
        "missing_actions": missing_actions,
        "episodes": updated + skipped + missing_actions,
    }


def collect_hdf5_paths(path_arg):
    p = Path(path_arg)
    if p.is_file() and p.suffix.lower() in {".hdf5", ".h5"}:
        return [p]
    if p.is_dir():
        return sorted(
            [x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in {".hdf5", ".h5"}]
        )
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Backfill 3D states from actions into HDF5 episodes."
    )
    parser.add_argument(
        "--path",
        type=str,
        # required=True,
        default="C:/Users/xz200/Downloads/Hold the milk carton_011.hdf5",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="smoothed",
        choices=["smoothed", "command"]
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.12,
        help="Smoothing factor used in smoothed mode (0,1].",
    )
    args = parser.parse_args()

    if not (0.0 < args.alpha <= 1.0):
        raise ValueError("--alpha must be in (0, 1].")

    hdf5_paths = collect_hdf5_paths(args.path)
    if not hdf5_paths:
        raise FileNotFoundError(f"No .hdf5/.h5 files found under: {args.path}")

    total_updated = 0
    total_skipped = 0
    total_missing_actions = 0
    total_episodes = 0

    for fp in hdf5_paths:
        stats = process_file(
            file_path=str(fp),
            overwrite=args.overwrite,
            mode=args.mode,
            alpha=args.alpha,
        )
        total_updated += stats["updated"]
        total_skipped += stats["skipped"]
        total_missing_actions += stats["missing_actions"]
        total_episodes += stats["episodes"]

        print(
            f"{fp}: episodes={stats['episodes']}, updated={stats['updated']}, "
            f"skipped={stats['skipped']}, missing_actions={stats['missing_actions']}"
        )

    print("=== Summary ===")
    print(f"files: {len(hdf5_paths)}")
    print(f"episodes: {total_episodes}")
    print(f"updated: {total_updated}")
    print(f"skipped: {total_skipped}")
    print(f"missing_actions: {total_missing_actions}")


if __name__ == "__main__":
    main()
