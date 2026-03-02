from __future__ import annotations

import dataclasses
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List, Optional

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro

@dataclasses.dataclass
class HannesHDF5Layout:
    state_key: Optional[str] = None
    action_key: str = "episode_0/actions"

    image_keys: Dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "image": "episode_0/frontview_images",
            "wrist_image": "episode_0/agentview_images",
        }
    )
    prompt_key: Optional[str] = None


DEFAULT_LAYOUT = HannesHDF5Layout()

def create_empty_dataset(repo_id: str, *, fps: int = 10) -> LeRobotDataset:
    features: Dict[str, Any] = {
        "image": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["actions"],
        },
    }

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="hannes",
        fps=fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )


def load_episode(
    ep_path: Path,
    layout: HannesHDF5Layout = DEFAULT_LAYOUT,
) -> Dict[str, np.ndarray]:
    with h5py.File(ep_path, "r") as f:
        actions = f[layout.action_key][:]
        if layout.state_key is not None and layout.state_key in f:
            state = f[layout.state_key][:]
        else:
            state = None
        if not layout.image_keys:
            raise ValueError("layout.image_keys is empty; please configure at least one camera key")

        # Some Hannes episodes were collected with frontview+agentview cameras,
        # newer policy episodes may have agentview+sideview instead. Detect
        # which layout this file uses and map to the common keys
        #   image        -> main view
        #   wrist_image  -> secondary view
        if all(path in f for path in layout.image_keys.values()):
            image_key_map = layout.image_keys
        elif "episode_0/agentview_images" in f and "episode_0/sideview_images" in f:
            image_key_map = {
                "image": "episode_0/agentview_images",
                "wrist_image": "episode_0/sideview_images",
            }
        else:
            available = list(f.keys())
            raise ValueError(
                f"Unsupported image layout in {ep_path}: expected frontview/agentview "
                f"or agentview/sideview under 'episode_0', got top-level keys: {available}"
            )

        images: Dict[str, np.ndarray] = {}
        for name, path in image_key_map.items():
            images[name] = f[path][:]
        prompt: str
        if layout.prompt_key is not None and layout.prompt_key in f:
            dataset = f[layout.prompt_key]
            value = dataset[()]
            if isinstance(value, bytes):
                prompt = value.decode("utf-8")
            elif isinstance(value, np.ndarray) and value.dtype.type is np.bytes_:
                prompt = value[0].decode("utf-8")
            else:
                prompt = str(value)
        else:
            # Derive prompt from filename, e.g. "Hold the bottle_2.hdf5" -> "Hold the bottle".
            stem = ep_path.stem
            # Drop a trailing "_number" suffix if present.
            m = re.match(r"^(.*)_\\d+$", stem)
            if m:
                stem = m.group(1)
            prompt = stem.replace("_", " ")

    return {
        "state": None if state is None else np.asarray(state),
        "actions": np.asarray(actions),
        "images": images,
        "prompt": prompt,
    }


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: List[Path],
    layout: HannesHDF5Layout = DEFAULT_LAYOUT,
) -> LeRobotDataset:
    """Append all HDF5 episodes to the LeRobot dataset."""

    from tqdm import tqdm  # local import to avoid hard dependency at import time
    import cv2

    for ep_path in tqdm(hdf5_files, desc="Converting Hannes episodes"):
        episode = load_episode(ep_path, layout=layout)
        states = episode["state"]
        actions = episode["actions"]
        images = episode["images"]  # dict with keys like "image", "wrist_image"
        prompt = episode["prompt"]
        # Determine episode length from actions.
        T = len(actions)

        # Prepare state: if missing, use zeros.
        if states is None:
            states = np.zeros((T, 8), dtype=np.float32)
        else:
            states = states.astype(np.float32)

        # Sanity checks on image lengths.
        for name, arr in images.items():
            if len(arr) != T:
                raise ValueError(
                    f"Mismatched lengths for {name} in {ep_path}: "
                    f"len(actions)={T}, len({name})={len(arr)}"
                )

        actions = actions.astype(np.float32)

        for i in range(T):
            # Resize images to 256x256 to match feature spec.
            img = images["image"][i]
            wrist = images["wrist_image"][i]
            img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            wrist_resized = cv2.resize(wrist, (256, 256), interpolation=cv2.INTER_AREA)

            frame = {
                "image": img_resized,
                "wrist_image": wrist_resized,
                "state": states[i],
                "actions": actions[i],
                # LeRobot convention: store language instruction under "task".
                "task": prompt,
            }
            dataset.add_frame(frame)

        # Each HDF5 file corresponds to a single episode
        dataset.save_episode()

    return dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Args:
    # Directory containing raw HDF5 files (e.g. `hannes_demonstrations/`).
    raw_dir: Path
    # Output LeRobot repo id, e.g. "your_username/hannes".
    repo_id: str
    # Frames per second for the dataset metadata.
    fps: int = 10
    # Whether to push the resulting dataset to the Hugging Face Hub.
    push_to_hub: bool = False


def main(args: Args) -> None:
    raw_dir = args.raw_dir
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir does not exist: {raw_dir}")

    hdf5_files = sorted(raw_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found in {raw_dir}")

    dataset = create_empty_dataset(args.repo_id, fps=args.fps)
    dataset = populate_dataset(dataset, hdf5_files, layout=DEFAULT_LAYOUT)

    if args.push_to_hub:
        dataset.push_to_hub(private=False)


if __name__ == "__main__":
    tyro.cli(main)
