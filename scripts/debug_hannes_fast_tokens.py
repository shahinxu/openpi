import argparse
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


def load_single_step(hdf5_path: str, t: int = 0):
    """Load a single timestep from a Hannes HDF5 demo.

    Returns (state, image, wrist_image, prompt).
    State was not used during training, so we follow eval_hannes_demo_actions.py
    and feed zeros here for consistency.
    """

    with h5py.File(hdf5_path, "r") as f:
        ep_key = "episode_0"
        if ep_key not in f:
            raise KeyError(f"Episode group {ep_key} not found in {hdf5_path}")
        g = f[ep_key]
        front = np.asarray(g["frontview_images"], dtype=np.uint8)
        agent = np.asarray(g["agentview_images"], dtype=np.uint8)
        task = g.attrs.get("task", None)

    if t < 0 or t >= len(front):
        raise IndexError(f"t={t} out of range for episode length {len(front)}")

    # Prompt: prefer task attr, fall back to filename stem.
    if isinstance(task, (bytes, bytearray)):
        prompt = task.decode("utf-8")
    else:
        prompt = task
    if not prompt:
        prompt = Path(hdf5_path).stem

    state = np.zeros((8,), dtype=np.float32)
    return state, front[t], agent[t], prompt


def build_input_transform(train_config: _config.TrainConfig, norm_stats) -> transforms.DataTransformFn:
    """Recreate the same input pipeline used by Policy for PI0-FAST Libero data,
    but without any output transforms (so we can see raw FAST tokens).
    """

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    return transforms.compose(
        [
            *data_config.data_transforms.inputs,  # LiberoInputs
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,  # InjectDefaultPrompt, ResizeImages, TokenizeFASTInputs
        ]
    )


def compute_ground_truth_tokens(
    train_config: _config.TrainConfig,
    hdf5_path: str,
    t: int,
):
    """Compute the ground-truth tokenized_prompt and loss mask used for teacher forcing.

    This approximates the same Libero + Normalize + FAST tokenization stack used during
    training, including the continuous action sequence so that the postfix ("Action: ...")
    tokens are present and supervised by the loss.
    """

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.norm_stats is None:
        raise ValueError("Normalization stats are required to compute training-time tokens.")

    # Build a transform pipeline similar to training:
    # LiberoInputs -> Normalize -> TokenizeFASTInputs.
    input_transform = transforms.compose(
        [
            *data_config.data_transforms.inputs,
            transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )

    # Load a short action chunk from the raw HDF5 episode.
    # We use the first `action_horizon` steps to form the action sequence that FAST encodes.
    with h5py.File(hdf5_path, "r") as f:
        ep_key = "episode_0"
        if ep_key not in f:
            raise KeyError(f"Episode group {ep_key} not found in {hdf5_path}")
        g = f[ep_key]
        actions = np.asarray(g["actions"], dtype=np.float32)
        front = np.asarray(g["frontview_images"], dtype=np.uint8)
        agent = np.asarray(g["agentview_images"], dtype=np.uint8)
        task = g.attrs.get("task", None)

    horizon = train_config.model.action_horizon
    if actions.shape[0] < horizon:
        action_chunk = actions
    else:
        action_chunk = actions[:horizon]

    # State was unused in training; we follow the eval script and feed zeros.
    state = np.zeros((8,), dtype=np.float32)

    if isinstance(task, (bytes, bytearray)):
        prompt = task.decode("utf-8")
    else:
        prompt = task
    if not prompt:
        prompt = Path(hdf5_path).stem

    raw = {
        "observation/state": state,
        "observation/image": front[0],
        "observation/wrist_image": agent[0],
        "actions": action_chunk,
        "prompt": prompt,
    }

    processed = input_transform(raw)

    tokens = np.asarray(processed["tokenized_prompt"], dtype=np.int32)
    loss_mask = np.asarray(processed["token_loss_mask"], dtype=bool)

    # Action tokens are exactly the positions where loss_mask == 1.
    action_tokens = tokens[loss_mask]

    return tokens, loss_mask, action_tokens


def sample_fast_tokens(
    ckpt_dir: Path,
    train_config: _config.TrainConfig,
    hdf5_path: str,
    t: int,
    max_decoding_steps: int,
):
    """Load a PI0-FAST checkpoint and sample discrete FAST tokens for one observation."""

    # Load norm stats from this checkpoint's assets to exactly match training.
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("DataConfig.asset_id is required to load norm stats.")
    norm_stats = _checkpoints.load_norm_stats(ckpt_dir / "assets", data_config.asset_id)

    input_transform = build_input_transform(train_config, norm_stats)

    # Load one timestep from the demo.
    state, img, wrist_img, prompt = load_single_step(hdf5_path, t=t)

    raw = {
        "observation/state": state,
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "prompt": prompt,
    }

    # Apply same transforms as policy inference up to tokenizer.
    inputs = input_transform(raw)

    # Add batch dimension and convert to JAX arrays.
    batched_inputs = jax.tree.map(lambda x: jnp.asarray(x)[jnp.newaxis, ...], inputs)
    observation = _model.Observation.from_dict(batched_inputs)

    # Load model params and sample tokens.
    params = _model.restore_params(ckpt_dir / "params", dtype=jnp.bfloat16)
    model = train_config.model.load(params)

    rng = jax.random.key(0)
    tokens = model.sample_actions(rng, observation, max_decoding_steps=max_decoding_steps, temperature=0.0)
    tokens_np = np.asarray(tokens)[0]  # (max_decoding_steps,)
    return tokens_np


def main():
    parser = argparse.ArgumentParser(description="Debug discrete FAST action tokens across Hannes checkpoints.")
    parser.add_argument(
        "--hdf5",
        type=str,
        default="hannes_demonstrations/Hold the milk carton_2.hdf5",
        help="Path to a Hannes demo HDF5 file.",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=0,
        help="Time index within the episode to inspect.",
    )
    parser.add_argument(
        "--max-decoding-steps",
        type=int,
        default=64,
        help="Number of FAST tokens to decode (prefix+actions).",
    )
    args = parser.parse_args()

    base_ckpt_root = Path("checkpoints/pi0_fast_hannes_low_mem_finetune")
    ckpts = {
        "v2_4000": base_ckpt_root / "hannes_pi0fast_lora_v2" / "4000",
        "v3_300": base_ckpt_root / "hannes_pi0fast_lora_v3" / "300",
        "v4_50": base_ckpt_root / "hannes_pi0fast_lora_v4" / "50",
    }

    train_config = _config.get_config("pi0_fast_hannes_low_mem_finetune")

    # First, show the ground-truth tokenized_prompt and which tokens are supervised by the loss.
    gt_tokens, gt_loss_mask, gt_action_tokens = compute_ground_truth_tokens(
        train_config=train_config,
        hdf5_path=args.hdf5,
        t=args.t,
    )
    print("\n=== Ground-truth tokenized_prompt (teacher-forcing targets) ===")
    print("Full tokenized_prompt shape:", gt_tokens.shape)
    print("token_loss_mask sum (num supervised tokens):", int(gt_loss_mask.sum()))
    print("First 32 token ids:", gt_tokens[:32].tolist())
    print("First 32 loss_mask entries:", gt_loss_mask[:32].astype(int).tolist())
    print("First 32 supervised (action) token ids:", gt_action_tokens[:32].tolist())

    tokens = {}
    for name, ckpt_dir in ckpts.items():
        print(f"\n=== Sampling tokens from {name} ({ckpt_dir}) ===")
        if not ckpt_dir.exists():
            print("  (missing checkpoint dir, skipping)")
            continue
        tokens[name] = sample_fast_tokens(
            ckpt_dir=ckpt_dir,
            train_config=train_config,
            hdf5_path=args.hdf5,
            t=args.t,
            max_decoding_steps=args.max_decoding_steps,
        )
        print("  First 32 tokens:", tokens[name][:32].tolist())

    # Compare sequences pairwise if we have them.
    names = list(tokens.keys())
    if len(names) >= 2:
        ref = names[0]
        for other in names[1:]:
            same = np.array_equal(tokens[ref], tokens[other])
            print(f"\n[Compare] {ref} vs {other}: {'IDENTICAL' if same else 'DIFFERENT'}")
            if not same:
                diff_idx = np.nonzero(tokens[ref] != tokens[other])[0]
                print("  First differing indices:", diff_idx[:10].tolist())


if __name__ == "__main__":
    main()
