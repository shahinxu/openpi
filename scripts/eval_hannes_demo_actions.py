import argparse
import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import websockets.sync.client

from openpi_client import msgpack_numpy


class _SimpleWebsocketClient:
    def __init__(self, host: str = "0.0.0.0", port: int | None = None):
        if host.startswith("ws"):
            uri = host
        else:
            uri = f"ws://{host}"
        if port is not None:
            uri += f":{port}"

        self._packer = msgpack_numpy.Packer()
        self._ws, self._metadata = self._wait_for_server(uri)

    def _wait_for_server(self, uri: str):
        while True:
            try:
                conn = websockets.sync.client.connect(
                    uri,
                    compression=None,
                    max_size=None,
                    ping_interval=None,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                time.sleep(5)

    def get_server_metadata(self) -> dict:
        return self._metadata

    def infer(self, obs: dict) -> dict:
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass


def load_episode(hdf5_path: str, episode: int = 0):
    with h5py.File(hdf5_path, "r") as f:
        ep_key = f"episode_{episode}"
        if ep_key not in f:
            raise KeyError(f"Episode group {ep_key} not found in {hdf5_path}")
        g = f[ep_key]
        actions = np.asarray(g["actions"], dtype=np.float32)
        states = np.asarray(g["states"], dtype=np.float32) if "states" in g else None
        wrist = None
        # Support legacy two-camera layouts and the current single-agentview layout.
        if "frontview_images" in g and "agentview_images" in g:
            image = np.asarray(g["frontview_images"], dtype=np.uint8)
            wrist = np.asarray(g["agentview_images"], dtype=np.uint8)
        elif "agentview_images" in g and "sideview_images" in g:
            image = np.asarray(g["agentview_images"], dtype=np.uint8)
            wrist = np.asarray(g["sideview_images"], dtype=np.uint8)
        elif "agentview_images" in g:
            image = np.asarray(g["agentview_images"], dtype=np.uint8)
        else:
            available = list(g.keys())
            raise KeyError(
                f"Unsupported image layout in {hdf5_path} under {ep_key}: "
                f"expected frontview+agentview, agentview+sideview, or agentview-only, got keys: {available}"
            )
        task = g.attrs.get("task", None)
    return actions, states, image, wrist, task


def style_eval_plot():
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "axes.linewidth": 1.1,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "legend.frameon": False,
            "lines.linewidth": 2.2,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.22,
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hannes policy vs demo actions on a single HDF5 file.")
    parser.add_argument(
        "--hdf5",
        type=str,
        default="dataset_hannes_total/Hold the milk carton_002.hdf5"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000
    )
    parser.add_argument(
        "--prompt",
        type=str
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="plots/eval_Hold_the_milk_carton_v2.png"
    )
    args = parser.parse_args()

    print(f"Loading demo from {args.hdf5}")
    gt_actions, gt_states, image_imgs, wrist_imgs, task_attr = load_episode(args.hdf5, episode=0)

    if args.max_steps is not None:
        T = min(args.max_steps, gt_actions.shape[0], image_imgs.shape[0])
    else:
        T = min(gt_actions.shape[0], image_imgs.shape[0])

    if wrist_imgs is not None:
        T = min(T, wrist_imgs.shape[0])

    if gt_states is not None:
        T = min(T, gt_states.shape[0])
        if gt_states.ndim == 1:
            gt_states = gt_states.reshape(T, -1)
        if gt_states.shape[1] < 8:
            pad = np.zeros((gt_states.shape[0], 8 - gt_states.shape[1]), dtype=np.float32)
            gt_states = np.concatenate([gt_states, pad], axis=1)
        elif gt_states.shape[1] > 8:
            gt_states = gt_states[:, :8]
    else:
        gt_states = np.zeros((T, 8), dtype=np.float32)

    # Decide prompt
    prompt = (
        args.prompt
        or (task_attr.decode("utf-8") if isinstance(task_attr, (bytes, bytearray)) else task_attr)
        or os.path.basename(args.hdf5).split(".")[0]
    )
    print(f"Using prompt: {prompt!r}")
    print(f"Episode length (frames used): {T}")
    pred_actions = np.zeros((T, 6), dtype=np.float32)

    chunk_size = args.chunk_size if args.chunk_size and args.chunk_size > 0 else T

    first_connect = True
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        print(f"Connecting to policy server at {args.host}:{args.port} for steps [{start}, {end}) ...")
        policy = _SimpleWebsocketClient(host=args.host, port=args.port)
        if first_connect:
            print("Connected. Server metadata:", policy.get_server_metadata())
            first_connect = False

        for t in range(start, end):
            obs = {
                "observation/state": gt_states[t],
                "observation/image": image_imgs[t],
                "prompt": prompt,
            }
            if wrist_imgs is not None:
                obs["observation/wrist_image"] = wrist_imgs[t]
            out = policy.infer(obs)
            actions_seq = np.asarray(out["actions"], dtype=np.float32)
            pred = actions_seq[0, :6]
            pred_actions[t] = pred

            if t < 5:
                print(f"t={t}: gt={gt_actions[t]}, pred={pred}")

        # 显式关闭连接，避免长时间空闲被 keepalive 认为超时
        try:
            if hasattr(policy, "_ws"):
                policy._ws.close()  # type: ignore[attr-defined]
        except Exception:
            pass
    diff = pred_actions - gt_actions[:T]

    l2_per_step = np.linalg.norm(diff, axis=-1)
    mae_per_dim = np.mean(np.abs(diff), axis=0)
    print("\n=== Evaluation summary ===")
    print(f"Mean L2 error per step: {l2_per_step.mean():.4f}")
    print(f"Std  L2 error per step: {l2_per_step.std():.4f}")
    print("Mean absolute error per dim (6 dims):", mae_per_dim)

    # Plot / save per-dimension trajectories: gt vs pred
    if not args.no_show or args.save_path is not None:
        style_eval_plot()
        t_axis = np.arange(T)
        plot_dims = 3
        dim_names = ["wrist_pitch", "wrist_yaw", "grip_mean"]
        gt_color = "#1f4e79"
        pred_color = "#c84c09"

        fig, axes = plt.subplots(plot_dims, 1, figsize=(11.5, 7.8), sharex=True)
        fig.suptitle(
            "Ground Truth vs Predicted Actions\n"
            f"{os.path.basename(args.hdf5)} | mean L2={l2_per_step.mean():.4f}",
            y=0.985,
            fontweight="bold",
        )

        for i in range(plot_dims):
            ax = axes[i]
            ax.plot(t_axis, gt_actions[:T, i], label="Ground truth", color=gt_color)
            ax.plot(
                t_axis,
                pred_actions[:, i],
                label="Prediction",
                color=pred_color,
                linestyle="--",
                linewidth=2.0,
            )
            ax.set_ylabel(dim_names[i])
            ax.set_ylim(-1.0, 1.0)
            ax.set_xlim(0, max(T - 1, 1))
            ax.grid(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if i == 0:
                ax.legend(loc="upper right", ncol=2, handlelength=2.8)

        axes[-1].set_xlabel("Time step")
        fig.align_ylabels(axes)
        plt.tight_layout(rect=(0.03, 0.03, 1, 0.95))

        if args.save_path is not None:
            out_path = args.save_path
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            fig.savefig(out_path, dpi=150)
            print(f"Saved plot to {out_path}")

        if not args.no_show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
