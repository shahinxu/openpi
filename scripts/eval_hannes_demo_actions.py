import argparse
import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import websockets.sync.client

from openpi_client import msgpack_numpy


class _SimpleWebsocketClient:
    """Minimal websocket client compatible with the OpenPI policy server.

    This implementation disables websocket keepalive pings (ping_interval=None)
    so that long JAX compilation on the server side does not trigger a
    ConnectionClosedError due to ping timeouts during the first few inferences.
    """

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
        front = np.asarray(g["frontview_images"], dtype=np.uint8)
        agent = np.asarray(g["agentview_images"], dtype=np.uint8)
        task = g.attrs.get("task", None)
    return actions, front, agent, task


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hannes policy vs demo actions on a single HDF5 file.")
    parser.add_argument(
        "--hdf5",
        type=str,
        default="hannes_demonstrations/Hold the milk carton_2.hdf5"
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
    gt_actions, front_imgs, agent_imgs, task_attr = load_episode(args.hdf5, episode=0)

    if args.max_steps is not None:
        T = min(args.max_steps, gt_actions.shape[0], front_imgs.shape[0], agent_imgs.shape[0])
    else:
        T = min(gt_actions.shape[0], front_imgs.shape[0], agent_imgs.shape[0])

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
                # State was not used during training; we can safely feed zeros
                "observation/state": np.zeros((8,), dtype=np.float32),
                # Training used frontview as main image and agentview as wrist image
                "observation/image": front_imgs[t],
                "observation/wrist_image": agent_imgs[t],
                "prompt": prompt,
            }
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
        t_axis = np.arange(T)
        dim_names = [f"dim{i}" for i in range(6)]

        fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(f"Actions GT vs Predicted on {os.path.basename(args.hdf5)}")

        for i in range(6):
            ax = axes[i]
            ax.plot(t_axis, gt_actions[:T, i], label="gt", linewidth=1.5)
            ax.plot(t_axis, pred_actions[:, i], label="pred", linewidth=1.0, linestyle="--")
            ax.set_ylabel(dim_names[i])
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(loc="upper right")

        axes[-1].set_xlabel("t (step)")
        plt.tight_layout(rect=(0, 0, 1, 0.96))

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
