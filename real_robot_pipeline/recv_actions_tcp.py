"""Simple TCP receiver for newline-delimited JSON action packets.

Usage:
  python real_robot_pipeline/recv_actions_tcp.py --host 127.0.0.1 --port 18080
"""

import argparse
import json
import socket
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)

    print(f"Listening TCP on {args.host}:{args.port}")

    while True:
        conn, addr = srv.accept()
        print(f"Connected from {addr[0]}:{addr[1]}")
        f = conn.makefile("r", encoding="utf-8", newline="\n")
        last_t = None
        try:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                now = time.time()
                dt_ms = (now - last_t) * 1000 if last_t is not None else 0.0
                last_t = now

                try:
                    msg = json.loads(line)
                except Exception:
                    print(f"invalid line: {line[:120]}")
                    continue

                frame = msg.get("frame")
                infer_ms = msg.get("infer_ms")
                cmd = msg.get("command", {})
                print(
                    f"frame={frame} infer={infer_ms}ms recv_dt={dt_ms:.1f}ms "
                    f"cmd=({cmd.get('pitch')},{cmd.get('yaw')},{cmd.get('grip')})"
                )
        except Exception:
            pass
        finally:
            try:
                f.close()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
            print("Connection closed; waiting for reconnect...")


if __name__ == "__main__":
    main()
