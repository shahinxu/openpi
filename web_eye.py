import gradio as gr
import cv2
import numpy as np
import time
from collections import deque
from gradio.components.image_editor import WebcamOptions

# 全局状态
_frame_count = 0
_start_time = None
_latency_history = deque(maxlen=30)  # 保留最近30帧延迟
_frame_timestamps = deque(maxlen=10)  # 保留最近10帧到达时间，用于滑动窗口FPS


def process_stream(image):
    global _frame_count, _start_time

    if image is None:
        return "等待画面..."

    t_recv = time.time()

    # 初始化计时
    if _start_time is None:
        _start_time = t_recv
    _frame_count += 1
    _frame_timestamps.append(t_recv)

    h, w = image.shape[:2]
    brightness = float(np.mean(image))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float(np.sum(edges > 0)) / (h * w) * 100

    t_proc = time.time()
    proc_ms = (t_proc - t_recv) * 1000
    _latency_history.append(proc_ms)

    # 计算实时 FPS（滑动窗口：最近10帧）
    elapsed = t_proc - _start_time
    if len(_frame_timestamps) >= 2:
        window_fps = (len(_frame_timestamps) - 1) / (_frame_timestamps[-1] - _frame_timestamps[0])
    else:
        window_fps = 0.0
    avg_fps = _frame_count / elapsed if elapsed > 0 else 0.0

    # 近30帧平均延迟
    avg_latency = sum(_latency_history) / len(_latency_history)
    min_latency = min(_latency_history)
    max_latency = max(_latency_history)

    result = (
        f"──── 帧统计 ────\n"
        f"累计帧数:     {_frame_count}\n"
        f"运行时长:     {elapsed:.1f} s\n"
        f"当前 FPS:     {window_fps:.2f}  (近{len(_frame_timestamps)}帧滑动窗口)\n"
        f"均值 FPS:     {avg_fps:.2f}  (全程平均)\n"
        f"\n──── 延迟（近{len(_latency_history)}帧）────\n"
        f"本帧处理:     {proc_ms:.2f} ms\n"
        f"平均延迟:     {avg_latency:.2f} ms\n"
        f"最小延迟:     {min_latency:.2f} ms\n"
        f"最大延迟:     {max_latency:.2f} ms\n"
        f"\n──── 图像信息 ────\n"
        f"分辨率:       {w} x {h}\n"
        f"平均亮度:     {brightness:.1f} / 255\n"
        f"边缘像素:     {edge_ratio:.2f}%\n"
        f"时间戳:       {time.strftime('%H:%M:%S')}.{int(t_recv*1000)%1000:03d}"
    )
    return result

demo = gr.Interface(
    fn=process_stream,
    inputs=gr.Image(sources=["webcam"], streaming=True,
                    webcam_options=WebcamOptions(mirror=False, constraints={"video": {"facingMode": {"ideal": "environment"}, "width": {"ideal": 640}, "height": {"ideal": 480}}})),  # 优先后置摄像头，限制分辨率降低传输量
    outputs="text",
    live=True,
    title="VLA 实时云端大脑"
)

# share=True 走公网中继延迟高（~2 FPS）；改用局域网直连可达 20+ FPS
# 手机和服务器需在同一局域网，访问 http://<服务器IP>:7860
import subprocess, threading

def _start_cloudflare_tunnel(port=7860):
    """后台启动 cloudflared tunnel，直接输出到终端"""
    def run():
        subprocess.Popen(
            ["/home/zhx/bin/cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
        )
    threading.Thread(target=run, daemon=True).start()

_start_cloudflare_tunnel(7860)
demo.launch(server_name="0.0.0.0", server_port=7860, share=False)