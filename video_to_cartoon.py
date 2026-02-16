import os
import cv2
import numpy as np
from cartoon import cartoon  # existing function

def make_dir(path: str):
    os.makedirs(path, exist_ok=True)

def extract_frames(video_path: str, frames_dir: str):
    make_dir(frames_dir)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    i = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        i += 1
    cap.release()
    return frames, fps

def preprocess_frame(frame_rgb, max_size=720):
    h, w = frame_rgb.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    nh, nw = int(h * scale), int(w * scale)

    # resize
    frame_rgb = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_AREA)

    # (optional) model-friendly size: multiple of 8
    nh = (nh // 8) * 8
    nw = (nw // 8) * 8
    frame_rgb = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    return frame_rgb

def stylize_frames(frames_rgb):
    out_bgr = []
    for f in frames_rgb:
        f2 = preprocess_frame(f)
        cartoon_bgr = cartoon(f2)  # your existing effect
        out_bgr.append(cartoon_bgr)
    return out_bgr

def reconstruct_video(frames_bgr, out_path: str, fps: float):
    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for fr in frames_bgr:
        vw.write(fr)
    vw.release()

def video_to_cartoon(video_path: str, out_path: str):
    frames, fps = extract_frames(video_path, frames_dir="tmp_frames")
    frames_bgr = stylize_frames(frames)
    reconstruct_video(frames_bgr, out_path, fps)
    return out_path
