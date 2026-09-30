"""Shared helpers for the whole project (video I/O, background model, box cropping).

Importing this module also puts every module folder on sys.path, so scripts can
import each other (e.g. pipeline.py imports A_segmentation/segmentation.py).
"""
import os
import sys
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
for _d in ["A_segmentation", "B_calibration", "C_optical_flow", "D_tracking", "E_recognition"]:
    _p = os.path.join(ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

CROP_SIZE = (96, 64)  # (width, height) of every normalised box crop


def video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    info = dict(
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def iter_video(path, max_frames=None):
    """Yield (frame_index, BGR frame)."""
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames is not None and i >= max_frames):
            break
        yield i, frame
        i += 1
    cap.release()


def get_frames(path, indices):
    cap = cv2.VideoCapture(path)
    out = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            out.append(fr)
    cap.release()
    return out


def build_background(path, n=40):
    """Median of n evenly spaced frames. Works because boxes MOVE: at every pixel
    the belt is visible in most of the sampled frames, so the median = empty belt."""
    info = video_info(path)
    idx = np.linspace(0, max(info["n"] - 1, 0), n).astype(int)
    frames = get_frames(path, idx)
    return np.median(np.stack(frames, 0), axis=0).astype(np.uint8)


def crop_box(frame, rect, size=CROP_SIZE):
    """Deskew a cv2.minAreaRect into a fixed-size crop with the LONG side horizontal.
    (A 180-degree ambiguity remains; recognize.py handles it by augmenting.)"""
    pts = cv2.boxPoints(rect).astype(np.float32)
    e0 = np.linalg.norm(pts[1] - pts[0])
    e1 = np.linalg.norm(pts[2] - pts[1])
    if e0 < e1:  # make edge 0->1 the long one
        pts = np.roll(pts, -1, axis=0)
    w, h = size
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    # keep orientation (avoid mirrored crops)
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[1]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    if cross < 0:
        pts = pts[::-1].copy()
        pts = np.roll(pts, 1, axis=0)
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(frame, M, (w, h), flags=cv2.INTER_LINEAR)
