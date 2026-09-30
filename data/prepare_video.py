"""Prepare a conveyor video for the project.

This mirrors the workflow used in the reference project: build a single clean
video file from source recordings, then extract representative frames for
inspection and module A testing.

Usage examples:
    python data/prepare_video.py --source-dir data/raw --out data/video.mp4
    python data/prepare_video.py --source data/video.mp4 --out data/video.mp4
    python data/prepare_video.py --source-dir data/raw --fps 30 --extract-frames 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "video.mp4"
DEFAULT_SOURCE_DIR = ROOT / "data" / "raw"
DEFAULT_REP_DIR = ROOT / "output" / "representative_frames"


def find_inputs(source: str | None, source_dir: str | None):
    if source:
        p = Path(source)
        if p.is_file():
            return [p]
        if p.is_dir():
            return sorted(p.glob("*"))
        raise FileNotFoundError(f"Source not found: {p}")

    candidates = []
    if source_dir:
        d = Path(source_dir)
        if d.exists():
            candidates.extend(sorted(d.glob("*.mp4")))
            candidates.extend(sorted(d.glob("*.avi")))
            candidates.extend(sorted(d.glob("*.mov")))
            candidates.extend(sorted(d.glob("*.mkv")))
            candidates.extend(sorted(d.glob("*.jpg")))
            candidates.extend(sorted(d.glob("*.png")))
    if not candidates:
        default = ROOT / "data" / "video.mp4"
        if default.exists():
            return [default]
        raise FileNotFoundError(
            "No input files found. Put clips in data/raw or pass --source path."
        )
    return candidates


def _read_frames_from_images(paths):
    frames = []
    for p in sorted(paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
    return frames


def prepare_video(source_paths, out_path: Path, fps: float | None = None):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Single file case: just copy/normalize to target.
    if len(source_paths) == 1 and source_paths[0].suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        src = source_paths[0]
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source video: {src}")

        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Unreadable video size in {src}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create output video: {out_path}")

        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            count += 1
        writer.release()
        cap.release()
        print(f"Copied {count} frames from {src.name} to {out_path.name}")
        return out_path

    # Multiple files or image sequence => concatenate in order.
    frames = []
    for p in source_paths:
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
        elif p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                print(f"Skipping unreadable source: {p}", file=sys.stderr)
                continue
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            cap.release()
        else:
            print(f"Skipping unsupported file: {p}", file=sys.stderr)

    if not frames:
        raise RuntimeError("No readable frames were found in the source set.")

    h, w = frames[0].shape[:2]
    if fps is None:
        fps = 30.0

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {out_path}")

    for frame in frames:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)
    writer.release()
    print(f"Wrote {len(frames)} frames to {out_path.name}")
    return out_path


def extract_representative_frames(video_path: Path, out_dir: Path, count: int = 5):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for frame extraction: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 1:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"No frames available in {video_path}")
        p = out_dir / "frame_00.png"
        cv2.imwrite(str(p), frame)
        cap.release()
        return [p]

    idxs = np.linspace(0, max(total - 1, 0), count, dtype=int)
    saved = []
    for i, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        p = out_dir / f"frame_{i:02d}.png"
        cv2.imwrite(str(p), frame)
        saved.append(p)
    cap.release()
    print(f"Saved {len(saved)} representative frames to {out_dir}")
    return saved


def main():
    ap = argparse.ArgumentParser(description="Prepare a conveyor video from raw clips or a single input video.")
    ap.add_argument("--source", type=str, default=None, help="single source video or image folder")
    ap.add_argument("--source-dir", type=str, default=str(DEFAULT_SOURCE_DIR), help="folder containing raw clips or images")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="output video path")
    ap.add_argument("--fps", type=float, default=None, help="override output fps")
    ap.add_argument("--extract-frames", type=int, default=5, help="number of representative frames to extract")
    args = ap.parse_args()

    srcs = find_inputs(args.source, args.source_dir)
    out_path = Path(args.out)
    out_path = out_path if out_path.is_absolute() else ROOT / out_path

    prepared = prepare_video(srcs, out_path, fps=args.fps)
    extract_representative_frames(prepared, DEFAULT_REP_DIR, count=args.extract_frames)
    print(f"Prepared video: {prepared}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
