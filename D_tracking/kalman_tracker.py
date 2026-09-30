"""MODULE D - Motion tracking with a Kalman filter written from scratch.

Run:  python D_tracking/kalman_tracker.py --video data/video.mp4 [--noise 3] [--drop_len 5]
Outputs: console table (predict -> measure -> update), output/kalman_plot.png,
         output/kalman_overlay.mp4 (raw detections vs Kalman trajectory incl. occlusion segment).

State x = [x, y, vx, vy]^T (px, px/frame), constant-velocity model:
    x_k = F x_{k-1} + w,  F = [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]
    z_k = H x_k + v,      H = [[1,0,0,0],[0,1,0,0]]   (we only OBSERVE position)
Observability: the pair (F,H) has rank([H; HF; HF^2; HF^3]) = 4, so velocity - never measured -
is still recoverable from a sequence of positions, and prediction alone keeps tracking while
measurements are missing.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from segmentation import detect_boxes


class KalmanFilter:
    def __init__(self, x0, y0, dt=1.0, sigma_a=0.3, sigma_z=2.0):
        self.x = np.array([x0, y0, 0.0, 0.0])
        self.P = np.diag([sigma_z ** 2, sigma_z ** 2, 25.0, 25.0])
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        g = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])   # acceleration -> state
        self.Q = sigma_a ** 2 * g @ g.T
        self.R = sigma_z ** 2 * np.eye(2)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, z):
        y = np.asarray(z, float) - self.H @ self.x                 # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)                   # Kalman gain
        self.x = self.x + K @ y
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T         # Joseph form (numerically stable)
        return self.x[:2].copy()


class Track:
    def __init__(self, det, sigma_a, sigma_z):
        self.kf = KalmanFilter(det["centroid"][0], det["centroid"][1], sigma_a=sigma_a, sigma_z=sigma_z)
        self.id = None            # assigned once the track is confirmed
        self.hits, self.missed = 1, 0
        self.last_det = det
        self.history = []         # dicts: frame, pred, meas, est, dropped


class MultiTracker:
    """Nearest-neighbour data association (Hungarian) + one Kalman filter per box."""
    def __init__(self, gate=80.0, max_missed=12, min_hits=3, sigma_a=0.3, sigma_z=2.0):
        self.tracks, self.finished, self.next_id = [], [], 1
        self.gate, self.max_missed, self.min_hits = gate, max_missed, min_hits
        self.sa, self.sz = sigma_a, sigma_z

    def step(self, frame_idx, dets, drop=False):
        preds = [t.kf.predict() for t in self.tracks]                    # 1) PREDICT
        matches, used_d = {}, set()
        if dets and self.tracks and not drop:                            # 2) MEASURE / associate
            C = np.array([[np.linalg.norm(p - np.array(d["centroid"])) for d in dets] for p in preds])
            for r, c in zip(*linear_sum_assignment(C)):
                if C[r, c] < self.gate:
                    matches[r] = c; used_d.add(c)
        for i, t in enumerate(self.tracks):
            meas = None
            if i in matches:                                             # 3) UPDATE
                d = dets[matches[i]]
                meas = np.array(d["centroid"])
                est = t.kf.update(meas)
                t.hits += 1; t.missed = 0; t.last_det = d
                if t.id is None and t.hits >= self.min_hits:
                    t.id = self.next_id; self.next_id += 1
            else:
                est = t.kf.x[:2].copy()                                  # coast on prediction only
                t.missed += 1; t.last_det = None
            t.history.append(dict(frame=frame_idx, pred=preds[i], meas=meas, est=est, dropped=drop))
        if not drop:
            for j, d in enumerate(dets):
                if j not in used_d:
                    self.tracks.append(Track(d, self.sa, self.sz))
                    self.tracks[-1].history.append(dict(frame=frame_idx, pred=np.array(d["centroid"]),
                                                        meas=np.array(d["centroid"]), est=np.array(d["centroid"]), dropped=False))
        alive = []
        for t in self.tracks:
            (self.finished if t.missed > self.max_missed else alive).append(t)
        self.tracks = alive
        return [t for t in self.tracks if t.id is not None]

    def all_tracks(self):
        return [t for t in self.finished + self.tracks if t.id is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(common.DATA, "video.mp4"))
    ap.add_argument("--noise", type=float, default=3.0, help="extra Gaussian noise (px) added to measurements; 0 = raw segmentation")
    ap.add_argument("--drop_len", type=int, default=5)
    ap.add_argument("--drop_start", type=int, default=None)
    ap.add_argument("--track", type=int, default=None, help="track id to plot (default: longest)")
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    info0 = common.video_info(a.video)
    print(f"NOTE: {a.noise} px of extra Gaussian noise is added to every measurement (use --noise 0 for raw segmentation output).")
    bg = common.build_background(a.video)
    all_dets = []
    for i, fr in common.iter_video(a.video):
        ds = [d for d in detect_boxes(fr, bg)]
        for d in ds:
            d["centroid"] = (d["centroid"][0] + rng.normal(0, a.noise), d["centroid"][1] + rng.normal(0, a.noise))
        all_dets.append(ds)

    def run(drop_frames):
        mt = MultiTracker(gate=0.125 * info0['w'], sigma_z=max(a.noise, 1.0))
        for i, ds in enumerate(all_dets):
            mt.step(i, ds, drop=i in drop_frames)
        return mt

    tr0 = run(set()).all_tracks()
    if not tr0:
        sys.exit('No box was tracked - check that the video has moving boxes and an empty-belt background.')
    chosen0 = max(tr0, key=lambda t: len(t.history)) if a.track is None else [t for t in tr0 if t.id == a.track][0]
    f0, f1 = chosen0.history[0]["frame"], chosen0.history[-1]["frame"]
    ds = a.drop_start if a.drop_start is not None else (f0 + f1) // 2
    drop = set(range(ds, ds + a.drop_len))
    mt = run(drop)
    # same physical box = the track that overlaps the chosen one in the frame just before the dropout
    cand = [t for t in mt.all_tracks() if t.history[0]["frame"] <= ds - 1 <= t.history[-1]["frame"]]
    if not cand:
        sys.exit('Could not follow the chosen box through the dropout; try --drop_start/--drop_len.')
    ref_pos = [h for h in chosen0.history if h["frame"] == ds - 1][0]["est"]
    tr = min(cand, key=lambda t: np.linalg.norm([h for h in t.history if h["frame"] == ds - 1][0]["est"] - ref_pos))
    H = tr.history

    print(f"Tracking box (track id {tr.id}), occlusion simulated on frames {ds}..{ds + a.drop_len - 1}")
    print(f"{'frame':>5s} | {'PREDICTED (x,y)':>18s} | {'MEASURED (x,y)':>18s} | {'CORRECTED (x,y)':>18s} | note")
    for h in H:
        if ds - 4 <= h["frame"] <= ds + a.drop_len + 3:
            m = "      -- none --   " if h["meas"] is None else f"({h['meas'][0]:7.1f},{h['meas'][1]:7.1f})"
            note = "OCCLUDED: prediction only" if h["dropped"] else ""
            print(f"{h['frame']:5d} | ({h['pred'][0]:7.1f},{h['pred'][1]:7.1f}) | {m:>18s} | ({h['est'][0]:7.1f},{h['est'][1]:7.1f}) | {note}")

    # ---- plot ----
    fr_ = np.array([h["frame"] for h in H])
    est = np.array([h["est"] for h in H])
    raw = np.array([h["meas"] if h["meas"] is not None else [np.nan, np.nan] for h in H])
    dm = np.array([h["dropped"] for h in H])
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    ax[0].plot(raw[:, 0], raw[:, 1], "r.", ms=4, label="raw noisy measurements")
    ax[0].plot(est[:, 0], est[:, 1], "g-", lw=1.5, label="Kalman estimate")
    ax[0].plot(est[dm, 0], est[dm, 1], "o", c="orange", ms=7, label="pure prediction (occlusion)")
    ax[0].invert_yaxis(); ax[0].set_xlabel("x (px)"); ax[0].set_ylabel("y (px)"); ax[0].legend(); ax[0].set_title("Trajectory in image")
    ax[1].plot(fr_, raw[:, 1], "r.", ms=4, label="raw y"); ax[1].plot(fr_, est[:, 1], "g-", label="Kalman y")
    ax[1].axvspan(ds, ds + a.drop_len - 1, color="orange", alpha=0.3, label="dropout")
    ax[1].set_xlabel("frame"); ax[1].set_ylabel("y (px)"); ax[1].legend(); ax[1].set_title("Cross-belt position (noise vs smoothed)")
    plt.tight_layout(); p = os.path.join(common.OUT, "kalman_plot.png"); plt.savefig(p, dpi=100); print("saved", p)

    # ---- overlay video (all confirmed tracks) ----
    info = common.video_info(a.video)
    vp = os.path.join(common.OUT, "kalman_overlay.mp4")
    vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (info["w"], info["h"]))
    by_frame = {}
    for t in mt.all_tracks():
        for h in t.history:
            by_frame.setdefault(h["frame"], []).append((t.id, h))
    trails = {}
    for i, frm in common.iter_video(a.video):
        for tid, h in by_frame.get(i, []):
            trails.setdefault(tid, []).append(h["est"])
            if h["meas"] is not None:
                cv2.circle(frm, tuple(np.int32(h["meas"])), 3, (0, 0, 255), -1)          # raw = red
            col = (0, 165, 255) if h["dropped"] else (0, 255, 0)                          # est = green / orange
            cv2.circle(frm, tuple(np.int32(h["est"])), 5, col, 2)
            cv2.putText(frm, f"#{tid}", tuple(np.int32(h["est"]) + [8, -8]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            pts = np.int32(trails[tid][-40:]).reshape(-1, 1, 2)
            cv2.polylines(frm, [pts], False, (0, 255, 0), 1)
        if i in drop:
            cv2.putText(frm, "OCCLUSION: measurements dropped, Kalman predicts", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        vw.write(frm)
    vw.release(); print("saved", vp)


if __name__ == "__main__":
    main()
