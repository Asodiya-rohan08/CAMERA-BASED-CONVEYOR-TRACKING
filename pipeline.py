"""FULL PIPELINE: A (segment) -> D (Kalman track) -> B (pixels -> cm) -> C (belt speed) -> E (type).

Run:  python pipeline.py --video data/video.mp4
Prints per tracked box: ID, real-world size (cm), belt speed (cm/s) and recognised type.
Also writes output/pipeline_out.mp4 (annotated) and output/pipeline_results.csv.
"""
import os, sys, argparse, csv
from collections import Counter
import numpy as np
import cv2
import common
from segmentation import detect_boxes                    # Module A
from kalman_tracker import MultiTracker                  # Module D
from camera_model import Camera                          # Module B
from optical_flow import estimate_belt_speed             # Module C
from recognize import load_dataset, train_all            # Module E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(common.DATA, "video.mp4"))
    ap.add_argument("--calib", default=os.path.join(common.ROOT, "B_calibration", "calib.npz"))
    ap.add_argument("--boxes", default=os.path.join(common.DATA, "boxes"))
    ap.add_argument("--box_h", type=float, default=0.0, help="height of box top faces above belt (cm)")
    ap.add_argument("--cm_per_px", type=float, default=None, help="only if no calibration file")
    ap.add_argument("--rec_method", default="Eigenboxes (PCA + 1-NN)",
                    choices=["Alignment (ORB+homography)", "Eigenboxes (PCA + 1-NN)", "Hu moments (nearest mean)"])
    ap.add_argument("--flow_every", type=int, default=8)
    ap.add_argument("--classify_every", type=int, default=5)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--no_video", action="store_true")
    a = ap.parse_args()

    info = common.video_info(a.video)
    fps = info["fps"]
    cam = Camera.load(a.calib) if os.path.exists(a.calib) else None
    if cam is not None and os.path.exists(a.calib):
        cw, ch = [int(v) for v in np.load(a.calib)['image_size']]
        if (cw, ch) != (info['w'], info['h']):
            print(f'WARNING: calibration images are {cw}x{ch} but the video is {info["w"]}x{info["h"]}. '
                  'Calibrate at the SAME resolution as the video, otherwise cm values are wrong.')
    if cam is None and a.cm_per_px is None:
        sys.exit("No calibration file: run B_calibration/calibrate.py or pass --cm_per_px")
    clf = None
    if os.path.isdir(a.boxes):
        X, y = load_dataset(a.boxes)
        clf = train_all(X, y)[a.rec_method]
        print(f"Module E trained on {len(X)} crops, classes: {sorted(set(map(str, y)))}")
    print("Module A: building background model ...")
    bg = common.build_background(a.video)

    tracker = MultiTracker(gate=0.125 * info['w'])   # association gate scales with resolution
    stats = {}
    prev_frame, prev_masks = None, {}
    vw = None if a.no_video else cv2.VideoWriter(os.path.join(common.OUT, "pipeline_out.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (info["w"], info["h"]))
    for i, frame in common.iter_video(a.video, a.max_frames):
        raw = frame.copy()                       # keep an un-annotated copy for optical flow
        dets = detect_boxes(frame, bg)                                            # A
        active = tracker.step(i, dets)                                    # D
        for t in active:
            s = stats.setdefault(t.id, dict(first=i, last=i, sizes=[], flow=[], kal=[], votes=Counter()))
            s["last"] = i
            d = t.last_det
            if d is None:
                continue
            if not d["touches_border"]:
                if cam is not None:                                       # B: size in cm
                    s["sizes"].append(cam.box_size_cm(d["corners"], a.box_h))
                else:
                    L, S = sorted(d["rect"][1], reverse=True)
                    s["sizes"].append((L * a.cm_per_px, S * a.cm_per_px))
                if clf is not None and i % a.classify_every == 0:         # E
                    s["votes"][str(clf.predict(common.crop_box(frame, d["rect"])))] += 1
            v = t.kf.x[2:4]                                               # D: velocity -> cm/s
            c = t.kf.x[:2]
            if t.hits > 8 and not d["touches_border"]:
                dist = cam.displacement_cm(c, c + v, a.box_h) if cam is not None else np.linalg.norm(v) * a.cm_per_px
                s["kal"].append(dist * fps)
            if prev_frame is not None and i % a.flow_every == 0 and t.id in prev_masks and t.hits > 3:   # C
                r = estimate_belt_speed(prev_frame, frame, prev_masks[t.id], fps, 1, cam, a.box_h, a.cm_per_px)
                if r is not None:
                    s["flow"].append(r["speed_cm_s"])
        prev_masks = {t.id: t.last_det["mask"] for t in active
                      if t.last_det is not None and not t.last_det["touches_border"]}
        prev_frame = raw
        if vw is not None:
            for t in active:
                if t.last_det is None:
                    continue
                s = stats[t.id]
                if not s["sizes"]:
                    continue                                   # not fully visible yet: nothing reliable to print
                typ = s["votes"].most_common(1)[0][0] if s["votes"] else "?"
                sz = np.median(np.array(s["sizes"]), axis=0) if s["sizes"] else (0, 0)
                cv2.polylines(frame, [np.int32(t.last_det["corners"])], True, (0, 255, 0), 2)
                x, y0 = np.int32(t.last_det["bbox"][:2])
                cv2.putText(frame, f"#{t.id} {typ} {sz[0]:.1f}x{sz[1]:.1f}cm", (x, max(y0 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            vw.write(frame)
        if i % 60 == 0:
            print(f"  frame {i}/{info['n']}  active tracks: {[t.id for t in active]}")
    if vw is not None:
        vw.release()

    rows = []
    print("\n" + "=" * 86)
    print(f"{'ID':>3s} {'frames':>11s} {'size LxS (cm)':>16s} {'speed flow (cm/s)':>19s} {'speed Kalman':>13s} {'type':>10s}")
    print("=" * 86)
    for tid, s in sorted(stats.items()):
        if len(s["sizes"]) < 5:
            continue                                    # never fully visible -> not a reliable box
        sz = np.median(np.array(s["sizes"]), axis=0)
        fl = float(np.median(s["flow"])) if s["flow"] else float("nan")
        ka = float(np.median(s["kal"])) if s["kal"] else float("nan")
        typ = s["votes"].most_common(1)[0][0] if s["votes"] else "n/a"
        rows.append([tid, s["first"], s["last"], round(sz[0], 2), round(sz[1], 2), round(fl, 2), round(ka, 2), typ])
        print(f"{tid:3d} {s['first']:5d}-{s['last']:<5d} {sz[0]:8.2f} x {sz[1]:<5.2f} {fl:19.2f} {ka:13.2f} {typ:>10s}")
    with open(os.path.join(common.OUT, "pipeline_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "first_frame", "last_frame", "size_long_cm", "size_short_cm", "speed_flow_cm_s", "speed_kalman_cm_s", "type"])
        w.writerows(rows)
    print("\nsaved output/pipeline_results.csv" + ("" if a.no_video else " and output/pipeline_out.mp4"))


if __name__ == "__main__":
    main()
