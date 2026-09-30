"""MODULE B (part 2) - real-world box size under FOUR camera models.

Run:
  python B_calibration/measure_box.py --video data/video.mp4 --frame 100 \
      --ref_idx 0 --unk_idx 1 --ref_cm 8.33 5.0 --ref_h 0 --unk_h 0 [--unk_cm 10 4]

Detects the boxes of that frame (printed as a list, sorted left->right), takes box ref_idx as
the reference of known size (long, short in cm) and box unk_idx as the unknown.
--ref_h/--unk_h = height of the top face above the belt in cm (0 if flat/very thin).
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common
import numpy as np
import cv2
from camera_model import Camera
from segmentation import detect_boxes


def px_sides(corners):
    e = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    a, b = (e[0] + e[2]) / 2, (e[1] + e[3]) / 2
    return max(a, b), min(a, b)


def estimate_all_models(cam, ref, unk, ref_cm, ref_h, unk_h):
    Lr, Sr = px_sides(ref["corners"])
    Lu, Su = px_sides(unk["corners"])
    f = (cam.K[0, 0] + cam.K[1, 1]) / 2
    res = {}
    # 1) ORTHOGRAPHIC: x = X. One constant scale (cm/px) for the whole scene, calibrated on the reference.
    m = (ref_cm[0] / Lr + ref_cm[1] / Sr) / 2
    res["orthographic"] = (m * Lu, m * Su)
    # 2) WEAK PERSPECTIVE: x = (f/Z0) X, all points of an object share ONE depth Z0 (its centre depth).
    Z = cam.depth(unk["centroid"], unk_h)
    res["weak perspective"] = (Lu * Z / f, Su * Z / f)
    # 3) AFFINE: x = A X + b, a general 2x2 linear map -> separate scale per axis, calibrated on the reference.
    res["affine"] = (Lu * ref_cm[0] / Lr, Su * ref_cm[1] / Sr)
    # 4) FULL PERSPECTIVE: back-project every corner through K, R, t onto the box top plane.
    res["full perspective"] = cam.box_size_cm(unk["corners"], unk_h)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=os.path.join(common.ROOT, "B_calibration", "calib.npz"))
    ap.add_argument("--video", default=os.path.join(common.DATA, "video.mp4"))
    ap.add_argument("--frame", type=int, default=100)
    ap.add_argument("--ref_idx", type=int, default=0)
    ap.add_argument("--unk_idx", type=int, default=1)
    ap.add_argument("--ref_cm", type=float, nargs=2, required=True, metavar=("LONG", "SHORT"))
    ap.add_argument("--ref_h", type=float, default=0.0)
    ap.add_argument("--unk_h", type=float, default=0.0)
    ap.add_argument("--unk_cm", type=float, nargs=2, default=None, help="true size, only to print errors")
    a = ap.parse_args()

    cam = Camera.load(a.calib)
    bg = common.build_background(a.video)
    fr = common.get_frames(a.video, [a.frame])[0]
    boxes = [b for b in detect_boxes(fr, bg) if not b["touches_border"]]
    print("fully visible boxes (left->right):")
    for i, b in enumerate(boxes):
        print(f"  [{i}] centroid=({b['centroid'][0]:.0f},{b['centroid'][1]:.0f}) px sides={px_sides(b['corners'])[0]:.1f} x {px_sides(b['corners'])[1]:.1f}")
    ref, unk = boxes[a.ref_idx], boxes[a.unk_idx]
    res = estimate_all_models(cam, ref, unk, a.ref_cm, a.ref_h, a.unk_h)
    print(f"\nreference box {a.ref_idx}: {a.ref_cm[0]} x {a.ref_cm[1]} cm | unknown box {a.unk_idx}")
    print(f"{'camera model':18s} {'long(cm)':>9s} {'short(cm)':>10s}" + ("   error(cm)" if a.unk_cm else ""))
    for k, (L, S) in res.items():
        line = f"{k:18s} {L:9.2f} {S:10.2f}"
        if a.unk_cm:
            line += f"   {abs(L - a.unk_cm[0]):.2f} / {abs(S - a.unk_cm[1]):.2f}"
        print(line)


# ---------------------------------------------------------------------------
# DISCUSSION (confirm with your own numbers): with a ceiling camera that looks straight down
# from ~60-100 cm with a moderate field of view, all belt-top points are at nearly the same
# depth, so weak perspective, affine and full perspective agree within a few %. Full
# perspective is the most accurate because it uses the real K, R, t and each corner's own
# depth, so it stays correct when the unknown box is taller (closer to the camera) than the
# reference. Orthographic/affine reuse the reference's scale, so they are wrong by roughly the
# ratio of the two depths whenever the boxes have different heights.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
