"""Helper: harvest box crops from your video so you do not have to crop by hand.

  python E_recognition/collect_crops.py --video data/video.mp4 --k 3 --per_frame_step 10
Detects fully visible boxes every N frames, deskews them to 96x64 and clusters them (KMeans on
colour + aspect ratio) into k folders  data/boxes/type_0 ... type_{k-1}.
LOOK AT THE FOLDERS, fix mistakes by moving files, and rename folders to real names.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common
import numpy as np, cv2
from sklearn.cluster import KMeans
from segmentation import detect_boxes

ap = argparse.ArgumentParser()
ap.add_argument("--video", default=os.path.join(common.DATA, "video.mp4"))
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--per_frame_step", type=int, default=10)
ap.add_argument("--max_per_class", type=int, default=15)
ap.add_argument("--out", default=os.path.join(common.DATA, "boxes_auto"))
a = ap.parse_args()

bg = common.build_background(a.video)
crops, feats = [], []
for i, fr in common.iter_video(a.video):
    if i % a.per_frame_step:
        continue
    for b in detect_boxes(fr, bg):
        if b["touches_border"]:
            continue
        c = common.crop_box(fr, b["rect"])
        hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).reshape(-1, 3).mean(0)
        w, h = b["rect"][1]
        crops.append(c); feats.append([hsv[0] * 2, hsv[1] / 2, hsv[2] / 4, 40 * max(w, h) / max(min(w, h), 1)])
lab = KMeans(a.k, n_init=10, random_state=0).fit_predict(np.array(feats))
for k in range(a.k):
    d = os.path.join(a.out, f"type_{k}"); os.makedirs(d, exist_ok=True)
    idx = np.where(lab == k)[0]
    for j, i in enumerate(idx[np.linspace(0, len(idx) - 1, min(len(idx), a.max_per_class)).astype(int)]):
        cv2.imwrite(os.path.join(d, f"{k}_{j:02d}.png"), crops[i])
    print(f"type_{k}: {len(idx)} crops found, saved up to {a.max_per_class}")
print("->", a.out, "(review + rename, then move into data/boxes/)")
