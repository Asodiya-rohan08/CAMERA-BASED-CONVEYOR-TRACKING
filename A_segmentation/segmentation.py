"""MODULE A - Segmentation: finding the boxes in each frame.

Run:  python A_segmentation/segmentation.py --video data/video.mp4
Produces output/segmentation_grid.png comparing 8 methods on 5 frames.

Also exports detect_boxes(), the production detector used by Modules C, D and the pipeline
(background subtraction + distance-transform watershed to split touching boxes).
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import segmentation as seg, color, filters
try:
    from skimage import graph                # scikit-image >= 0.20
except ImportError:
    from skimage.future import graph          # older versions (the name used in the assignment text)
from sklearn.cluster import MeanShift, estimate_bandwidth


# ----------------------------------------------------------------------------
# Production detector (used by the rest of the project)
# ----------------------------------------------------------------------------
def foreground_mask(frame, bg, min_diff=25):
    a = cv2.GaussianBlur(frame, (5, 5), 0)
    b = cv2.GaussianBlur(bg, (5, 5), 0)
    diff = cv2.absdiff(a, b).max(axis=2)
    t, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ((diff > max(0.6 * t, min_diff)) * 255).astype(np.uint8)   # 0.6*Otsu: sits nearer the true edge
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    # fill holes (box regions that look like the belt)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cnts, -1, 255, -1)
    return filled


def cut_at_concavities(mask, min_depth_ratio=0.03, min_depth_px=4.0):
    """Two touching boxes leave notches (concavities) where their outlines meet. Join pairs of deep
    notches by a thin cut so each box gets its own marker. Returns the mask with cuts drawn as 0."""
    out = mask.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 400 or len(c) < 10:
            continue
        hull = cv2.convexHull(c, returnPoints=False)
        try:
            defects = cv2.convexityDefects(c, hull)
        except cv2.error:
            continue
        if defects is None:
            continue
        defects = defects.reshape(-1, 4)
        dmin = max(min_depth_px, min_depth_ratio * np.sqrt(area))
        M = cv2.moments(c)
        th = 0.5 * np.arctan2(2 * M['mu11'], M['mu20'] - M['mu02'])       # principal axis of the blob
        axis = np.array([np.cos(th), np.sin(th)])
        pts = []
        for start_idx, end_idx, far_idx, depth in defects:
            if depth / 256.0 >= dmin:
                pts.append(tuple(int(v) for v in c[far_idx][0]))
        pairs = sorted((np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]), i, j)
                       for i in range(len(pts)) for j in range(i + 1, len(pts)))
        used = set()
        for _, i, j in pairs:
            if i in used or j in used:
                continue
            p, q = pts[i], pts[j]
            v = np.array([q[0] - p[0], q[1] - p[1]], float)
            if np.linalg.norm(v) < 1 or abs(v @ axis) / np.linalg.norm(v) > 0.5:   # cut must run ACROSS the blob
                continue
            t = np.linspace(0, 1, 25)
            xs = np.round(p[0] + (q[0] - p[0]) * t).astype(int)
            ys = np.round(p[1] + (q[1] - p[1]) * t).astype(int)
            if mask[ys, xs].mean() < 0.9:          # the cut must run through the blob, not outside it
                continue
            cv2.line(out, p, q, 0, 3)
            used |= {i, j}
    return out


def watershed_split(frame, mask, core_ratio=0.6):
    """Concavity cuts + distance-transform markers + cv2.watershed. Returns labels (0 = background)."""
    cut = cut_at_concavities(mask)
    dist = cv2.distanceTransform(cut, cv2.DIST_L2, 5)
    n, cc = cv2.connectedComponents(cut)
    sure_fg = np.zeros_like(mask)
    for i in range(1, n):
        m = cc == i
        sure_fg[m & (dist > core_ratio * dist[m].max())] = 255
    sure_bg = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(frame.copy(), markers.astype(np.int32))
    markers[markers <= 1] = 0
    return clean_labels(markers, mask)


def clean_labels(labels, mask, k=7):
    """Watershed can leak a thin ring of one label around a neighbouring box. Keep only the main
    compact part of every label (opening + largest component), then hand the removed thin pixels
    back to the nearest surviving label so box outlines stay complete."""
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = np.zeros_like(labels)
    for l in np.unique(labels):
        if l == 0:
            continue
        m = cv2.morphologyEx((labels == l).astype(np.uint8), cv2.MORPH_OPEN, ker)
        n, cc, stats, _ = cv2.connectedComponentsWithStats(m)
        if n <= 1:
            continue
        out[cc == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])] = l
    lab = out.astype(np.float32)
    for _ in range(k):
        dil = cv2.dilate(lab, np.ones((3, 3), np.uint8))
        lab = np.where((lab == 0) & (mask > 0), dil, lab)
    return lab.astype(np.int32)


def boxes_from_labels(labels, shape, min_area=800):
    H, W = shape[:2]
    out = []
    for l in np.unique(labels):
        if l == 0:
            continue
        m = (labels == l).astype(np.uint8)
        area = int(m.sum())
        if area < min_area:
            continue
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        M = cv2.moments(m, binaryImage=True)
        x, y, w, h = cv2.boundingRect(c)
        out.append(dict(
            centroid=(M["m10"] / M["m00"], M["m01"] / M["m00"]),
            rect=rect, corners=cv2.boxPoints(rect), bbox=(x, y, w, h), area=area,
            mask=m, contour=c,
            touches_border=bool(x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2)))
    out.sort(key=lambda b: b["centroid"][0])
    return out


def detect_boxes(frame, bg, min_area=None, core_ratio=0.6, work_width=640):
    """frame, bg: BGR uint8. Returns list of dicts (centroid, rect, corners, mask, ...).
    min_area defaults to 800 px at 640-px width and scales with the square of the resolution."""
    H, W = frame.shape[:2]
    if min_area is None:
        min_area = int(800 * (W / 640.0) ** 2)
    mask = foreground_mask(frame, bg)                    # full-resolution silhouette (accurate sizes)
    if W > work_width:
        # The watershed split is tuned/robust at ~640 px width, so decide WHICH pixels belong to which
        # box at that scale, then intersect with the full-resolution mask for accurate edges.
        s = work_width / W
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        mask_s = (cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA) > 127).astype(np.uint8) * 255
        lab_s = watershed_split(small, mask_s, core_ratio).astype(np.float32)
        lab = cv2.resize(lab_s, (W, H), interpolation=cv2.INTER_NEAREST)
        lab = cv2.dilate(lab, np.ones((2 * int(1 / s) + 1,) * 2, np.uint8))    # close the 1-2 px band at the border
        labels = np.where(mask > 0, lab, 0).astype(np.int32)
    else:
        labels = watershed_split(frame, mask, core_ratio)
    return boxes_from_labels(labels, frame.shape, min_area)


# ----------------------------------------------------------------------------
# Method 1: active contour (snake)
# ----------------------------------------------------------------------------
def m_active_contour(img, bg):
    gray = filters.gaussian(color.rgb2gray(img[..., ::-1]), 2)
    boxes = [b for b in detect_boxes(img, bg, min_area=300) if not b["touches_border"]]
    vis = img.copy()
    if not boxes:
        return vis, "no box"
    b = max(boxes, key=lambda b: b["area"])  # initialise near the largest visible box
    bx, by, bw, bh = b["bbox"]
    cx, cy = bx + bw / 2, by + bh / 2
    s = np.linspace(0, 2 * np.pi, 200)
    # ellipse slightly LARGER than the box's bounding box; the snake shrinks onto the edge
    init = np.c_[cy + 0.78 * bh * np.sin(s), cx + 0.78 * bw * np.cos(s)]  # (row, col)
    snake = seg.active_contour(gray, init, alpha=0.05, beta=2.0, gamma=0.01, w_line=0, w_edge=3,
                               max_px_move=1.0, max_num_iter=800)
    cv2.polylines(vis, [init[:, ::-1].astype(np.int32)], True, (0, 255, 255), 1)
    cv2.polylines(vis, [snake[:, ::-1].astype(np.int32)], True, (0, 0, 255), 2)
    return vis, "1 snake"


# ----------------------------------------------------------------------------
# Methods 2-4: quadtree split, merge, split & merge (written from scratch)
# ----------------------------------------------------------------------------
def quadtree_split(gray, var_thr=60.0, min_size=4):
    """Recursively split a region into 4 quadrants while its intensity variance > var_thr."""
    labels = np.zeros(gray.shape, np.int32)
    counter = [0]

    def rec(y0, y1, x0, x1):
        reg = gray[y0:y1, x0:x1]
        if (y1 - y0) <= min_size or (x1 - x0) <= min_size or reg.var() <= var_thr:
            counter[0] += 1
            labels[y0:y1, x0:x1] = counter[0]
            return
        ym, xm = (y0 + y1) // 2, (x0 + x1) // 2
        for a, b, c, d in [(y0, ym, x0, xm), (y0, ym, xm, x1), (ym, y1, x0, xm), (ym, y1, xm, x1)]:
            rec(a, b, c, d)

    rec(0, gray.shape[0], 0, gray.shape[1])
    return labels


def merge_regions(gray, labels, mean_thr=10.0):
    """Merge ADJACENT regions whose mean intensity differs < mean_thr (union-find)."""
    n = labels.max() + 1
    sums = np.bincount(labels.ravel(), gray.ravel().astype(np.float64), n)
    cnt = np.bincount(labels.ravel(), minlength=n).astype(np.float64)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    a = np.r_[labels[:, :-1].ravel(), labels[:-1, :].ravel()]
    b = np.r_[labels[:, 1:].ravel(), labels[1:, :].ravel()]
    keep = a != b
    pairs = np.unique(np.sort(np.stack([a[keep], b[keep]]), axis=0), axis=1).T
    mean0 = sums / np.maximum(cnt, 1)
    pairs = pairs[np.argsort(np.abs(mean0[pairs[:, 0]] - mean0[pairs[:, 1]]))]
    changed = True
    while changed:
        changed = False
        for p, q in pairs:
            rp, rq = find(p), find(q)
            if rp == rq:
                continue
            if abs(sums[rp] / cnt[rp] - sums[rq] / cnt[rq]) < mean_thr:
                parent[rq] = rp
                sums[rp] += sums[rq]
                cnt[rp] += cnt[rq]
                changed = True
    lut = np.array([find(i) for i in range(n)])
    return lut[labels]


def grid_labels(shape, block=8):
    ys, xs = np.mgrid[:shape[0], :shape[1]]
    ncol = (shape[1] + block - 1) // block
    return ((ys // block) * ncol + xs // block + 1).astype(np.int32)


def m_region_split(img, bg):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return quadtree_split(g, 80.0, 4)


def m_region_merge(img, bg):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return merge_regions(g, grid_labels(g.shape, 8), 10.0)


def m_split_merge(img, bg):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return merge_regions(g, quadtree_split(g, 80.0, 4), 10.0)


# ----------------------------------------------------------------------------
# Method 5: watershed  |  6: Felzenszwalb graph  |  7: mean shift  |  8: N-Cut
# ----------------------------------------------------------------------------
def m_watershed(img, bg):
    return watershed_split(img, foreground_mask(img, bg))


def m_felzenszwalb(img, bg):
    rgb = img[..., ::-1]
    labels = seg.felzenszwalb(rgb, scale=200, sigma=0.8, min_size=80)
    rag = graph.rag_mean_color(rgb, labels)  # region adjacency graph
    return labels, f"{len(np.unique(labels))} reg | RAG {rag.number_of_nodes()}n/{rag.number_of_edges()}e"


def m_meanshift(img, bg):
    small = cv2.resize(img, (80, int(80 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    ys, xs = np.mgrid[:small.shape[0], :small.shape[1]]
    X = np.c_[lab.reshape(-1, 3), 0.6 * xs.ravel(), 0.6 * ys.ravel()]  # colour + position
    bw = max(estimate_bandwidth(X, quantile=0.12, n_samples=500, random_state=0), 5.0)
    ms = MeanShift(bandwidth=bw, bin_seeding=True).fit(X)
    lab_small = ms.labels_.reshape(small.shape[:2]).astype(np.int32)
    return cv2.resize(lab_small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)


def m_ncut(img, bg):
    small = cv2.resize(img, (160, int(160 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
    rgb = small[..., ::-1]
    sl = seg.slic(rgb, n_segments=150, compactness=20, start_label=1)
    rag = graph.rag_mean_color(rgb, sl, mode="similarity")
    nc = graph.cut_normalized(sl, rag, thresh=0.001, num_cuts=10)
    return cv2.resize(nc.astype(np.int32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)


# ----------------------------------------------------------------------------
def overlay(img, labels):
    b = seg.mark_boundaries(img[..., ::-1], labels, color=(1, 1, 0), mode="thick")
    return (b * 255).astype(np.uint8)


def run_grid(video, frame_ids, out_path, width=320):
    info = common.video_info(video)
    bg_full = common.build_background(video)
    if frame_ids is None:
        frame_ids = [int(i) for i in np.linspace(0.3, 0.9, 5) * info["n"]]
    frames = common.get_frames(video, frame_ids)
    scale = width / frames[0].shape[1]
    resize = lambda im: cv2.resize(im, (width, int(im.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    frames = [resize(f) for f in frames]
    bg = resize(bg_full)

    methods = [("1 Active contour", m_active_contour), ("2 Split & merge", m_split_merge),
               ("3a Region split", m_region_split), ("3b Region merge", m_region_merge),
               ("4 Watershed", m_watershed), ("5 Felzenszwalb", m_felzenszwalb),
               ("6 Mean shift", m_meanshift), ("7 Normalized cut", m_ncut)]
    fig, axes = plt.subplots(len(frames), len(methods) + 1,
                             figsize=(2.6 * (len(methods) + 1), 2.0 * len(frames)))
    for r, fr in enumerate(frames):
        axes[r, 0].imshow(fr[..., ::-1])
        axes[r, 0].set_title("original" if r == 0 else "", fontsize=8)
        axes[r, 0].set_ylabel(f"frame {frame_ids[r]}")
        for c, (name, fn) in enumerate(methods, start=1):
            res = fn(fr, bg)
            extra = ""
            if isinstance(res, tuple):
                res, extra = res
            if res.ndim == 3:
                vis = res[..., ::-1]
            else:
                vis = overlay(fr, res)
                extra = extra or f"{len(np.unique(res))} regions"
            axes[r, c].imshow(vis)
            axes[r, c].set_title((name + "\n" + extra) if r == 0 else extra, fontsize=8)
            print(f"frame {frame_ids[r]:4d} | {name:18s} | {extra}")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    print("saved", out_path)


# ---------------------------------------------------------------------------
# COMPARISON NOTES (confirm these against YOUR grid image before submitting)
# Watershed with distance-transform markers is normally the best at separating touching
# boxes: it needs no colour difference between the two boxes, only a "neck" in the
# silhouette, and each box gets its own marker. Split&merge, Felzenszwalb, mean shift and
# N-Cut are colour/intensity driven, so two touching boxes of similar colour tend to be
# merged into one region (or one box gets over-split by its printed texture). The snake
# only follows ONE contour and depends on initialisation, so it cannot separate a pair.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(common.DATA, "video.mp4"))
    ap.add_argument("--frames", type=int, nargs="*", default=None, help="5 frame indices")
    ap.add_argument("--out", default=os.path.join(common.OUT, "segmentation_grid.png"))
    a = ap.parse_args()
    run_grid(a.video, a.frames, a.out)
