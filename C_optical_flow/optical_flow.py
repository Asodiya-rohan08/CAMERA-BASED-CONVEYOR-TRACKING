"""MODULE C - Motion representation: how fast is the belt moving?

Run:  python C_optical_flow/optical_flow.py --video data/video.mp4 --frame 150 --box_idx 0
Outputs output/optical_flow.png with (a) hand-built dense Lucas-Kanade, (b) sparse OpenCV LK,
(c) RANSAC affine fit with inliers/outliers, (d) printed belt speed in cm/s.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from camera_model import Camera
from segmentation import detect_boxes


# ---------------------------------------------------------------------------
# (a) Lucas-Kanade from scratch
#   brightness constancy:  Ix*u + Iy*v + It = 0
#   for every pixel stack this equation over a win x win window -> least squares:
#   [sum IxIx  sum IxIy] [u]   [-sum IxIt]
#   [sum IxIy  sum IyIy] [v] = [-sum IyIt]       (solved in closed form for the 2x2 system)
# ---------------------------------------------------------------------------
def _lk_step(I1, I2w, win, min_eig):
    Ix = cv2.Sobel(I1, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    Iy = cv2.Sobel(I1, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    It = I2w - I1
    S = lambda a: cv2.boxFilter(a, -1, (win, win), normalize=False, borderType=cv2.BORDER_REPLICATE)
    Sxx, Sxy, Syy, Sxt, Syt = S(Ix * Ix), S(Ix * Iy), S(Iy * Iy), S(Ix * It), S(Iy * It)
    det = Sxx * Syy - Sxy ** 2
    tr = Sxx + Syy
    lam_min = 0.5 * (tr - np.sqrt(np.maximum(tr ** 2 - 4 * det, 0))) / (win * win)
    ok = (lam_min > min_eig) & (det > 1e-12)          # aperture-problem test: both eigenvalues large
    safe = np.where(ok, det, 1.0)
    du = np.where(ok, (-Syy * Sxt + Sxy * Syt) / safe, 0.0)
    dv = np.where(ok, (Sxy * Sxt - Sxx * Syt) / safe, 0.0)
    return du, dv, lam_min


def lucas_kanade_dense(img1, img2, win=15, levels=3, iters=3, min_eig=1e-4):
    """Coarse-to-fine iterative LK. Inputs: BGR or gray uint8. Returns flow (H,W,2) and confidence map."""
    g = lambda im: cv2.GaussianBlur((cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im)
                                    .astype(np.float32) / 255.0, (0, 0), 1.0)
    P1, P2 = [g(img1)], [g(img2)]
    for _ in range(levels - 1):
        P1.append(cv2.pyrDown(P1[-1])); P2.append(cv2.pyrDown(P2[-1]))
    flow = None
    for L in range(levels - 1, -1, -1):
        I1, I2 = P1[L], P2[L]
        h, w = I1.shape
        flow = np.zeros((h, w, 2), np.float32) if flow is None else \
            cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) * 2.0
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        for _ in range(iters):
            I2w = cv2.remap(I2, xs + flow[..., 0], ys + flow[..., 1], cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
            du, dv, conf = _lk_step(I1, I2w, win, min_eig)
            flow[..., 0] += du; flow[..., 1] += dv
    return flow, conf


def draw_flow(img, pts, vecs, color=(0, 255, 0), scale=4.0):
    out = img.copy()
    for (x, y), (u, v) in zip(pts, vecs):
        cv2.arrowedLine(out, (int(x), int(y)), (int(x + scale * u), int(y + scale * v)), color, 1, tipLength=0.3)
    return out


def dense_points(flow, conf, step=12, min_eig=1e-4, mask=None):
    h, w = conf.shape
    ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
    xs, ys = xs.ravel(), ys.ravel()
    ok = conf[ys, xs] > min_eig
    if mask is not None:
        ok &= mask[ys, xs] > 0
    return np.c_[xs[ok], ys[ok]], flow[ys[ok], xs[ok]]


# ---------------------------------------------------------------------------
# (b) sparse feature-based flow (OpenCV)
# ---------------------------------------------------------------------------
def sparse_flow(img1, img2, mask=None, max_corners=300):
    g1, g2 = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in (img1, img2)]
    p0 = cv2.goodFeaturesToTrack(g1, max_corners, 0.01, 7, mask=mask)
    if p0 is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, p0, None, winSize=(21, 21), maxLevel=3)
    st = st.ravel() == 1
    return p0.reshape(-1, 2)[st], (p1 - p0).reshape(-1, 2)[st]


# ---------------------------------------------------------------------------
# (c) 6-parameter affine flow model + RANSAC
#   u(x,y) = a1 + a2 x + a3 y ;  v(x,y) = a4 + a5 x + a6 y
# ---------------------------------------------------------------------------
def fit_affine(pts, vecs):
    A = np.c_[np.ones(len(pts)), pts[:, 0], pts[:, 1]]
    pu = np.linalg.lstsq(A, vecs[:, 0], rcond=None)[0]
    pv = np.linalg.lstsq(A, vecs[:, 1], rcond=None)[0]
    return np.vstack([pu, pv])  # 2x3


def predict_affine(params, pts):
    A = np.c_[np.ones(len(pts)), pts[:, 0], pts[:, 1]]
    return A @ params.T


def ransac_affine(pts, vecs, thr=0.5, iters=300, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        idx = rng.choice(len(pts), 3, replace=False)
        p = pts[idx]
        if abs(np.linalg.det(np.c_[np.ones(3), p])) < 1e-3:   # collinear sample
            continue
        prm = fit_affine(p, vecs[idx])
        inl = np.linalg.norm(predict_affine(prm, pts) - vecs, axis=1) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 3:
        best = np.ones(len(pts), bool)
    return fit_affine(pts[best], vecs[best]), best


# ---------------------------------------------------------------------------
# (d) belt speed from the affine flow at the box centroid + calibration
# ---------------------------------------------------------------------------
def estimate_belt_speed(f1, f2, mask, fps, gap=1, cam=None, box_h=0.0, cm_per_px=None):
    """mask = box mask in frame f1. Returns dict or None."""
    flow, conf = lucas_kanade_dense(f1, f2)
    m = cv2.erode(mask, np.ones((7, 7), np.uint8))
    pts, vecs = dense_points(flow, conf, step=3, mask=m)
    src = "dense LK"
    if len(pts) < 10:                                   # textureless box: fall back to feature flow
        pts, vecs = sparse_flow(f1, f2, mask=m)
        src = "sparse LK"
    if len(pts) < 6:
        return None
    params, inl = ransac_affine(pts, vecs)
    M = cv2.moments(mask, binaryImage=True)
    c = np.array([[M["m10"] / M["m00"], M["m01"] / M["m00"]]])
    d = predict_affine(params, c)[0]                    # px displacement of the centroid
    dt = gap / fps
    if cam is not None:
        dist_cm = cam.displacement_cm(c[0], c[0] + d, box_h)
    else:
        dist_cm = np.linalg.norm(d) * cm_per_px
    return dict(speed_cm_s=dist_cm / dt, flow_px=d, params=params, pts=pts, vecs=vecs, inliers=inl, source=src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(common.DATA, "video.mp4"))
    ap.add_argument("--frame", type=int, default=150)
    ap.add_argument("--gap", type=int, default=1)
    ap.add_argument("--box_idx", type=int, default=0)
    ap.add_argument("--box_h", type=float, default=0.0)
    ap.add_argument("--calib", default=os.path.join(common.ROOT, "B_calibration", "calib.npz"))
    ap.add_argument("--inject_outliers", type=float, default=0.1,
                    help="fraction of vectors corrupted on purpose in panel (c) to demo RANSAC; 0 = off. "
                         "The printed belt speed always uses the clean, un-corrupted fit.")
    ap.add_argument("--cm_per_px", type=float, default=None, help="used only if no calibration file")
    a = ap.parse_args()

    info = common.video_info(a.video)
    f1, f2 = common.get_frames(a.video, [a.frame, a.frame + a.gap])
    bg = common.build_background(a.video)
    boxes = [b for b in detect_boxes(f1, bg) if not b["touches_border"]]
    if not boxes:
        sys.exit('No fully visible box in this frame - choose another --frame.')
    if a.box_idx >= len(boxes):
        sys.exit(f'--box_idx {a.box_idx} out of range: only {len(boxes)} fully visible box(es).')
    box = boxes[a.box_idx]
    cam = Camera.load(a.calib) if os.path.exists(a.calib) else None

    # (a) hand-built dense LK
    flow, conf = lucas_kanade_dense(f1, f2)
    pts_d, vec_d = dense_points(flow, conf, step=14)
    # (b) sparse feature LK
    pts_s, vec_s = sparse_flow(f1, f2)
    # (c) affine + RANSAC on the chosen box (noise injected into some vectors to show outlier rejection)
    if cam is None and a.cm_per_px is None:
        sys.exit('No calibration file: run B_calibration/calibrate.py or pass --cm_per_px')
    res = estimate_belt_speed(f1, f2, box["mask"], info["fps"], a.gap, cam, a.box_h, a.cm_per_px)
    if res is None:
        sys.exit('Too few flow vectors on this box (textureless?) - try another --box_idx/--frame.')
    pts, vecs = res["pts"].copy(), res["vecs"].copy()
    rng = np.random.default_rng(1)
    bad = rng.random(len(pts)) < a.inject_outliers          # simulated gross errors (DEMO of RANSAC only)
    vecs[bad] += rng.normal(0, 6, (bad.sum(), 2))
    params, inl = ransac_affine(pts, vecs)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].imshow(draw_flow(f1, pts_d, vec_d)[..., ::-1]); ax[0].set_title(f"(a) Lucas-Kanade from scratch ({len(pts_d)} confident vectors)")
    ax[1].imshow(draw_flow(f1, pts_s, vec_s, (0, 200, 255))[..., ::-1]); ax[1].set_title(f"(b) sparse feature LK ({len(pts_s)} corners)")
    x, y, w, h = box["bbox"]
    crop = f1[max(y - 10, 0):y + h + 10, max(x - 10, 0):x + w + 10]
    ax[2].imshow(crop[..., ::-1], extent=(max(x - 10, 0), max(x - 10, 0) + crop.shape[1], max(y - 10, 0) + crop.shape[0], max(y - 10, 0)))
    ax[2].scatter(*pts[inl].T, c="lime", s=8, label=f"inliers {inl.sum()}")
    ax[2].scatter(*pts[~inl].T, c="red", s=25, marker="x", label=f"outliers {(~inl).sum()}")
    ax[2].legend(); ax[2].set_title(f"(c) RANSAC affine fit ({100 * a.inject_outliers:.0f}% vectors corrupted on purpose)")
    for q in ax:
        q.axis("off")
    plt.tight_layout()
    p = os.path.join(common.OUT, "optical_flow.png")
    plt.savefig(p, dpi=100)
    print("saved", p)
    print("affine params [a1 a2 a3; a4 a5 a6] =\n", np.round(params, 4))
    print(f"centroid flow = ({res['flow_px'][0]:.2f}, {res['flow_px'][1]:.2f}) px/frame  [{res['source']}]")
    print(f"ESTIMATED BELT SPEED = {res['speed_cm_s']:.2f} cm/s")


# ---------------------------------------------------------------------------
# DISCUSSION (dense differential LK vs sparse feature LK)
# Dense LK produces a vector at every textured pixel, but on a flat-coloured box face the
# structure tensor is (near) singular: only the gradient along one direction is visible, so
# the aperture problem lets only the normal flow be recovered along box edges and nothing at
# all inside the face (we mask these pixels via the min-eigenvalue test). Sparse feature LK
# only tracks corners, which have two strong gradient directions, so its vectors are reliable
# but sit on edges/printed marks and leave textureless regions empty. Where the two disagree
# is exactly on edges and flat regions; the 6-parameter affine fit + RANSAC regularises this by
# forcing one rigid motion on all vectors of the box and discarding outliers.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
