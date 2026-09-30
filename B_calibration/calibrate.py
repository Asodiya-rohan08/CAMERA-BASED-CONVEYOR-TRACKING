"""MODULE B (part 1) - direct parameter calibration with a checkerboard.

Run:
  python B_calibration/calibrate.py --images "data/calib/*.png" --cols 9 --rows 6 --square_cm 3.0
(--cols/--rows = INNER corners.)  Saves B_calibration/calib.npz (K, dist, extrinsics),
prints K/distortion, shows undistort before/after, and decomposes P = K[R|t].

IMPORTANT for real data: capture ONE extra photo with the checkerboard lying FLAT on the belt,
camera fixed exactly as when filming. Pass its index with --belt_idx: its pose defines the belt plane.
"""
import os, sys, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common
import numpy as np
import cv2
from scipy.linalg import rq


def decompose_P(P):
    """Manual decomposition P = K [R|t] via RQ decomposition of the left 3x3 block."""
    P = P.copy()
    if np.linalg.det(P[:, :3]) < 0:
        P = -P                                   # P is only defined up to scale (incl. sign)
    Kx, Rx = rq(P[:, :3])
    T = np.diag(np.sign(np.diag(Kx)))            # force positive focal lengths
    Kx, Rx = Kx @ T, T @ Rx
    tx = np.linalg.inv(Kx) @ P[:, 3]
    return Kx / Kx[2, 2], Rx, tx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(common.DATA, "calib", "*.png"))
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square_cm", type=float, default=3.0)
    ap.add_argument("--belt_idx", type=int, default=0, help="index of the image with board flat on belt")
    ap.add_argument("--out", default=os.path.join(common.ROOT, "B_calibration", "calib.npz"))
    ap.add_argument("--sample", default=None, help="image to undistort for the demo (default: 1st calib image)")
    a = ap.parse_args()

    files = sorted(glob.glob(a.images))
    if not files:
        sys.exit(f"No images match {a.images}")
    objp = np.zeros((a.rows * a.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[:a.cols, :a.rows].T.reshape(-1, 2) * a.square_cm
    objpts, imgpts, used = [], [], []
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    size = None
    for f in files:
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        size = g.shape[::-1]
        ok, c = cv2.findChessboardCorners(g, (a.cols, a.rows))
        if not ok:
            print("  corners NOT found:", os.path.basename(f))
            continue
        c = cv2.cornerSubPix(g, c, (11, 11), (-1, -1), crit)
        objpts.append(objp); imgpts.append(c); used.append(f)
    print(f"boards detected in {len(used)}/{len(files)} images")
    if len(used) < 8:
        sys.exit("Need at least ~10 good views for a stable calibration.")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, size, None, None)
    print("\nRMS reprojection error (px):", round(rms, 4))
    print("Intrinsic matrix K =\n", np.round(K, 3))
    print(f"fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
    k = dist.ravel()
    print(f"radial k1={k[0]:.5f} k2={k[1]:.5f} k3={k[4]:.5f} | tangential p1={k[2]:.5f} p2={k[3]:.5f}")

    bi = min(a.belt_idx, len(used) - 1)
    np.savez(a.out, K=K, dist=dist, rms=rms, image_size=np.array(size), square_cm=a.square_cm,
             belt_rvec=rvecs[bi], belt_tvec=tvecs[bi], all_rvecs=np.array(rvecs), all_tvecs=np.array(tvecs))
    print("saved", a.out, "(belt plane = pose of", os.path.basename(used[bi]) + ")")

    # ---- undistort before / after --------------------------------------------------
    sample = cv2.imread(a.sample or used[0])
    h, w = sample.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)
    und = cv2.undistort(sample, K, dist, None, newK)
    both = np.hstack([sample, und])
    cv2.putText(both, "original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(both, "undistorted", (w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    p = os.path.join(common.OUT, "undistort_before_after.png")
    cv2.imwrite(p, both)
    print("saved", p)

    # ---- extrinsics + manual P = K[R|t] decomposition -------------------------------
    R, _ = cv2.Rodrigues(rvecs[bi])
    t = tvecs[bi].reshape(3)
    P = K @ np.hstack([R, t.reshape(3, 1)])
    K2, R2, t2 = decompose_P(P)
    print("\nExtrinsics of belt image: R =\n", np.round(R, 4), "\n t (cm) =", np.round(t, 3))
    print("Projection matrix P = K[R|t] =\n", np.round(P, 3))
    print("Manual RQ decomposition recovered:")
    print(" K err  :", np.abs(K2 - K).max(), "\n R err  :", np.abs(R2 - R).max(), "\n t err  :", np.abs(t2 - t).max())
    # cross-check with OpenCV's own decomposition
    Kc, Rc, tc = cv2.decomposeProjectionMatrix(P)[:3]
    print(" OpenCV decomposeProjectionMatrix K err:", np.abs(Kc / Kc[2, 2] - K).max())
    print(f" camera height above belt plane ~ {abs(R.T @ -t)[2]:.1f} cm")


if __name__ == "__main__":
    main()
