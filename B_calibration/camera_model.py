"""Camera helper: loads calib.npz and maps pixels <-> belt-plane centimetres."""
import numpy as np
import cv2


class Camera:
    def __init__(self, K, dist, rvec, tvec):
        self.K, self.dist = np.asarray(K, float), np.asarray(dist, float).ravel()
        self.R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
        self.t = np.asarray(tvec, float).reshape(3)
        self.n = self.R[:, 2]                       # belt-plane normal (camera coords)
        self.up = -1.0 if self.n[2] > 0 else 1.0    # sign that moves TOWARDS the camera

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["K"], d["dist"], d["belt_rvec"], d["belt_tvec"])

    def rays(self, pts_px):
        """Undistorted pixel -> (x/z, y/z) normalised rays, shape (N,3) with z=1."""
        p = np.asarray(pts_px, np.float64).reshape(-1, 1, 2)
        xy = cv2.undistortPoints(p, self.K, self.dist).reshape(-1, 2)
        return np.c_[xy, np.ones(len(xy))]

    def _plane_point(self, h_cm):
        return self.t + self.up * self.n * h_cm    # plane parallel to belt, h cm above it

    def to_cam3d(self, pts_px, h_cm=0.0):
        d = self.rays(pts_px)
        lam = (self.n @ self._plane_point(h_cm)) / (d @ self.n)
        return d * lam[:, None]

    def to_plane(self, pts_px, h_cm=0.0):
        """Back-project pixels onto the plane h_cm above the belt -> (N,2) cm in board XY frame."""
        P = self.to_cam3d(pts_px, h_cm)
        Xb = (P - self.t) @ self.R                  # = R^T (P - t) for each row
        return Xb[:, :2]

    def depth(self, pt_px, h_cm=0.0):
        return float(self.to_cam3d([pt_px], h_cm)[0, 2])

    def box_size_cm(self, corners_px, h_cm=0.0):
        """Full-perspective size (long, short) in cm from the 4 ordered corners of a box."""
        q = self.to_plane(corners_px, h_cm)
        e = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
        a, b = (e[0] + e[2]) / 2, (e[1] + e[3]) / 2
        return max(a, b), min(a, b)

    def displacement_cm(self, p0_px, p1_px, h_cm=0.0):
        q = self.to_plane([p0_px, p1_px], h_cm)
        return float(np.linalg.norm(q[1] - q[0]))
