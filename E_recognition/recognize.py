"""MODULE E - Object recognition: what type of box is it?

Data layout:  data/boxes/<type_name>/*.png   (>=3 types, ~10 crops each; see collect_crops.py)
Run:          python E_recognition/recognize.py
Prints an accuracy table (clean / rotated / lighting-changed test crops), saves
output/eigenboxes.png (mean + top-3 eigenvectors) and shows Hu-moment invariants per class.

Classifiers:
  1. AlignmentClassifier : ORB keypoints + RANSAC homography to each template, score = residual
  2. EigenBoxClassifier  : PCA ("eigenboxes") + nearest neighbour in eigenspace
  3. HuClassifier        : Hu-moment invariants + nearest class mean
"""
import os, sys, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIZE = common.CROP_SIZE  # (w, h)


# ------------------------------------------------------------------ data
def load_dataset(root):
    X, y = [], []
    for cls in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        for f in sorted(glob.glob(os.path.join(root, cls, "*.*"))):
            im = cv2.imread(f)
            if im is not None:
                X.append(cv2.resize(im, SIZE)); y.append(cls)
    return X, np.array(y)


def split(X, y, train_frac=0.6, seed=0):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        k = max(1, int(round(train_frac * len(idx))))
        tr += list(idx[:k]); te += list(idx[k:])
    return tr, te


def rotate(img, ang):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def change_light(img, alpha, gamma):
    return np.clip(255 * (img / 255.0) ** gamma * alpha, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ 1. alignment
class AlignmentClassifier:
    """Align query to every stored template with a homography from ORB matches (RANSAC).
    cost = (mean inlier residual + 0.5) / n_inliers^2 -> low residual AND many supporting points."""
    def __init__(self):
        self.orb = cv2.ORB_create(500, 1.2, 4, edgeThreshold=15, patchSize=15, fastThreshold=10)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def _feat(self, im):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=2, fy=2)
        return self.orb.detectAndCompute(g, None)

    def fit(self, X, y):
        self.T = [(self._feat(x), c) for x, c in zip(X, y)]
        return self

    def cost(self, q, t):
        (kq, dq), (kt, dt) = q, t
        if dq is None or dt is None or len(kq) < 4 or len(kt) < 4:
            return np.inf, 0
        good = [m[0] for m in self.bf.knnMatch(dq, dt, k=2) if len(m) == 2 and m[0].distance < 0.85 * m[1].distance]
        if len(good) < 4:
            return np.inf, len(good)
        src = np.float32([kq[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kt[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None or inl.sum() < 8:          # a homography from <8 points is easily a fluke
            return np.inf, len(good)
        inl = inl.ravel() == 1
        res = np.linalg.norm(cv2.perspectiveTransform(src, H).reshape(-1, 2)[inl] - dst.reshape(-1, 2)[inl], axis=1).mean()
        return (res + 0.5) / inl.sum() ** 2, int(inl.sum())

    def predict(self, im):
        q = self._feat(im)
        best, lab, fallback = np.inf, "unknown", (0, "unknown")
        for t, c in self.T:
            cst, n = self.cost(q, t)
            if cst < best:
                best, lab = cst, c
            if n > fallback[0]:
                fallback = (n, c)
        return lab if np.isfinite(best) else fallback[1]


# ------------------------------------------------------------------ 2. eigenboxes
class EigenBoxClassifier:
    def __init__(self, var_keep=0.95, size=(48, 32)):
        self.var_keep, self.size = var_keep, size

    def _vec(self, im):
        g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), self.size).astype(np.float64).ravel()
        return (g - g.mean()) / (g.std() + 1e-6)          # per-image normalisation = lighting robustness

    def fit(self, X, y):
        rows, labs = [], []
        for x, c in zip(X, y):
            for v in (x, np.rot90(x, 2)):                  # crops are ambiguous up to 180 deg -> augment
                rows.append(self._vec(np.ascontiguousarray(v))); labs.append(c)
        A = np.array(rows)
        self.mean = A.mean(0)
        U, S, Vt = np.linalg.svd(A - self.mean, full_matrices=False)   # PCA via SVD
        var = S ** 2 / (S ** 2).sum()
        k = int(np.searchsorted(np.cumsum(var), self.var_keep) + 1)
        self.V, self.k, self.var = Vt[:k], k, var
        self.Z, self.y = (A - self.mean) @ self.V.T, np.array(labs)
        return self

    def predict(self, im):
        z = (self._vec(im) - self.mean) @ self.V.T
        return self.y[np.argmin(np.linalg.norm(self.Z - z, axis=1))]

    def save_figure(self, path):
        w, h = self.size
        fig, ax = plt.subplots(1, 4, figsize=(10, 2.8))
        ax[0].imshow(self.mean.reshape(h, w), cmap="gray"); ax[0].set_title("mean box")
        for i in range(3):
            ax[i + 1].imshow(self.V[i].reshape(h, w), cmap="gray")
            ax[i + 1].set_title(f"eigenbox {i + 1} ({100 * self.var[i]:.1f}% var)")
        for a in ax: a.axis("off")
        plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


# ------------------------------------------------------------------ 3. Hu invariants
def hu_features(im):
    """Hu moments of the (circular-masked, brightness-normalised) grey image, log-scaled.
    Circular support => a rotated crop contains the same pixels; dividing by the mean => brightness invariant."""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = g.shape
    yy, xx = np.mgrid[:h, :w]
    m = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) <= (min(h, w) / 2 - 1) ** 2
    g = np.where(m, g, 0.0)
    g = g / (g[m].mean() + 1e-6)
    hu = cv2.HuMoments(cv2.moments(g)).ravel()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)


class HuClassifier:
    def __init__(self, n_hu=4):
        self.n = n_hu

    def fit(self, X, y):
        F = np.array([hu_features(x)[:self.n] for x in X])
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-6
        Z = (F - self.mu) / self.sd
        self.cls = np.unique(y)
        self.proto = np.array([Z[y == c].mean(0) for c in self.cls])
        return self

    def predict(self, im):
        z = (hu_features(im)[:self.n] - self.mu) / self.sd
        return self.cls[np.argmin(np.linalg.norm(self.proto - z, axis=1))]


# ------------------------------------------------------------------ evaluation
def make_variants(X_te, y_te, kind, n=5, seed=1):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for x, c in zip(X_te, y_te):
        if kind == "clean":
            Xs.append(x); ys.append(c)
            continue
        for _ in range(n):
            if kind == "rotated":
                Xs.append(rotate(x, rng.uniform(0, 360)))
            else:  # lighting
                Xs.append(change_light(x, rng.uniform(0.5, 1.5), rng.uniform(0.7, 1.4)))
            ys.append(c)
    return Xs, np.array(ys)


def train_all(X, y, n_hu=4):
    return {"Alignment (ORB+homography)": AlignmentClassifier().fit(X, y),
            "Eigenboxes (PCA + 1-NN)": EigenBoxClassifier().fit(X, y),
            "Hu moments (nearest mean)": HuClassifier(n_hu).fit(X, y)}


def accuracy(clf, X, y):
    return float(np.mean([clf.predict(x) == t for x, t in zip(X, y)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(common.DATA, "boxes"))
    ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--n_hu", type=int, default=4)
    a = ap.parse_args()

    X, y = load_dataset(a.data)
    classes = np.unique(y)
    if len(classes) < 2:
        sys.exit(f'Need >=2 class folders with images in {a.data}')
    print(f"{len(X)} crops, {len(classes)} classes: {[str(c) for c in classes]}")
    tr, te = split(X, y, a.train_frac)
    Xtr, ytr = [X[i] for i in tr], y[tr]
    Xte, yte = [X[i] for i in te], y[te]
    if not te:
        sys.exit('No held-out test crops: put at least 2 images per class.')
    print(f"train={len(tr)} test={len(te)} (held-out crops are never used for fitting)\n")
    models = train_all(Xtr, ytr, a.n_hu)
    models["Eigenboxes (PCA + 1-NN)"].save_figure(os.path.join(common.OUT, "eigenboxes.png"))
    print("saved output/eigenboxes.png  (eigenspace dim =", models["Eigenboxes (PCA + 1-NN)"].k, ")")

    # Hu invariants per class + rotation check
    print("\nHu moment invariants (log10, first 4) - class means, and mean |change| when the crop is rotated:")
    for c in classes:
        F = np.array([hu_features(x)[:4] for x, l in zip(Xtr, ytr) if l == c])
        Fr = np.array([hu_features(rotate(x, 47))[:4] for x, l in zip(Xtr, ytr) if l == c])
        print(f"  {c:10s} {np.round(F.mean(0), 3)}   rotation change {np.abs(F - Fr).mean():.3f}")

    conds = ["clean", "rotated", "lighting"]
    table = {n: [] for n in models}
    for cnd in conds:
        Xv, yv = make_variants(Xte, yte, cnd)
        for n, m in models.items():
            table[n].append(accuracy(m, Xv, yv))
    print("\nACCURACY ON HELD-OUT TEST CROPS")
    print(f"{'method':30s}" + "".join(f"{c:>10s}" for c in conds))
    for n, v in table.items():
        print(f"{n:30s}" + "".join(f"{100 * s:9.1f}%" for s in v))


# ---------------------------------------------------------------------------
# ROBUSTNESS NOTES (measured on the synthetic smoke-test set - re-check with YOUR table)
# Hu moments: rotation-invariant by construction (circular support + normalisation) and
# brightness-invariant, so they held 100% on rotated and re-lit crops - but they only encode
# coarse intensity layout, so they can fail when box types differ only in colour.
# Eigenboxes: per-image mean/std normalisation makes them robust to LIGHTING, but PCA on raw
# pixels is not rotation-invariant (only the 180-degree flip is in training), so accuracy drops
# sharply on rotated crops.
# Alignment (ORB + homography): in theory rotation-invariant, but on tiny 96x64 crops a rotation
# cuts corners off and leaves few keypoints, so it was the weakest on rotation here and also
# sensitive to strong lighting changes; it works best on high-resolution, well-textured crops.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
