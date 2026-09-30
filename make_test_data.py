"""Generate a SYNTHETIC test set so you can smoke-test every script end-to-end.

!!! The assignment requires REAL footage. Use this only to check that the code runs.
Creates:
  data/video.mp4            360 frames, 3 box types on a moving 'belt' (10 cm/s)
  data/calib/*.png          checkerboard views rendered from a true pinhole camera
  data/boxes/<type>/*.png   deskewed 96x64 crops for module E
Ground truth: 12 px = 1 cm on the belt plane, belt speed 4 px/frame = 10 cm/s at 30 fps.
"""
import os, json, numpy as np, cv2
from common import DATA

W, H, FPS, N = 640, 480, 30, 360
PPC = 12.0                              # pixels per cm on the belt plane
F, CX, CY = 800.0, 320.0, 240.0
K = np.array([[F, 0, CX], [0, F, CY], [0, 0, 1.0]])
Z0 = F / PPC                            # camera height above belt (cm)
SPEED = 4.0                             # px / frame

KINDS = {  # name: (w_px, h_px, BGR colour)
    "typeA": (100, 60, (40, 40, 200)),
    "typeB": (66, 88, (200, 90, 30)),
    "typeC": (120, 48, (60, 170, 60)),
}


def sprite(kind, rng):
    w, h, col = KINDS[kind]
    img = np.zeros((h, w, 3), np.float32) + col
    if kind == "typeA":
        cv2.line(img, (0, h // 2), (w, h // 2 - 20), (255, 255, 255), 6)
        cv2.circle(img, (w // 4, h // 2 + 10), 8, (0, 0, 0), -1)
    elif kind == "typeB":
        for yy in range(12, h, 20):
            for xx in range(12, w, 20):
                cv2.circle(img, (xx, yy), 4, (255, 255, 255), -1)
    else:
        cv2.line(img, (5, 5), (w - 5, h - 5), (0, 0, 0), 4)
        cv2.line(img, (5, h - 5), (w - 5, 5), (0, 0, 0), 4)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 0), 3)
    img += rng.normal(0, 10, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def belt_background(rng):
    bg = np.full((H, W, 3), 95, np.float32)
    bg += rng.normal(0, 4, (H, W, 1))
    bg[::40, :, :] -= 12
    return np.clip(cv2.GaussianBlur(bg, (3, 3), 0), 0, 255).astype(np.uint8)


def paste(canvas, spr, cx, cy, ang):
    h, w = spr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    M[0, 2] += cx - w / 2
    M[1, 2] += cy - h / 2
    warped = cv2.warpAffine(spr, M, (W, H), flags=cv2.INTER_LINEAR)
    alpha = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (W, H), flags=cv2.INTER_NEAREST)
    m = alpha > 200
    canvas[m] = warped[m]


def make_video(rng):
    bg = belt_background(rng)
    kinds = ["typeA", "typeC", "typeB", "typeA", "typeB"]
    boxes = []
    for i, s in enumerate([0, 70, 140, 210, 280]):
        boxes.append((s, 130, kinds[i], rng.uniform(-6, 6)))
    kinds2 = ["typeB", "typeA", "typeC", "typeC", "typeA"]
    for i, s in enumerate([35, 105, 175, 245, 315]):
        boxes.append((s, 350, kinds2[i], rng.uniform(-6, 6)))
    sprites = {}
    # touching pair in middle lane (C in front, A directly behind)
    boxes.append((190, 240, "typeC", 0.0))
    boxes.append((190 - int(110 / SPEED), 240, "typeA", 0.0))
    path = os.path.join(DATA, "video.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for t in range(N):
        fr = bg.copy()
        for k, (t0, y, kind, ang) in enumerate(boxes):
            x = -80 + SPEED * (t - t0)
            if -80 <= x <= W + 80 and t >= t0 - 30:
                key = k
                if key not in sprites:
                    sprites[key] = sprite(kind, rng)
                paste(fr, sprites[key], x, y, ang)
        fr = np.clip(fr.astype(np.float32) + rng.normal(0, 2, fr.shape), 0, 255).astype(np.uint8)
        vw.write(fr)
    vw.release()
    print("video ->", path)


def make_calib(rng, n_keep=14):
    os.makedirs(os.path.join(DATA, "calib"), exist_ok=True)
    sq, cols, rows, ppc_t, m = 3.0, 10, 7, 20, 60
    tw, th = int(cols * sq * ppc_t + 2 * m), int(rows * sq * ppc_t + 2 * m)
    tex = np.full((th, tw), 255, np.uint8)
    s = int(sq * ppc_t)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                tex[m + r * s:m + (r + 1) * s, m + c * s:m + (c + 1) * s] = 0
    A = np.array([[1 / ppc_t, 0, -tw / 2 / ppc_t], [0, 1 / ppc_t, -th / 2 / ppc_t], [0, 0, 1]])
    saved, tries = 0, 0
    while saved < n_keep and tries < 200:
        if saved == 0:
            R, t = np.eye(3), np.array([0, 0, Z0])
        else:
            rv = np.deg2rad(rng.uniform(-30, 30, 3)) * np.array([1, 1, 0.6])
            R, _ = cv2.Rodrigues(rv)
            t = np.array([rng.uniform(-4, 4), rng.uniform(-3, 3), rng.uniform(50, 80)])
        Hm = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ A
        img = cv2.warpPerspective(tex, Hm, (W, H), flags=cv2.INTER_AREA, borderValue=110)
        img = np.clip(img.astype(np.float32) + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
        tries += 1
        ok, _ = cv2.findChessboardCorners(img, (cols - 1, rows - 1))
        if ok:
            cv2.imwrite(os.path.join(DATA, "calib", f"calib_{saved:02d}.png"), img)
            saved += 1
    print(f"calibration images -> {saved} saved")


def make_crops(rng, per_class=12):
    for kind in KINDS:
        d = os.path.join(DATA, "boxes", kind)
        os.makedirs(d, exist_ok=True)
        for i in range(per_class):
            spr = sprite(kind, rng)
            if spr.shape[0] > spr.shape[1]:
                spr = np.rot90(spr, int(rng.choice([1, 3]))).copy()
            if rng.random() < 0.5:
                spr = np.rot90(spr, 2).copy()
            crop = cv2.resize(spr, (96, 64))
            a, b = rng.uniform(0.85, 1.15), rng.uniform(-12, 12)
            crop = np.clip(crop.astype(np.float32) * a + b + rng.normal(0, 4, crop.shape), 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(d, f"{kind}_{i:02d}.png"), crop)
    print("box crops ->", os.path.join(DATA, "boxes"))


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    rng = np.random.default_rng(0)
    make_video(rng)
    make_calib(rng)
    make_crops(rng)
    gt = {"px_per_cm": PPC, "belt_speed_cm_s": SPEED * FPS / PPC,
          "sizes_cm": {k: sorted([round(v[0] / PPC, 2), round(v[1] / PPC, 2)], reverse=True) for k, v in KINDS.items()}}
    json.dump(gt, open(os.path.join(DATA, "ground_truth.json"), "w"), indent=2)
    print(gt)
