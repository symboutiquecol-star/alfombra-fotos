import sys, os, numpy as np, cv2
WOODMD = float(os.environ.get('WOODMD', '2.6'))
MD2 = float(os.environ.get('MD2', '2.2'))
from PIL import Image, ImageOps
import pillow_heif; pillow_heif.register_heif_opener()
from colorize import colorize

COLORS = {'azul': (118, 182, 232), 'crema': (222, 196, 160)}
WHITE = (251, 248, 242)


def box(x, r):
    return cv2.blur(x, (2 * r + 1, 2 * r + 1))


def guided(I, p, r, eps):
    mI, mp = box(I, r), box(p, r)
    a = (box(I * p, r) - mI * mp) / (box(I * I, r) - mI * mI + eps)
    b = mp - a * mI
    return box(a, r) * I + box(b, r)


def wood_model(sm_rgb):
    h, w = sm_rgb.shape[:2]
    lab = cv2.cvtColor(sm_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(sm_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    b = 12
    bm = np.zeros((h, w), bool); bm[:b] = 1; bm[-b:] = 1; bm[:, :b] = 1; bm[:, -b:] = 1
    Hh, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    woodish = (Hh >= 12) & (Hh <= 40) & (S >= 35) & (S <= 175) & (V >= 25) & (V <= 200)
    seed = bm & woodish
    X = lab[seed]
    mu = X.mean(0); C = np.cov(X.T) + np.eye(3) * 4
    return mu, np.linalg.inv(C), seed, len(X)


def mdist(rgb, mu, Ci):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    d = lab - mu
    return np.sqrt(np.einsum('...i,ij,...j->...', d, Ci, d))


def shadow_wood(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]; ca = (lab[..., 1] - 128) / (L + 10); cb = (lab[..., 2] - 128) / (L + 10)
    return (L > 18) & (L < 120) & (ca > -0.04) & (ca < 0.11) & (cb > 0.07) & (cb < 0.20)


def segment(im):
    H, W = im.shape[:2]
    s = 1000 / max(H, W)
    sm = cv2.resize(im, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    h, w = sm.shape[:2]
    mu, Ci, seed, nseed = wood_model(sm)
    md = mdist(sm, mu, Ci)
    mask = np.full((h, w), cv2.GC_PR_FGD, np.uint8)
    shw = shadow_wood(sm)
    mask[(md < 3.0) | shw] = cv2.GC_PR_BGD
    mask[seed & (md < 2.0)] = cv2.GC_BGD
    mask[int(h * .25):int(h * .45), int(w * .35):int(w * .65)] = cv2.GC_FGD
    bgd = np.zeros((1, 65)); fgd = np.zeros((1, 65))
    cv2.grabCut(sm, mask, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK)
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    shc = (shw & (fg == 1)).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(shc, 8)
    bgd_d = cv2.dilate((1 - fg).astype(np.uint8), np.ones((3, 3), np.uint8))
    for i in range(1, n):
        comp = lbl == i
        if (bgd_d[comp] > 0).any():
            fg[comp] = 0
    bgs = cv2.erode((1 - fg).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    md2 = np.full_like(md, 99.0)
    if bgs.sum() > 2000:
        lab = cv2.cvtColor(sm, cv2.COLOR_RGB2LAB).astype(np.float32)
        X2 = lab[bgs]; mu2 = X2.mean(0); C2 = np.cov(X2.T) + np.eye(3) * 4
        d2 = lab - mu2; md2 = np.sqrt(np.einsum('...i,ij,...j->...', d2, np.linalg.inv(C2), d2))
    woodlike = (((md < WOODMD) | (md2 < MD2) | shw) & (fg == 1)).astype(np.uint8)
    woodlike = cv2.morphologyEx(woodlike, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(woodlike, 8)
    for i in range(1, n):
        if st[i, 4] > 0.0003 * h * w:
            comp = lbl == i
            fg[cv2.dilate(comp.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = 0
    n, lbl, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    keep = np.zeros_like(fg)
    for i in range(1, n):
        if st[i, 4] > 0.01 * h * w:
            keep[lbl == i] = 1
    inv = 1 - keep
    n2, l2, st2, _ = cv2.connectedComponentsWithStats(inv, 8)
    for i in range(1, n2):
        x, y, ww, hh, a = st2[i]
        touches = x == 0 or y == 0 or x + ww == w or y + hh == h
        if not touches and a < 0.02 * h * w and ((md[l2 == i] < 2.6) | (md2[l2 == i] < 2.2) | shw[l2 == i]).mean() < 0.4:
            keep[l2 == i] = 1
    return keep, mu, Ci, nseed


def full_alpha(im, keep, mu, Ci):
    H, W = im.shape[:2]
    k = cv2.resize(keep.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    a = guided(gray, k, 10, 1e-3)
    a = np.clip((a - 0.5) * 3 + 0.5, 0, 1)
    band = cv2.dilate((np.abs(k - 0.5) < 0.499).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    md = mdist(im, mu, Ci)
    woodf = np.clip((md - 1.8) / 1.4, 0, 1)
    woodf[shadow_wood(im)] = 0
    a = np.where(band, a * woodf, a)
    a = cv2.erode(a, np.ones((5, 5), np.uint8))
    a = cv2.GaussianBlur(a, (0, 0), 1.2)
    return np.clip(a, 0, 1)


def white_ref(im, a):
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.float32)
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    std = np.sqrt(np.clip(box(g * g, 7) - box(g, 7) ** 2, 0, None))
    H, W = g.shape
    region = np.zeros_like(g, bool); region[int(H * .05):int(H * .6), int(W * .15):int(W * .85)] = 1
    cand = (a > 0.99) & region & (hsv[..., 1] < 50) & (std < 3.0)
    vals = g[cand]
    if len(vals) < 20000:
        return None, int(len(vals))
    thr = np.percentile(vals, 70)
    sel = cand & (g >= thr)
    if sel.sum() < 15000:
        return None, int(sel.sum())
    return im[sel].reshape(-1, 3).astype(np.float32).mean(0), int(sel.sum())


def correct(im, ref):
    x = im.astype(np.float32) / 255
    if ref is not None:
        r = ref / 255
        gains = np.clip(r.mean() / r, 0.94, 1.06)
        gains = gains * (0.94 / r.mean())
        gains = np.minimum(gains, 1.35)
        x = x * gains
    x = np.clip(x, 0, 1)
    x = x + 0.06 * (x - 0.5) * (1 - np.abs(2 * x - 1))
    l = x.mean(-1, keepdims=True)
    x = l + (x - l) * 1.10
    return np.clip(x, 0, 1)


def composite(fg, a, rug_rgb, shadow=1.0):
    H, W = a.shape
    cs = cv2.GaussianBlur(np.roll(np.roll(a, 6, 0), 4, 1), (0, 0), 7)
    amb = cv2.GaussianBlur(np.roll(a, 22, 0), (0, 0), 55)
    bg = rug_rgb * (1 - 0.45 * shadow * cs[..., None]) * (1 - 0.30 * shadow * amb[..., None])
    out = fg * a[..., None] + bg * (1 - a[..., None])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 1.414
    vig = 1 - 0.16 * np.clip(r, 0, 1) ** 2.2
    return np.clip(out * vig[..., None], 0, 1)


def add_cones(bg, a, cones, rng):
    """Pega piñas (RGBA float) sobre el tapete, en zonas libres de la prenda."""
    H, W = a.shape
    free = cv2.erode((a < 0.02).astype(np.uint8), np.ones((15, 15), np.uint8))
    placed = 0
    tries = 0
    target = 3
    out = bg.copy()
    while placed < target and tries < 400:
        tries += 1
        c = cones[rng.integers(len(cones))]
        s = rng.uniform(0.08, 0.10) * W / c.shape[1]
        cw, ch = int(c.shape[1] * s), int(c.shape[0] * s)
        if cw < 10 or ch < 10:
            continue
        cc = cv2.resize(c, (cw, ch), interpolation=cv2.INTER_AREA)
        if rng.random() < .5:
            cc = cc[:, ::-1]
        x = rng.integers(0, W - cw); y = rng.integers(int(H * .45), H - ch)
        if free[y:y + ch, x:x + cw].mean() < 0.999:
            continue
        al = cc[..., 3:4]
        sh = cv2.GaussianBlur(al[..., 0], (0, 0), max(2, cw * .05))
        oy, ox = int(ch * .06), int(cw * .04)
        reg = out[y:y + ch, x:x + cw]
        shadow = np.zeros_like(sh); shadow[oy:, ox:] = sh[:ch - oy, :cw - ox]
        reg *= (1 - 0.35 * shadow[..., None])
        reg[:] = cc[..., :3] * al + reg * (1 - al)
        free[max(0, y - ch // 2):y + ch + ch // 2, max(0, x - cw // 2):x + cw + cw // 2] = 0
        placed += 1
    return out, placed

