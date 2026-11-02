"""Núcleo del servicio (FOTO-05): foto sobre tablas -> solo alfombra crema ajustada, o tapete navideño;
foto con modelo -> la original, en vertical 3:4 si viene horizontal."""
import os, io, glob, numpy as np, cv2
from PIL import Image, ImageOps
import pillow_heif; pillow_heif.register_heif_opener()
HERE = os.path.dirname(os.path.abspath(__file__))
RUG = os.environ.get('RUG', os.path.join(HERE, 'rug_full.npy'))
exec(open(os.path.join(HERE, 'core_base.py'), encoding='utf-8').read())
TARGET = np.array([234, 214, 198], np.float32)          # crema ajustada a la paleta (#EAD6C6 aprox.)
RATIO = TARGET / np.array(COLORS['crema'], np.float32)
_SHADE = None
_CONES = None

def _shade():
    global _SHADE
    if _SHADE is None:
        _SHADE = np.load(RUG).astype(np.float32)
    return _SHADE

def _cones():
    global _CONES
    if _CONES is None:
        import base64, conos
        _CONES = [cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(b), np.uint8), cv2.IMREAD_UNCHANGED),
                               cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255 for b in conos.CONOS]
    return _CONES

def load_bytes(b):
    return np.array(ImageOps.exif_transpose(Image.open(io.BytesIO(b))).convert('RGB'))

# ---------- FOTO-05: personas (foto con modelo) ----------
_FACE = None
_HOG = None

def _detectores():
    global _FACE, _HOG
    if _FACE is None:
        _FACE = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
        _HOG = cv2.HOGDescriptor(); _HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return _FACE, _HOG

def _piel(rgb):
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    return (ycc[..., 1] >= 135) & (ycc[..., 1] <= 175) & (ycc[..., 2] >= 85) & (ycc[..., 2] <= 130) & (ycc[..., 0] > 50)

def persona(im):
    """Devuelve ('si' | 'duda' | 'no', detalle). 'si' = rostro real (con piel); 'duda' = silueta sin rostro claro."""
    H, W = im.shape[:2]
    s = 800 / max(H, W)
    sm = cv2.resize(im, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    h, w = sm.shape[:2]
    face, hog = _detectores()
    g = cv2.equalizeHist(cv2.cvtColor(sm, cv2.COLOR_RGB2GRAY))
    mn = int(min(h, w) * 0.05)
    caras = face.detectMultiScale(g, scaleFactor=1.1, minNeighbors=6, minSize=(mn, mn))
    piel = _piel(sm)
    reales = []
    for (x, y, fw, fh) in caras:
        cx0, cy0 = x + fw // 5, y + fh // 5
        frac = float(piel[cy0:y + fh - fh // 5, cx0:x + fw - fw // 5].mean())
        if frac >= 0.40:
            reales.append(round(frac, 2))
    if reales:
        return 'si', f'rostro con piel ({reales[0]:.0%})'
    rects, pesos = hog.detectMultiScale(sm, winStride=(8, 8), padding=(8, 8), scale=1.05)
    altos = [float(p) for (r, p) in zip(rects, np.ravel(pesos) if len(rects) else []) if r[3] > 0.45 * h and p > 0.8]
    if altos and float(piel.mean()) > 0.03:
        return 'duda', 'posible persona de cuerpo entero'
    return 'no', ''

def vertical(rgb, ratio=0.75):
    """Lleva la foto a marco vertical 3:4 sin recortar: agrega franjas arriba/abajo y las rellena
    prolongando el borde de la propia foto, desenfocado y fundido con ella."""
    H, W = rgb.shape[:2]
    if W / H <= ratio + 0.01:
        return rgb, False
    H2 = int(round(W / ratio))
    top = (H2 - H) // 2; bot = H2 - H - top
    # prolonga el borde (sin reflejar, para no duplicar a la persona) y lo desenfoca hacia afuera
    ref = cv2.copyMakeBorder(rgb, top, bot, 0, 0, cv2.BORDER_REPLICATE)
    k = max(9, int(W * 0.02))
    blur = cv2.GaussianBlur(cv2.GaussianBlur(ref, (0, 0), k), (0, 0), k)
    m = np.zeros((H2, 1), np.float32); m[top:top + H] = 1
    fe = max(4, int(H * 0.03))
    m = cv2.GaussianBlur(m, (1, 4 * fe + 1), fe)
    m[top + 2 * fe:top + H - 2 * fe] = 1
    m = m[..., None]
    # lejos del borde, el relleno pasa a la misma foto ampliada y muy desenfocada (sin rayas)
    cov = cv2.resize(rgb, (int(round(W * H2 / H)), H2), interpolation=cv2.INTER_AREA)
    cx = (cov.shape[1] - W) // 2
    cov = cv2.GaussianBlur(cov[:, cx:cx + W], (0, 0), max(20, W * 0.05)).astype(np.float32)
    yy = np.arange(H2, dtype=np.float32)[:, None]
    dist = np.where(yy < top, top - yy, np.where(yy >= top + H, yy - (top + H - 1), 0))
    t = np.clip(dist / max(1, 0.6 * max(top, bot)), 0, 1)[..., None]
    fill = blur.astype(np.float32) * (1 - t) + cov * t
    grano = np.random.default_rng(0).normal(0, 2.0, (H2, W, 1)).astype(np.float32) * (1 - m)
    out = ref.astype(np.float32) * m + fill * (1 - m) + grano
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), True

def render(im, navidad=False, seed=0):
    """FOTO-05. Devuelve (fotos: dict nombre->RGB uint8, calidad, revisar, motivo).
    - Foto con modelo: la original (en vertical 3:4 si viene horizontal), sin alfombra.
    - Navidad: solo el tapete blanco con piñas.
    - Resto: solo la alfombra crema ajustada. Ya no se genera la azul."""
    H, W = im.shape[:2]
    calidad = {'ancho': W, 'alto': H}
    hay, det = persona(im)
    calidad['persona'] = hay
    if hay == 'si':
        foto, giro = vertical(im)
        calidad['modelo'] = det; calidad['reencuadre_3x4'] = giro
        motivos = ['Foto de baja resolución'] if min(H, W) < 700 else []
        return {'principal': foto}, calidad, bool(motivos), '; '.join(motivos)
    try:
        keep, mu, Ci, nseed = segment(im)
    except Exception as e:  # sin madera en el borde, el modelo no se puede estimar
        calidad['error'] = str(e)[:120]
        return {}, calidad, True, 'No se encontró piso de madera en la foto'
    a = full_alpha(im, keep, mu, Ci)
    ref, nref = white_ref(im, a)
    fg = correct(im, ref)
    shade = _shade()
    rng = np.random.default_rng(seed)
    scl = max(W / 3024, H / 4032)
    if abs(scl - 1) > 0.001:
        shade = cv2.resize(shade, (max(W, int(shade.shape[1] * scl)), max(H, int(shade.shape[0] * scl))), interpolation=cv2.INTER_AREA)
    y0 = rng.integers(0, shade.shape[0] - H + 1); x0 = rng.integers(0, shade.shape[1] - W + 1)
    sh = shade[y0:y0 + H, x0:x0 + W]
    if rng.random() < 0.5:
        sh = sh[:, ::-1]
    to8 = lambda x: (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
    if navidad:
        rug = colorize(sh, WHITE, floor=0.72)
        rug, ncones = add_cones(rug, a, _cones(), rng)
        fotos = {'principal': to8(composite(fg, a, rug, shadow=0.8))}
    else:
        # crema ajustada a la paleta: el ajuste de tono se aplica solo a la alfombra, la prenda no se toca
        rug = np.clip(colorize(sh, COLORS['crema']) * RATIO, 0, 1)
        fotos = {'principal': to8(composite(fg, a, rug))}
    # control de calidad (umbrales a escala 1000 px)
    fgf = float(a.mean())
    s = 1000 / max(H, W)
    calidad.update({'madera_borde': int(nseed), 'prenda': round(fgf, 3), 'ref_blanco': ref is not None})
    motivos = []
    if hay == 'duda':
        motivos.append('Posible persona en la foto (revisar si es foto con modelo)')
    hs, ws = int(H * s), int(W * s)
    borde = 2 * 12 * ws + 2 * 12 * (hs - 24)
    calidad['madera_borde_pct'] = round(nseed / max(1, borde), 3)
    if nseed / max(1, borde) < 0.25:
        motivos.append('Poco piso de madera en el borde (¿la foto ya tiene otro fondo?)')
    if not (0.35 <= fgf <= 0.97):
        motivos.append(f'Recorte dudoso (prenda {fgf:.0%} de la foto)')
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lisas = ((g.std(1) < 12) & (g.mean(1) > 225)).mean()   # filas planas y claras (barras de una captura)
    calidad['filas_planas'] = round(float(lisas), 3)
    if lisas > 0.08:
        motivos.append('Parece una captura de pantalla')
    if min(H, W) < 700:
        motivos.append('Foto de baja resolución')
    return fotos, calidad, bool(motivos), '; '.join(motivos)

def webp(rgb, size, q):
    im = Image.fromarray(rgb); im.thumbnail(size, Image.LANCZOS)
    b = io.BytesIO(); im.save(b, 'WEBP', quality=q, method=6); return b.getvalue()
