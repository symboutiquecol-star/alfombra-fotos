"""Núcleo del servicio: foto sobre tablas -> alfombra crema ajustada (principal) + azul, o tapete navideño."""
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

def render(im, navidad=False, seed=0):
    """Devuelve (fotos: dict nombre->RGB uint8, calidad: dict, revisar: bool, motivo: str)."""
    H, W = im.shape[:2]
    try:
        keep, mu, Ci, nseed = segment(im)
    except Exception as e:  # sin madera en el borde, el modelo no se puede estimar
        return {}, {'error': str(e)[:120]}, True, 'No se encontró piso de madera en la foto'
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
    fotos = {}
    if navidad:
        rug = colorize(sh, WHITE, floor=0.72)
        rug, ncones = add_cones(rug, a, _cones(), rng)
        fotos['principal'] = to8(composite(fg, a, rug, shadow=0.8))
    crema = composite(fg, a, colorize(sh, COLORS['crema']))
    azul = composite(fg, a, colorize(sh, COLORS['azul']))
    # crema ajustada: solo los píxeles de alfombra (donde azul y crema difieren)
    diff = np.abs(crema - azul).sum(-1) * 255
    bg = cv2.GaussianBlur(np.clip((diff - 20) / 60, 0, 1).astype(np.float32), (0, 0), 1.5)[..., None]
    crema_aj = crema * (1 - bg) + np.clip(crema * RATIO, 0, 1) * bg
    if navidad:
        fotos['crema'] = to8(crema_aj)
    else:
        fotos['principal'] = to8(crema_aj)
    fotos['azul'] = to8(azul)
    # control de calidad (umbrales a escala 1000 px)
    fgf = float(a.mean())
    s = 1000 / max(H, W)
    calidad = {'ancho': W, 'alto': H, 'madera_borde': int(nseed), 'prenda': round(fgf, 3), 'ref_blanco': ref is not None}
    motivos = []
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
