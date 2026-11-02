"""FOTO-04 · Procesa en GitHub Actions las fotos nuevas de Drive (La Boutique de Sofii).
Pide a la Edge Function 'fotos-alfombra' los diseños pendientes, genera la alfombra crema (principal) + azul
(o tapete navideño) y devuelve las fotos a la misma función, que las guarda en Supabase.
Solo usa el secreto SYNC_KEY; no tiene llaves de Supabase."""
import os, sys, time, json, base64, random, urllib.request, urllib.parse
import core

EDGE = 'https://tcxoumemwkvmscfuvfjw.supabase.co/functions/v1/fotos-alfombra'
HOSTS = ('tcxoumemwkvmscfuvfjw.supabase.co', 'drive.google.com', 'lh3.googleusercontent.com', 'drive.usercontent.google.com')
KEY = os.environ.get('SYNC_KEY', '')
MAX = 25 * 1024 * 1024
LIMITE = 20 * 60  # segundos por ejecución

def edge(body):
    req = urllib.request.Request(EDGE, data=json.dumps(body).encode(), method='POST',
                                 headers={'content-type': 'application/json', 'x-sync-key': KEY})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try: return json.loads(e.read())
        except Exception: return {'ok': False, 'error': 'HTTP %d' % e.code}

def descargar(u):
    if (urllib.parse.urlparse(u).hostname or '') not in HOSTS:
        raise ValueError('origen no permitido')
    with urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'alfombra'}), timeout=60) as r:
        raw = r.read(MAX + 1)
    if not raw or len(raw) > MAX:
        raise ValueError('imagen vacía o muy grande')
    return raw

def procesar(p):
    t0 = time.time()
    try:
        im = core.load_bytes(descargar(p['imagen']))
    except Exception as e:
        return edge({'accion': 'error', 'id': p['id'], 'foto_file_id': p['foto_file_id'], 'motivo': 'no se pudo leer la imagen: ' + str(e)[:100]})
    fotos, calidad, revisar, motivo = core.render(im, bool(p.get('navidad')), random.randint(0, 999))
    out = [{'tipo': t,
            'w': base64.b64encode(core.webp(fotos[t], (810, 1080), 66)).decode(),
            's': base64.b64encode(core.webp(fotos[t], (380, 507), 64)).decode()}
           for t in ('principal', 'crema', 'azul') if t in fotos]
    if not out:
        return edge({'accion': 'error', 'id': p['id'], 'foto_file_id': p['foto_file_id'], 'motivo': motivo or 'sin fotos'})
    r = edge({'accion': 'guardar', 'id': p['id'], 'foto_file_id': p['foto_file_id'], 'navidad': bool(p.get('navidad')),
              'revisar': revisar, 'motivo': motivo, 'calidad': calidad, 'fotos': out})
    r['segundos'] = round(time.time() - t0, 1)
    return r

def main():
    if not KEY:
        sys.exit('Falta el secreto SYNC_KEY')
    t0, vistos = time.time(), set()
    while time.time() - t0 < LIMITE:
        pend = edge({'accion': 'pendientes', 'n': 3})
        if not pend.get('ok'):
            sys.exit('pendientes: %s' % pend.get('error'))
        nuevos = [p for p in pend.get('pendientes') or [] if (p['id'], p['foto_file_id']) not in vistos]
        if not nuevos:
            break
        for p in nuevos:
            vistos.add((p['id'], p['foto_file_id']))
            r = procesar(p)
            print(p['id'], '→', r.get('estado') or ('error registrado' if r.get('ok') else r.get('error')), r.get('segundos', ''), flush=True)
    print('Listo: %d foto(s) en %d s' % (len(vistos), time.time() - t0))

if __name__ == '__main__':
    main()
