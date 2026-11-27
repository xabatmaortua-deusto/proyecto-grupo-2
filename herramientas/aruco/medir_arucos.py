#!/usr/bin/env python3
"""Detecta los 4 marcadores ArUco (DICT_4X4_50, IDs 0-3) en el vídeo de una cámara USB
y calcula en cada frame la distancia en cm entre los centros de cada par de marcadores.

No hace falta la matriz de la cámara: la escala sale del tamaño conocido de los marcadores.

Pipeline por frame:
  1. Detección ArUco.
  2. Refinado sub-píxel por bordes: en cada lado del marcador se buscan los puntos al 50 %
     entre el negro y el blanco locales, se ajusta una recta y las esquinas son las
     intersecciones. Mucho más estable que cornerSubPix con imágenes borrosas.
  3. Corrección de la distorsión radial de la lente y del desplazamiento del borde por
     sobreexposición (parámetros de calibracion_camara.json, generados con calibrar.py).
  4. Modo plano (por defecto): una única homografía plano->imagen + posición/giro de cada
     marcador en el plano, ajustando todas las esquinas a la vez. Da igual la perspectiva
     y la focal. Supone marcadores coplanares.
     Modo 3D (--calib K/dist de OpenCV): solvePnP por marcador, sirve aunque no sean coplanares.
  5. Control de calidad: los frames movidos o con mal ajuste no actualizan el suavizado.

Uso:
  python3 medir_arucos.py                      # cámara USB (/dev/video2 en Linux, 0 en Windows/macOS)
  python3 medir_arucos.py --list-cams          # ver qué índices de cámara funcionan
  python3 medir_arucos.py --image foto.png     # una imagen
  python3 medir_arucos.py --no-cal             # sin aplicar calibracion_camara.json

Teclas: q / ESC salir · s guardar captura
"""
import argparse
import itertools
import json
import os
import sys
import time

import cv2
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CAL = os.path.join(HERE, "calibracion_camara.json")
# En el PC Linux del puesto la cámara USB es /dev/video2; en Windows y macOS suele ser la 0 o la 1.
DEFAULT_CAM = "2" if sys.platform.startswith("linux") else "0"


def _backends():
    """Backends de vídeo de OpenCV a probar, según el sistema operativo."""
    if sys.platform.startswith("linux"):
        return [cv2.CAP_V4L2, cv2.CAP_ANY]
    if sys.platform == "win32":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def open_camera(cam, width=1280, height=720):
    """Abre la cámara en Linux, Windows o macOS. Devuelve None si no hay imagen."""
    src = int(cam) if str(cam).isdigit() else cam
    for backend in _backends():
        cap = cv2.VideoCapture(src, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, _ = cap.read()          # algunos backends abren pero no dan imagen
            if ok:
                return cap
        cap.release()
    return None


def list_cameras(max_index=6):
    """Prueba los índices 0..max_index-1 e imprime los que dan imagen."""
    found = []
    for i in range(max_index):
        cap = open_camera(i, 640, 480)
        if cap is not None:
            found.append(i)
            print(f"  cámara {i}: {int(cap.get(3))}x{int(cap.get(4))}")
            cap.release()
    if not found:
        print("  no se ha encontrado ninguna cámara")
    return found


# ---------------------------------------------------------------- geometría básica
def local_corners(s):
    """Esquinas de un marcador de lado s centrado en 0, orden OpenCV: TL, TR, BR, BL."""
    h = s / 2
    return np.array([[-h, -h], [h, -h], [h, h], [-h, h]], dtype=np.float64)


def rigid_fit(src, dst):
    """Ajuste rígido 2D (rotación + traslación) src -> dst. Devuelve (theta, t)."""
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    r = (u @ vt).T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = (u @ vt).T
    return np.arctan2(r[1, 0], r[0, 0]), cd - r @ cs


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def project(G, pts):
    p = np.c_[pts, np.ones(len(pts))] @ G.T
    return p[:, :2] / p[:, 2:3]


# ---------------------------------------------------------------- calibración ligera
class LensModel:
    """Distorsión radial polinómica (k1, k2) en coordenadas normalizadas por la semidiagonal
    + desplazamiento del borde (px) hacia fuera para compensar la sobreexposición."""

    def __init__(self, k1=0.0, k2=0.0, edge_offset_px=0.0, width=None, height=None):
        self.k1, self.k2, self.edge_offset_px = k1, k2, edge_offset_px
        self.width, self.height = width, height

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d.get("k1", 0), d.get("k2", 0), d.get("edge_offset_px", 0),
                   d.get("width"), d.get("height"))

    def save(self, path, extra=None):
        d = {"k1": self.k1, "k2": self.k2, "edge_offset_px": self.edge_offset_px,
             "width": self.width, "height": self.height}
        d.update(extra or {})
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    def offset_for(self, shape):
        """El desplazamiento está en px a la resolución calibrada; se escala si cambia."""
        if self.width:
            return self.edge_offset_px * shape[1] / self.width
        return self.edge_offset_px

    def undistort(self, pts, shape):
        if self.k1 == 0 and self.k2 == 0:
            return pts
        h, w = shape[:2]
        c = np.array([w / 2, h / 2])
        r = np.hypot(w, h) / 2
        v = (pts - c) / r
        r2 = (v ** 2).sum(-1, keepdims=True)
        return c + r * v * (1 + self.k1 * r2 + self.k2 * r2 ** 2)


# ---------------------------------------------------------------- refinado de esquinas
def edge_refine(gray_f, q, offset_px=0.0, n_mod=6, iters=2, n_samples=40):
    """Refina las 4 esquinas de un marcador ajustando rectas a sus bordes exteriores.
    gray_f: imagen en float32 (ya suavizada). Devuelve (4,2) o None si falla."""
    q = q.astype(np.float64).copy()
    ts = np.linspace(0.12, 0.88, n_samples)
    for it in range(iters):
        c = q.mean(0)
        lines = []
        for k in range(4):
            a, b = q[k], q[(k + 1) % 4]
            L = np.linalg.norm(b - a)
            if L < 12:
                return None
            d = (b - a) / L
            nrm = np.array([d[1], -d[0]])
            if np.dot(nrm, (a + b) / 2 - c) < 0:
                nrm = -nrm                                   # normal hacia fuera
            w = 0.6 * L / n_mod                              # < 1 módulo: dentro sólo hay borde negro
            ss = np.linspace(-w, w, max(9, int(2 * w) + 1))
            base = a[None, :] + (ts[:, None] * L) * d        # puntos sobre el borde
            P = base[:, None, :] + ss[None, :, None] * nrm
            prof = cv2.remap(gray_f, P[..., 0].astype(np.float32), P[..., 1].astype(np.float32),
                             cv2.INTER_LINEAR)
            nin = max(2, len(ss) // 5)
            lo = np.median(prof[:, :nin], axis=1)
            hi = np.median(prof[:, -nin:], axis=1)
            pts = []
            for i_t in range(len(ts)):
                if hi[i_t] - lo[i_t] < 20:                   # sin contraste (oclusión, reflejo)
                    continue
                pr, mid = prof[i_t], (lo[i_t] + hi[i_t]) / 2
                idx = np.where((pr[:-1] < mid) & (pr[1:] >= mid))[0]
                if len(idx) == 0:
                    continue
                j = idx[np.argmin(np.abs(idx - len(ss) // 2))]
                s = ss[j] + (mid - pr[j]) / (pr[j + 1] - pr[j]) * (ss[j + 1] - ss[j])
                pts.append(base[i_t] + s * nrm)
            if len(pts) < n_samples // 4:
                return None
            vx, vy, x0, y0 = cv2.fitLine(np.float32(pts), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            p0 = np.array([x0, y0], dtype=np.float64)
            if it == iters - 1:
                p0 = p0 + offset_px * nrm                    # compensación del borde
            lines.append((p0, np.array([vx, vy], dtype=np.float64)))
        new = []
        for k in range(4):                                   # esquina k = borde k-1 ∩ borde k
            (p1, v1), (p2, v2) = lines[(k - 1) % 4], lines[k]
            A = np.array([v1, -v2]).T
            if abs(np.linalg.det(A)) < 1e-6:
                return None
            t = np.linalg.solve(A, p2 - p1)
            new.append(p1 + t[0] * v1)
        new = np.array(new)
        if np.abs(new - q).max() > 0.25 * L:                 # salto absurdo -> descartar
            return None
        q = new
    return q


# ---------------------------------------------------------------- solvers
def solve_plane(det, size_m):
    """det: {id: (4,2) esquinas (sin distorsión) en px}.
    Devuelve ({id: centro (x,y) en m en el plano}, error RMS de reproyección en px)."""
    ids = sorted(det)
    loc = local_corners(size_m)
    ref = max(ids, key=lambda i: cv2.contourArea(det[i].astype(np.float32)))
    H_ref, _ = cv2.findHomography(det[ref], loc)
    G0 = np.linalg.inv(H_ref)
    G0 /= G0[2, 2]

    others = [i for i in ids if i != ref]
    poses0 = []
    for i in others:
        pp = cv2.perspectiveTransform(det[i].reshape(-1, 1, 2), H_ref).reshape(-1, 2)
        th, t = rigid_fit(loc, pp)
        poses0 += [th, t[0], t[1]]
    x0 = np.r_[G0.ravel()[:8], poses0]
    obs = {i: det[i].astype(np.float64) for i in ids}

    def unpack(x):
        G = np.r_[x[:8], 1.0].reshape(3, 3)
        poses = {ref: (0.0, np.zeros(2))}
        for k, i in enumerate(others):
            th, tx, ty = x[8 + 3 * k: 11 + 3 * k]
            poses[i] = (th, np.array([tx, ty]))
        return G, poses

    def residuals(x):
        G, poses = unpack(x)
        return np.concatenate([(project(G, loc @ rot(th).T + t) - obs[i]).ravel()
                               for i, (th, t) in ((i, poses[i]) for i in ids)])

    x = least_squares(residuals, x0, method="lm").x if others else x0
    _, poses = unpack(x)
    rms = float(np.sqrt(np.mean(residuals(x) ** 2)))
    return {i: poses[i][1] for i in ids}, rms


def solve_3d(det, size_m, K, dist):
    h = size_m / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
    out = {}
    for i, c in det.items():
        ok, _, tvec = cv2.solvePnP(obj, c.astype(np.float64), K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            out[i] = tvec.ravel()
    return out


def load_calib(path):
    if path.endswith(".npz"):
        d = np.load(path)
        K = d["K"] if "K" in d else d["camera_matrix"]
        dist = d["dist"] if "dist" in d else d["dist_coeffs"]
    else:
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        K = fs.getNode("camera_matrix").mat()
        dist = fs.getNode("distortion_coefficients").mat()
    return K.astype(np.float64), dist.astype(np.float64)


# ---------------------------------------------------------------- detección completa
def make_detector(dict_name="DICT_4X4_50"):
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE   # refinamos nosotros
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name)), p)


def detect(gray, detector, ids_wanted, lens=None, refine=True):
    """Devuelve (det_pix, det_und, corners_raw, ids_raw):
    det_pix = esquinas refinadas en la imagen (para dibujar),
    det_und = las mismas corregidas de distorsión (para medir)."""
    corners, ids, _ = detector.detectMarkers(gray)
    det_pix = {}
    if ids is not None:
        gf = cv2.GaussianBlur(gray, (0, 0), 1.0).astype(np.float32) if refine else None
        off = lens.offset_for(gray.shape) if lens else 0.0
        for c, i in zip(corners, ids.ravel()):
            i = int(i)
            if i not in ids_wanted:
                continue
            q = c.reshape(4, 2)
            if refine:
                q = edge_refine(gf, q, off)
                if q is None:
                    continue
            det_pix[i] = q
    det_und = {i: (lens.undistort(q, gray.shape) if lens else q) for i, q in det_pix.items()}
    return det_pix, det_und, corners, ids


def sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ---------------------------------------------------------------- vídeo / dibujo
class Smoother:
    """EMA por par de marcadores; sólo se actualiza con frames de buena calidad."""

    def __init__(self, alpha, max_age=15):
        self.alpha, self.max_age, self.v, self.age = alpha, max_age, {}, {}

    def update(self, key, value):
        if self.alpha >= 1 or key not in self.v:
            self.v[key] = value
        else:
            self.v[key] = self.alpha * value + (1 - self.alpha) * self.v[key]
        self.age[key] = 0

    def tick(self):
        for k in list(self.age):
            self.age[k] += 1
            if self.age[k] > self.max_age:           # par perdido hace tiempo
                del self.age[k], self.v[k]

    def get(self, key):
        return self.v.get(key)


def put(img, txt, org, color=(0, 255, 0), scale=0.6):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def process(frame, detector, args, calib, lens, smooth):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    det_pix, det_und, corners, ids = detect(gray, detector, args.ids, lens, not args.no_refine)
    sharp = sharpness(gray)

    raw, info, good = {}, "", False
    if len(det_und) >= 2:
        if calib is not None:
            centers, rms = solve_3d(det_und, args.size, *calib), 0.0
            info = "modo 3D"
        else:
            centers, rms = solve_plane(det_und, args.size)
            info = f"ajuste {rms:.2f} px"
        good = rms <= args.max_rms and sharp >= args.min_sharp
        info += f" · nitidez {sharp:.0f}" + ("" if good else "  [BAJA CALIDAD: no se usa]")
        for a, b in itertools.combinations(sorted(centers), 2):
            raw[(a, b)] = float(np.linalg.norm(centers[a] - centers[b])) * 100
            if good:
                smooth.update((a, b), raw[(a, b)])
    smooth.tick()

    # valores mostrados: suavizados si existen, si no el bruto
    shown = {k: (smooth.get(k) if smooth.get(k) is not None else v) for k, v in raw.items()}

    vis = frame.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    for i, q in det_pix.items():
        cv2.polylines(vis, [np.int32(np.round(q))], True, (255, 0, 255), 1, cv2.LINE_AA)
    pix_c = {i: q.mean(0) for i, q in det_pix.items()}
    col = (0, 255, 255) if good else (0, 140, 255)
    for (a, b), d in shown.items():
        pa, pb = pix_c[a], pix_c[b]
        cv2.line(vis, tuple(map(int, pa)), tuple(map(int, pb)), col, 2)
        put(vis, f"{d:.1f} cm", tuple(((pa + pb) / 2).astype(int)), col)

    status = f"Detectados {len(det_pix)}/{len(args.ids)}: {sorted(det_pix)}"
    if len(det_pix) < 2:
        status += "  (faltan marcadores para medir)"
    y = 25
    for line in [status, info] + [f"{a}-{b}: {d:6.2f} cm" for (a, b), d in shown.items()]:
        if line:
            put(vis, line, (10, y))
            y += 24
    return vis, det_pix, shown, good


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cam", default=DEFAULT_CAM, help=f"índice o ruta de la cámara (por defecto {DEFAULT_CAM})")
    p.add_argument("--list-cams", action="store_true", help="listar las cámaras disponibles y salir")
    p.add_argument("--image", help="procesar una imagen en vez de la cámara")
    p.add_argument("--size", type=float, default=0.08, help="lado del marcador en metros (0.08)")
    p.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--dict", default="DICT_4X4_50")
    p.add_argument("--cal", default=DEFAULT_CAL, help="calibración ligera (calibrar.py)")
    p.add_argument("--no-cal", action="store_true", help="no aplicar distorsión/offset de borde")
    p.add_argument("--no-refine", action="store_true", help="desactivar refinado por bordes")
    p.add_argument("--calib", help="npz (K, dist) o yaml de OpenCV -> modo 3D")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--smooth", type=float, default=0.3, help="alpha EMA (1 = sin suavizado)")
    p.add_argument("--max-rms", type=float, default=1.5, help="error de ajuste máximo (px) para aceptar el frame")
    p.add_argument("--min-sharp", type=float, default=6.0, help="nitidez mínima (var. Laplaciano)")
    p.add_argument("--no-gui", action="store_true", help="sin ventana, sólo consola")
    args = p.parse_args()

    if args.list_cams:
        list_cameras()
        return

    detector = make_detector(args.dict)
    calib = load_calib(args.calib) if args.calib else None
    lens = None
    if not args.no_cal and os.path.exists(args.cal):
        lens = LensModel.load(args.cal)
        print(f"Calibración: k1={lens.k1:.4f} k2={lens.k2:.4f} borde={lens.edge_offset_px:.2f}px ({args.cal})")
    else:
        print("Sin calibración de lente (ejecuta calibrar.py para mejorar la precisión)")

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"No se pudo leer {args.image}")
        vis, det, dists, good = process(frame, detector, args, calib, lens, Smoother(1.0))
        print(f"marcadores {sorted(det)}  calidad {'OK' if good else 'BAJA'}")
        for (a, b), d in dists.items():
            print(f"{a}-{b}: {d:.2f} cm")
        if not args.no_gui:
            cv2.imshow("ArUco", vis)
            cv2.waitKey(0)
        return

    cap = open_camera(args.cam, args.width, args.height)
    if cap is None:
        sys.exit(f"No se pudo abrir la cámara {args.cam}. Prueba --list-cams para ver cuáles hay.")
    w, h = int(cap.get(3)), int(cap.get(4))
    print(f"Cámara {args.cam}: {w}x{h}", flush=True)
    if lens and lens.width and (lens.width, lens.height) != (w, h):
        print(f"Aviso: calibración hecha a {lens.width}x{lens.height}; se adapta a {w}x{h}")

    smooth = Smoother(args.smooth)
    t_last, fps, n = time.time(), 0.0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame perdido")
                time.sleep(0.05)
                continue
            vis, det, dists, good = process(frame, detector, args, calib, lens, smooth)
            n += 1
            now = time.time()
            if now - t_last >= 1.0:
                fps, n, t_last = n / (now - t_last), 0, now
                txt = " | ".join(f"{a}-{b} {d:.1f}cm" for (a, b), d in dists.items())
                print(f"[{fps:4.1f} fps] marcadores {sorted(det)} {'' if good or not det else '(baja calidad)'} {txt}",
                      flush=True)
            put(vis, f"{fps:.1f} fps", (vis.shape[1] - 110, 25), (255, 255, 255))
            if not args.no_gui:
                cv2.imshow("ArUco", vis)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k == ord("s"):
                    fn = time.strftime("captura_%Y%m%d_%H%M%S.png")
                    cv2.imwrite(fn, vis)
                    print("Guardado", fn)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
