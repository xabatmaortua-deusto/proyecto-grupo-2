#!/usr/bin/env python3
"""Calibración ligera de la cámara a partir de capturas de la hoja de marcadores.

Estima y guarda en calibracion_camara.json:
  * k1, k2: distorsión radial de la lente. NO usa las distancias reales: busca la distorsión
    que hace que todas las esquinas encajen en un plano (error de ajuste mínimo).
  * edge_offset_px: cuánto hay que desplazar los bordes hacia fuera para compensar que la
    sobreexposición/desenfoque "encoge" el negro. Usa el layout conocido de la hoja
    (arucos_calibracion_A4.json), así que las capturas deben ser de la hoja impresa.

Uso:
  python3 capturar.py               # captura la hoja desde varios ángulos y distancias
  python3 calibrar.py capturas/     # calibra y guarda calibracion_camara.json

Recalibra si cambias de cámara, de resolución, de enfoque o de iluminación de forma notable.
"""
import argparse
import glob
import itertools
import json
import os

import cv2
import numpy as np
from scipy.optimize import minimize, minimize_scalar

import medir_arucos as m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", nargs="?", default=os.path.join(m.HERE, "capturas"))
    p.add_argument("--sheet", default=os.path.join(m.HERE, "arucos_calibracion_A4.json"))
    p.add_argument("--out", default=m.DEFAULT_CAL)
    p.add_argument("--min-markers", type=int, default=3)
    p.add_argument("--min-sharp", type=float, default=6.0)
    p.add_argument("--no-offset", action="store_true", help="no estimar el offset de borde")
    args = p.parse_args()

    with open(args.sheet) as f:
        sheet = json.load(f)
    size = sheet["marker_size_m"]
    centers_true = {int(k): np.array(v["center_m"]) for k, v in sheet["markers"].items()}
    ids_wanted = sorted(centers_true)
    true_d = {(a, b): np.linalg.norm(centers_true[a] - centers_true[b]) * 100
              for a, b in itertools.combinations(ids_wanted, 2)}

    detector = m.make_detector(sheet.get("dictionary", "DICT_4X4_50"))
    exts = (".png", ".jpg", ".jpeg")          # vale en Linux, Windows y macOS, en mayúsculas o minúsculas
    files = sorted(f for f in glob.glob(os.path.join(args.folder, "*")) if f.lower().endswith(exts))
    frames, shape = [], None
    for f in files:
        gray = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        if m.sharpness(gray) < args.min_sharp:
            continue
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            continue
        raw = {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.ravel()) if int(i) in ids_wanted}
        if len(raw) < args.min_markers:
            continue
        frames.append({"file": f, "gf": cv2.GaussianBlur(gray, (0, 0), 1.0).astype(np.float32), "raw": raw})
        shape = gray.shape
    if len(frames) < 3:
        raise SystemExit(f"Sólo {len(frames)} imágenes útiles (≥{args.min_markers} marcadores, nítidas). Captura más.")
    print(f"{len(frames)} imágenes útiles de {len(files)} ({shape[1]}x{shape[0]})")

    def refined(offset):
        out = []
        for fr in frames:
            d = {i: q for i, q0 in fr["raw"].items() if (q := m.edge_refine(fr["gf"], q0, offset)) is not None}
            if len(d) >= 2:
                out.append(d)
        return out

    def evaluate(lens, dets):
        rms2, errs = [], []
        for d in dets:
            du = {i: lens.undistort(q, shape) for i, q in d.items()}
            cen, rms = m.solve_plane(du, size)
            rms2.append(rms ** 2)
            errs += [(np.linalg.norm(cen[a] - cen[b]) * 100 / true_d[(a, b)] - 1) * 100
                     for a, b in true_d if a in cen and b in cen]
        return float(np.sqrt(np.mean(rms2))), np.array(errs)

    def report(name, lens, dets):
        rms, e = evaluate(lens, dets)
        print(f"  {name:28s} ajuste {rms:.2f} px · error distancias: sesgo {e.mean():+.2f} %  "
              f"desv {e.std():.2f} %  máx {np.abs(e).max():.2f} %")

    dets0 = refined(0.0)
    base = m.LensModel(width=shape[1], height=shape[0])
    print("Resultados (error relativo frente a las distancias reales de la hoja):")
    report("sin corrección", base, dets0)

    # 1) distorsión radial: minimizar el error de ajuste al plano (no usa distancias reales)
    res = minimize(lambda k: evaluate(m.LensModel(k[0], k[1]), dets0)[0], [0.0, 0.0],
                   method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-5})
    lens = m.LensModel(float(res.x[0]), float(res.x[1]), 0.0, shape[1], shape[0])
    report(f"+ distorsión k1={lens.k1:.3f} k2={lens.k2:.3f}", lens, dets0)

    # 2) offset de borde: sesgo de escala nulo frente al layout conocido
    if not args.no_offset:
        def bias2(off):
            lens.edge_offset_px = off
            return evaluate(lens, refined(off))[1].mean() ** 2
        r = minimize_scalar(bias2, bounds=(-3, 3), method="bounded", options={"xatol": 0.02})
        lens.edge_offset_px = float(r.x)
        report(f"+ offset borde {lens.edge_offset_px:+.2f} px", lens, refined(lens.edge_offset_px))

    lens.save(args.out, {"n_images": len(frames), "sheet": os.path.basename(args.sheet)})
    print(f"Guardado {args.out}")


if __name__ == "__main__":
    main()
