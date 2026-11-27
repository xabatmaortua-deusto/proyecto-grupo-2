#!/usr/bin/env python3
"""Guarda imágenes del streaming de la cámara USB para procesarlas después.

Las imágenes se guardan SIN anotaciones (frame original, PNG sin pérdida) en ./capturas/

Uso:
  python3 capturar.py                    # ventana: ESPACIO/s guarda, a = auto on/off, q/ESC sale
  python3 capturar.py --interval 2       # además guarda automáticamente cada 2 s
  python3 capturar.py --no-gui --interval 1 --count 20   # sin ventana: 20 fotos, 1 por segundo
"""
import argparse
import os
import sys
import time

import cv2

import medir_arucos as m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cam", default=m.DEFAULT_CAM, help=f"índice o ruta de la cámara (por defecto {m.DEFAULT_CAM})")
    p.add_argument("--list-cams", action="store_true", help="listar las cámaras disponibles y salir")
    p.add_argument("--out", default="capturas", help="carpeta de salida")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--interval", type=float, default=0, help="guardado automático cada N s (0 = off)")
    p.add_argument("--count", type=int, default=0, help="parar tras N imágenes (0 = sin límite)")
    p.add_argument("--fmt", default="png", choices=["png", "jpg"])
    p.add_argument("--no-gui", action="store_true")
    args = p.parse_args()

    if args.list_cams:
        m.list_cameras()
        return

    if args.no_gui and args.interval <= 0:
        sys.exit("Sin ventana hace falta --interval para guardar algo.")
    os.makedirs(args.out, exist_ok=True)

    cap = m.open_camera(args.cam, args.width, args.height)
    if cap is None:
        sys.exit(f"No se pudo abrir la cámara {args.cam}. Prueba --list-cams para ver cuáles hay.")
    print(f"Cámara {args.cam}: {int(cap.get(3))}x{int(cap.get(4))} -> {os.path.abspath(args.out)}", flush=True)

    # detector sólo para informar en la previsualización (no se dibuja en la imagen guardada)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))

    saved, auto, last = 0, args.interval > 0, 0.0

    def save(frame):
        nonlocal saved
        ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        fn = os.path.join(args.out, f"img_{ts}.{args.fmt}")
        cv2.imwrite(fn, frame)
        saved += 1
        print(f"[{saved}] {fn}", flush=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            now = time.time()
            if auto and now - last >= args.interval:
                save(frame)
                last = now

            if not args.no_gui:
                _, ids, _ = det.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                n = 0 if ids is None else len(ids)
                vis = frame.copy()
                txt = f"marcadores: {n}  guardadas: {saved}  auto: {'ON' if auto else 'off'}"
                cv2.putText(vis, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(vis, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(vis, "ESPACIO/s guardar · a auto · q salir", (10, vis.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Captura", vis)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k in (ord(" "), ord("s")):
                    save(frame)
                if k == ord("a"):
                    auto = not auto
                    if auto and args.interval <= 0:
                        args.interval = 1.0
                    last = 0.0

            if args.count and saved >= args.count:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Total guardadas: {saved}")


if __name__ == "__main__":
    main()
