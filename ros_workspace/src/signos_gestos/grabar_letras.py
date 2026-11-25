#!/usr/bin/env python3
"""Graba ejemplos de cada letra para entrenar el clasificador (paso 1 de 3).

Cada grabación se guarda dos veces, con el mismo nombre <persona>_<fecha>:
  dataset_letras/<LETRA>/<id>.npz   los 21 puntos de la mano de cada fotograma (≈25 KB, va a git)
  datos/videos/<LETRA>/<id>.mp4     el vídeo LIMPIO (sin esqueleto ni espejo), para la vía clásica
                                    y para sacar los recortes de la red (extraer_recortes.py).
                                    datos/ no va a git: se comparte por Drive.
El fotograma n del vídeo corresponde a puntos[n] del .npz. Con --sin-video solo se guardan los puntos.

Con la webcam (recomendado):
  python grabar_letras.py --persona ainhoa                 # recorre todo el abecedario
  python grabar_letras.py --persona ainhoa --letras A B LL # solo esas letras
  python grabar_letras.py --persona ainhoa --camara 1      # otra cámara

  Teclas:  ESPACIO  grabar la letra actual (cuenta atrás de 1 s y graba --segundos)
           d / a    letra siguiente / anterior
           q / ESC  salir

  En cada grabación haz la letra y, si es una letra quieta, muévela un poco (más cerca, más
  lejos, algo girada). Si es una letra con movimiento, repite el movimiento varias veces.
  Graba también la clase «nada» (última de la lista): mano en reposo, bajando o cambiando de
  letra. Sirve para que el reconocedor diga «?» en vez de inventarse una letra.

Desde un vídeo ya grabado (todo el vídeo es de una misma letra):
  python grabar_letras.py --persona keltse --video A_keltse.mp4 --letra A

Objetivo orientativo: 3 o más grabaciones por letra y persona, con alguna luz distinta.
"""
import argparse
import os
import time
from datetime import datetime

import cv2
import numpy as np

from letras_comun import (AQUI, ALFABETO, CARPETA_DATOS, CARPETA_VIDEOS, NADA, VENTANA,
                          crear_detector, detectar, dibujar_mano, reducir, texto)


def nuevo_id(persona):
    """Nombre común de la grabación en dataset_letras/ y en datos/videos/."""
    return f"{persona}_{datetime.now():%Y%m%d-%H%M%S}"


class VideoLimpio:
    """Escribe los fotogramas tal cual llegan (sin dibujos ni espejo) en datos/videos/<LETRA>/<id>.mp4.

    Se abre con el primer fotograma (hasta entonces no se sabe el tamaño). Si la grabación no se
    guarda (poca mano), descartar() borra el vídeo para que no quede un vídeo sin sus puntos.
    """

    def __init__(self, letra, ident, fps, activo=True):
        self.ruta = os.path.join(CARPETA_VIDEOS, letra, f"{ident}.mp4") if activo else None
        self.fps, self.escritor = fps, None

    def escribir(self, frame):
        if self.ruta is None:
            return
        if self.escritor is None:
            os.makedirs(os.path.dirname(self.ruta), exist_ok=True)
            self.escritor = cv2.VideoWriter(self.ruta, cv2.VideoWriter_fourcc(*"mp4v"),
                                            self.fps, frame.shape[1::-1])
        self.escritor.write(frame)

    def cerrar(self):
        if self.escritor is not None:
            self.escritor.release()

    def descartar(self):
        self.cerrar()
        if self.ruta and os.path.exists(self.ruta):
            os.remove(self.ruta)
        self.ruta = None

    def relativa(self):
        """Ruta respecto a Proyecto/ y con «/», para guardarla en el .npz (vale en Windows y macOS)."""
        return os.path.relpath(self.ruta, AQUI).replace(os.sep, "/") if self.ruta else ""


def guardar(letra, persona, ident, puntos, lados, fps, video, origen=""):
    """Guarda una grabación. puntos: lista de arrays (21, 3) o None si no había mano."""
    video.cerrar()
    con_mano = [p for p in puntos if p is not None]
    if len(con_mano) < VENTANA:
        print(f"  {letra}: solo {len(con_mano)} fotogramas con mano; no se guarda (mínimo {VENTANA})")
        video.descartar()
        return None
    arr = np.stack([p if p is not None else np.full((21, 3), np.nan, np.float32) for p in puntos])
    lado = max(set(lados), key=lados.count)              # la mano que más ha salido
    carpeta = os.path.join(CARPETA_DATOS, letra)
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, f"{ident}.npz")
    np.savez_compressed(ruta, puntos=arr, lado=lado, fps=float(fps), letra=letra, persona=persona,
                        video=video.relativa(), origen=origen)
    print(f"  {letra}: guardada {os.path.relpath(ruta)} ({len(con_mano)}/{len(puntos)} fotogramas con mano, mano {lado})"
          + (f"\n     vídeo: {video.relativa()}" if video.ruta else ""))
    return ruta


def cuantas(letra):
    carpeta = os.path.join(CARPETA_DATOS, letra)
    return len([f for f in os.listdir(carpeta) if f.endswith(".npz")]) if os.path.isdir(carpeta) else 0


def desde_video(args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ident = nuevo_id(args.persona)
    video = VideoLimpio(args.letra, ident, fps, activo=not args.sin_video)
    puntos, lados, n = [], [], 0
    with crear_detector() as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Reducimos ANTES de detectar: así los puntos están en píxeles del vídeo guardado
            frame = reducir(frame)
            video.escribir(frame)
            p, lado, _ = detectar(detector, frame, n * 1000 / fps)
            puntos.append(p)
            if lado:
                lados.append(lado)
            n += 1
    cap.release()
    print(f"{args.video}: {n} fotogramas, mano en {len(lados)}")
    if not lados:
        video.descartar()
        raise SystemExit("No se ha detectado ninguna mano en el vídeo.")
    guardar(args.letra, args.persona, ident, puntos, lados, fps, video, origen=os.path.basename(args.video))


def desde_camara(args):
    letras = args.letras or ALFABETO + [NADA]
    cap = cv2.VideoCapture(args.camara)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara {args.camara}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps_camara = cap.get(cv2.CAP_PROP_FPS) or 30   # solo para la cabecera del vídeo; el real va al .npz
    i = 0
    estado, t_estado = "espera", 0.0          # espera -> cuenta -> graba -> espera
    puntos, lados, t_ultimo = [], [], -1
    ident, video = None, None
    with crear_detector() as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = reducir(frame)
            t_ms = int(time.monotonic() * 1000)
            t_ms = max(t_ms, t_ultimo + 1)                   # MediaPipe exige tiempos crecientes
            t_ultimo = t_ms
            p, lado, _ = detectar(detector, frame, t_ms)
            ahora = time.monotonic()

            if estado == "cuenta" and ahora - t_estado >= 1.0:
                estado, t_estado, puntos, lados = "graba", ahora, [], []
                ident = nuevo_id(args.persona)
                video = VideoLimpio(letras[i], ident, fps_camara, activo=not args.sin_video)
            if estado == "graba":
                video.escribir(frame)                    # el fotograma limpio, antes de dibujar nada
                puntos.append(p)
                if lado:
                    lados.append(lado)
                if ahora - t_estado >= args.segundos:
                    fps = len(puntos) / max(1e-6, ahora - t_estado)
                    if lados:
                        guardar(letras[i], args.persona, ident, puntos, lados, fps, video)
                    else:
                        video.descartar()
                        print(f"  {letras[i]}: no se ha visto la mano; repite")
                    estado = "espera"

            # dibujo (en espejo, como un selfie: es más fácil colocarse)
            vis = frame.copy()
            dibujar_mano(vis, p)
            vis = cv2.flip(vis, 1)
            H, W = vis.shape[:2]
            e = H / 720
            texto(vis, letras[i], (int(30 * e), int(150 * e)), 4.5 * e, (255, 255, 255), int(10 * e))
            texto(vis, f"{i + 1}/{len(letras)} · grabaciones de {letras[i]}: {cuantas(letras[i])}",
                  (int(30 * e), int(200 * e)), 0.9 * e, (255, 255, 255), 2)
            if estado == "cuenta":
                texto(vis, "PREPARATE", (W // 2 - int(200 * e), H // 2), 2.2 * e, (0, 220, 255), int(5 * e))
            elif estado == "graba":
                cv2.circle(vis, (W - int(50 * e), int(50 * e)), int(22 * e), (0, 0, 230), -1)
                resto = max(0.0, args.segundos - (ahora - t_estado))
                texto(vis, f"GRABANDO {resto:.1f} s", (W - int(420 * e), int(65 * e)), 1.2 * e, (0, 0, 255), int(3 * e))
            if p is None:
                texto(vis, "no veo la mano", (int(30 * e), H - int(60 * e)), 1.0 * e, (0, 0, 255), 2)
            texto(vis, "ESPACIO: grabar   d/a: letra siguiente/anterior   q: salir",
                  (int(30 * e), H - int(20 * e)), 0.75 * e, (255, 255, 255), 2)
            cv2.imshow("Grabar letras", vis)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                if estado == "graba":                    # salir a media grabación: no dejar vídeo suelto
                    video.descartar()
                break
            if estado == "espera":
                if k == ord(" "):
                    estado, t_estado = "cuenta", ahora
                elif k == ord("d"):
                    i = (i + 1) % len(letras)
                elif k == ord("a"):
                    i = (i - 1) % len(letras)
    cap.release()
    cv2.destroyAllWindows()
    print("\nGrabaciones por letra:")
    print("  " + "  ".join(f"{l}={cuantas(l)}" for l in letras))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--persona", required=True, help="quién graba (va en el nombre del fichero)")
    ap.add_argument("--letras", nargs="+", help=f"letras a grabar (por defecto todas + «{NADA}»)")
    ap.add_argument("--camara", type=int, default=0, help="número de cámara (0)")
    ap.add_argument("--segundos", type=float, default=3.0, help="duración de cada grabación (3 s)")
    ap.add_argument("--video", help="sacar los puntos de un vídeo ya grabado en vez de la webcam")
    ap.add_argument("--letra", help="con --video: la letra que se hace en ese vídeo")
    ap.add_argument("--sin-video", action="store_true",
                    help="guardar solo los puntos, sin el vídeo limpio en datos/videos/")
    args = ap.parse_args()

    if "_" in args.persona or not args.persona.isalnum():
        # el nombre va delante de «_» en los ficheros, y de ahí se saca para separar por persona
        raise SystemExit("--persona: solo letras y números, sin «_» ni espacios (p. ej. keltse)")
    args.persona = args.persona.lower()

    validas = ALFABETO + [NADA]
    for l in (args.letras or []) + ([args.letra] if args.letra else []):
        if l not in validas:
            raise SystemExit(f"Letra desconocida: {l}. Válidas: {' '.join(validas)}")
    if args.video:
        if not args.letra:
            raise SystemExit("Con --video hay que decir la letra: --letra A")
        desde_video(args)
    else:
        desde_camara(args)


if __name__ == "__main__":
    main()
