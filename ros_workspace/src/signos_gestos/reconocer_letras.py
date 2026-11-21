#!/usr/bin/env python3
"""Reconoce letras y forma palabras con el modelo entrenado (paso 3 de 3).

Uso:
  python reconocer_letras.py                          # webcam
  python reconocer_letras.py --camara 1
  python reconocer_letras.py IMG_2485.MOV IMG_2487.MOV  # vídeos (guarda el vídeo anotado)
  python reconocer_letras.py IMG_2485.MOV --sin-ventana # vídeos sin mostrar nada (más rápido)

Teclas (webcam): ESPACIO = añadir espacio · BORRAR = quitar la última letra · c = borrar todo · q = salir

Desde otro código (la interfaz común acordada en REPARTO_TAREAS.md):
  from reconocer_letras import ReconocedorLetras
  rec = ReconocedorLetras()
  letra = rec.reconocer(frame_bgr)   # 'A', 'B', ..., 'CH', 'LL', 'Ñ', ... o '?'

La construcción de la palabra es la misma clase Palabra de prueba_mediapipe_letras.py.
"""
import argparse
import os
import time
from collections import deque

import cv2
import joblib
import numpy as np

from letras_comun import (AQUI, MODELO_LETRAS, NADA, caracteristicas, crear_detector, detectar,
                          dibujar_mano, texto)
from prueba_mediapipe_letras import Palabra

SALIDA = os.path.join(AQUI, "resultados_letras")
UMBRAL = 0.5     # si el modelo está menos seguro que esto, decimos «?»


class ReconocedorLetras:
    """Guarda los últimos fotogramas con mano y clasifica la ventana en cada fotograma."""

    def __init__(self, ruta_modelo=MODELO_LETRAS, umbral=UMBRAL):
        if not os.path.exists(ruta_modelo):
            raise SystemExit(f"No existe {ruta_modelo}. Entrena primero con entrenar_letras.py")
        datos = joblib.load(ruta_modelo)
        self.modelo, self.ventana = datos["modelo"], datos["ventana"]
        self.umbral = umbral
        self.detector = crear_detector()
        self.buffer, self.lado = deque(maxlen=self.ventana), None
        self.t_ultimo = -1
        self.puntos, self.confianza = None, 0.0     # del último fotograma, para dibujar

    def reconocer(self, frame_bgr, t_ms=None):
        """Recibe un fotograma BGR de OpenCV y devuelve la letra o '?'."""
        if t_ms is None:
            t_ms = time.monotonic() * 1000
        t_ms = max(int(t_ms), self.t_ultimo + 1)          # MediaPipe exige tiempos crecientes
        self.t_ultimo = t_ms
        self.puntos, lado, _ = detectar(self.detector, frame_bgr, t_ms)
        self.confianza = 0.0
        if self.puntos is None:                           # sin mano: la ventana empieza de cero
            self.buffer.clear()
            return "?"
        if lado != self.lado:
            self.buffer.clear()
            self.lado = lado
        self.buffer.append(self.puntos)
        if len(self.buffer) < self.ventana:
            return "?"
        proba = self.modelo.predict_proba([caracteristicas(np.array(self.buffer), self.lado)])[0]
        k = int(np.argmax(proba))
        letra, self.confianza = self.modelo.classes_[k], float(proba[k])
        if letra == NADA or self.confianza < self.umbral:
            return "?"
        return letra

    def cerrar(self):
        self.detector.close()


def dibujar(frame, rec, letra, palabra, aceptada, espejo):
    dibujar_mano(frame, rec.puntos)
    if espejo:
        frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]
    e = H / 720
    banda = int(150 * e)
    capa = frame.copy()
    cv2.rectangle(capa, (0, 0), (W, banda), (40, 40, 40), -1)
    frame = cv2.addWeighted(capa, 0.8, frame, 0.2, 0)
    color = (80, 220, 80) if letra != "?" else (60, 60, 230)
    texto(frame, letra, (int(20 * e), int(120 * e)), 3.5 * e, color, 8 * e)
    texto(frame, f"seguridad {rec.confianza:.0%}", (int(260 * e), int(50 * e)), 0.9 * e, (255, 255, 255), 2)
    texto(frame, f"Palabra: {palabra}", (int(260 * e), int(115 * e)), 1.6 * e,
          (255, 170, 0) if aceptada else (255, 255, 255), 4 * e)
    return frame


def desde_camara(args):
    cap = cv2.VideoCapture(args.camara)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara {args.camara}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    rec, palabra = ReconocedorLetras(umbral=args.umbral), Palabra()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        letra = rec.reconocer(frame)
        nueva = palabra.actualizar(letra)
        cv2.imshow("Reconocer letras", dibujar(frame, rec, letra, palabra.texto, bool(nueva), espejo=True))
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            palabra.texto += " "
        elif k in (8, 127):                               # BORRAR (Windows / macOS)
            palabra.texto = palabra.texto[:-1]
        elif k == ord("c"):
            palabra.texto = ""
    cap.release()
    cv2.destroyAllWindows()
    rec.cerrar()
    print(f"Texto: {palabra.texto!r}")


def desde_video(ruta, args):
    cap = cv2.VideoCapture(ruta)
    if not cap.isOpened():
        print(f"No se pudo abrir {ruta}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    rec, palabra = ReconocedorLetras(umbral=args.umbral), Palabra()
    nombre = os.path.splitext(os.path.basename(ruta))[0]
    escritor, n = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        letra = rec.reconocer(frame, n * 1000 / fps)
        nueva = palabra.actualizar(letra)
        vis = dibujar(frame, rec, letra, palabra.texto, bool(nueva), espejo=False)
        if vis.shape[0] > 720:                            # vídeo de salida más ligero
            vis = cv2.resize(vis, (vis.shape[1] * 720 // vis.shape[0], 720))
        if escritor is None:
            os.makedirs(SALIDA, exist_ok=True)
            escritor = cv2.VideoWriter(os.path.join(SALIDA, f"{nombre}_letras.mp4"),
                                       cv2.VideoWriter_fourcc(*"mp4v"), fps, vis.shape[1::-1])
        escritor.write(vis)
        if not args.sin_ventana:
            cv2.imshow("Reconocer letras", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        n += 1
    cap.release()
    if escritor:
        escritor.release()
    rec.cerrar()
    print(f"{nombre}: {n} fotogramas -> {palabra.texto!r}")
    return palabra.texto


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="*", help="vídeos a procesar (sin vídeos: webcam)")
    ap.add_argument("--camara", type=int, default=0)
    ap.add_argument("--umbral", type=float, default=UMBRAL, help=f"seguridad mínima para dar una letra ({UMBRAL})")
    ap.add_argument("--sin-ventana", action="store_true", help="con vídeos: no mostrar nada en pantalla")
    args = ap.parse_args()
    if args.videos:
        resultados = {v: desde_video(v, args) for v in args.videos}
        cv2.destroyAllWindows()
        print("\nResumen:")
        for v, p in resultados.items():
            print(f"  {os.path.basename(v):20s} -> {p!r}")
        print(f"Vídeos anotados en {os.path.relpath(SALIDA)}/")
    else:
        desde_camara(args)


if __name__ == "__main__":
    main()
