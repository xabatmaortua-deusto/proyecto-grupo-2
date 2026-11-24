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
import csv
import os
import time
from collections import Counter, deque

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
        self.prediccion, self.estado = None, "sin_mano"

    def reconocer(self, frame_bgr, t_ms=None):
        """Recibe un fotograma BGR de OpenCV y devuelve la letra o '?'."""
        if t_ms is None:
            t_ms = time.monotonic() * 1000
        t_ms = max(int(t_ms), self.t_ultimo + 1)          # MediaPipe exige tiempos crecientes
        self.t_ultimo = t_ms
        self.puntos, lado, _ = detectar(self.detector, frame_bgr, t_ms)
        self.confianza, self.prediccion = 0.0, None
        if self.puntos is None:                           # sin mano: la ventana empieza de cero
            self.buffer.clear()
            self.estado = "sin_mano"
            return "?"
        if lado != self.lado:
            self.buffer.clear()
            self.lado = lado
        self.buffer.append(self.puntos)
        if len(self.buffer) < self.ventana:
            self.estado = "ventana_incompleta"
            return "?"
        proba = self.modelo.predict_proba([caracteristicas(np.array(self.buffer), self.lado)])[0]
        k = int(np.argmax(proba))
        self.prediccion, self.confianza = self.modelo.classes_[k], float(proba[k])
        if self.prediccion == NADA:
            self.estado = "nada"
            return "?"
        if self.confianza < self.umbral:
            self.estado = "confianza_baja"
            return "?"
        self.estado = "reconocida"
        return self.prediccion

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


def abrir_diagnostico(ruta, habilitado):
    if not habilitado:
        return None, None
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    archivo = open(ruta, "w", newline="", encoding="utf-8")
    escritor = csv.writer(archivo, delimiter=";")
    escritor.writerow(("fotograma", "mano_detectada", "estado", "prediccion",
                      "confianza", "letra_devuelta", "letra_aceptada", "palabra"))
    return archivo, escritor


def registrar_diagnostico(escritor, fotograma, rec, letra, aceptada, palabra):
    if escritor:
        escritor.writerow((fotograma, rec.puntos is not None, rec.estado, rec.prediccion or "",
                           f"{rec.confianza:.4f}", "" if letra == "?" else letra,
                           aceptada or "", palabra))


def imprimir_diagnostico(estados):
    print("  Diagnóstico por fotograma: " + ", ".join(
        f"{estado}={cantidad}" for estado, cantidad in sorted(estados.items())))


def desde_camara(args):
    cap = cv2.VideoCapture(args.camara)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara {args.camara}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    rec, palabra = ReconocedorLetras(umbral=args.umbral), Palabra()
    ruta_diag = os.path.join(SALIDA, "webcam_diagnostico.csv")
    archivo_diag, escritor_diag = abrir_diagnostico(ruta_diag, args.diagnostico)
    estados, fotograma = Counter(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            letra = rec.reconocer(frame)
            nueva = palabra.actualizar(letra)
            estados[rec.estado] += 1
            registrar_diagnostico(escritor_diag, fotograma, rec, letra, nueva, palabra.texto)
            cv2.imshow("Reconocer letras", dibujar(
                frame, rec, letra, palabra.texto, bool(nueva), espejo=True))
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord(" "):
                palabra.texto += " "
            elif k in (8, 127):                           # BORRAR (Windows / macOS)
                palabra.texto = palabra.texto[:-1]
            elif k == ord("c"):
                palabra.texto = ""
            fotograma += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        rec.cerrar()
        if archivo_diag:
            archivo_diag.close()
    print(f"Texto: {palabra.texto!r}")
    imprimir_diagnostico(estados)
    if escritor_diag:
        print(f"Registro de diagnóstico: {ruta_diag}")


def desde_video(ruta, args):
    cap = cv2.VideoCapture(ruta)
    if not cap.isOpened():
        print(f"No se pudo abrir {ruta}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    rec, palabra = ReconocedorLetras(umbral=args.umbral), Palabra()
    nombre = os.path.splitext(os.path.basename(ruta))[0]
    ruta_diag = os.path.join(SALIDA, f"{nombre}_diagnostico.csv")
    archivo_diag, escritor_diag = abrir_diagnostico(ruta_diag, args.diagnostico)
    escritor, estados, n = None, Counter(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            letra = rec.reconocer(frame, n * 1000 / fps)
            nueva = palabra.actualizar(letra)
            estados[rec.estado] += 1
            registrar_diagnostico(escritor_diag, n, rec, letra, nueva, palabra.texto)
            vis = dibujar(frame, rec, letra, palabra.texto, bool(nueva), espejo=False)
            if vis.shape[0] > 720:                        # vídeo de salida más ligero
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
    finally:
        cap.release()
        if escritor:
            escritor.release()
        rec.cerrar()
        if archivo_diag:
            archivo_diag.close()
        if not args.sin_ventana:
            cv2.destroyAllWindows()
    print(f"{nombre}: {n} fotogramas -> {palabra.texto!r}")
    imprimir_diagnostico(estados)
    if escritor_diag:
        print(f"  Registro: {ruta_diag}")
    return palabra.texto


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="*", help="vídeos a procesar (sin vídeos: webcam)")
    ap.add_argument("--camara", type=int, default=0)
    ap.add_argument("--umbral", type=float, default=UMBRAL, help=f"seguridad mínima para dar una letra ({UMBRAL})")
    ap.add_argument("--sin-ventana", action="store_true", help="con vídeos: no mostrar nada en pantalla")
    ap.add_argument("--diagnostico", action="store_true",
                    help="guardar por fotograma detección, confianza y letra predicha en CSV")
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
