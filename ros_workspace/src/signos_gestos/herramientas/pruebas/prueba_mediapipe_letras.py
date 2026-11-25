#!/usr/bin/env python3
"""Prueba inicial: reconocer letras del alfabeto dactilológico con MediaPipe + reglas.

Idea (la misma de la Práctica 5, pero con la MANO en vez del cuerpo):
  1. MediaPipe HandLandmarker encuentra la mano en cada fotograma y devuelve 21 puntos
     (muñeca, y 4 articulaciones por dedo).
  2. Con esos puntos calculamos ÁNGULOS y DISTANCIAS: qué dedos están estirados, si el
     pulgar está separado, si la punta del pulgar toca la del índice...
  3. Unas REGLAS escritas a mano convierten esas medidas en una letra (o '?' si no encaja).
  4. Una pequeña MÁQUINA DE ESTADOS (como el contador de sentadillas) sólo acepta una letra
     cuando se mantiene estable un rato, y así va formando la palabra.

Uso, desde la carpeta signos_gestos/ (la del package.xml):
  python herramientas/pruebas/prueba_mediapipe_letras.py              # todos los .MOV/.mp4 de signos_gestos/
  python herramientas/pruebas/prueba_mediapipe_letras.py IMG_2485.MOV # sólo ese vídeo
  python herramientas/pruebas/prueba_mediapipe_letras.py --ver        # además lo muestra (q = salir)

Resultados en signos_gestos/resultados_mediapipe/:
  <video>_anotado.mp4      vídeo con el esqueleto de la mano, la letra y la palabra
  <video>_datos.csv        medidas fotograma a fotograma (para ajustar umbrales)
  <video>_linea_tiempo.png gráfica de qué dedos están estirados y qué letra sale en cada momento
"""
import argparse
import csv
import glob
import math
import os
import sys
import urllib.request

import cv2
import matplotlib
matplotlib.use("Agg")                      # guardamos las gráficas en fichero, sin abrir ventanas
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

# Este script vive en herramientas/pruebas/: el modelo, los vídeos y los resultados están en la
# carpeta del paquete (RAIZ, dos niveles más arriba), y desde ahí se importa signos_gestos.
RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

# La palabra (máquina de estados) se construye con la clase común a las dos vías
from signos_gestos.comun.palabra import Palabra  # noqa: E402

MODELO = os.path.join(RAIZ, "hand_landmarker.task")     # el mismo que usa letras_comun.py
URL_MODELO = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/1/hand_landmarker.task")
SALIDA = os.path.join(RAIZ, "resultados_mediapipe")

# ------------------------------------------------------------------ los 21 puntos de la mano
# 0 = muñeca. Cada dedo tiene 4 puntos, de la base (MCP) a la punta (TIP):
#   pulgar 1-4 · índice 5-8 · corazón 9-12 · anular 13-16 · meñique 17-20
MUNECA = 0
DEDOS = {                     # nombre: (base MCP, PIP, DIP, punta)
    "pulgar":  (1, 2, 3, 4),  # en el pulgar son CMC, MCP, IP, punta
    "indice":  (5, 6, 7, 8),
    "corazon": (9, 10, 11, 12),
    "anular":  (13, 14, 15, 16),
    "menique": (17, 18, 19, 20),
}
CONEXIONES = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]

# ------------------------------------------------------------------ umbrales (a ajustar)
# Valores ajustados mirando las gráficas de los tres vídeos de prueba: un dedo estirado da
# 135-175° (MediaPipe nunca da 180° exactos) y uno doblado 90-110°.
ESTIRADO = 135      # grados: un dedo con la articulación PIP más abierta que esto está estirado
DOBLADO = 120       #         y más cerrada que esto está doblado (entre medias: «dudoso»)
TOCAN = 0.35        # punta pulgar–punta índice < 0.35 × tamaño de la palma  => se tocan (≈0,25 en la pinza)
PULGAR_FUERA = 0.7  # punta pulgar–punta índice > 0.7 × palma => pulgar separado del índice (≈1,05 en la L)
JUNTOS = 0.45       # puntas índice–corazón < 0.45 × palma => dedos juntos (B ≈ 0,3), si no separados (V)
CERCA = 0.6         # O: punta pulgar–punta índice < 0.6 × palma (≈0,25-0,5 según el ángulo de la mano)
SEPARADO_O = 0.6    # O: punta índice–punta corazón > 0.6 × palma (en la O ≈0,8-1,1; en la B ≈0,3)


# ------------------------------------------------------------------ geometría
def p3(lm):
    """Punto de MediaPipe -> vector numpy (x, y, z)."""
    return np.array([lm.x, lm.y, lm.z])


def angulo(a, b, c):
    """Ángulo en b (grados) del triángulo a-b-c. 180° = recto, 0° = doblado del todo.
    Es la misma función de la práctica 5, pero en 3D."""
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return math.degrees(math.acos(np.clip(cos, -1, 1)))


def medidas(mundo):
    """Calcula las medidas de la mano a partir de los 21 puntos en coordenadas «del mundo».

    Usamos hand_world_landmarks (en metros, con origen en la mano) y no los de la imagen
    porque son 3D y no dependen de lo lejos que esté la mano de la cámara.
    """
    P = [p3(q) for q in mundo]
    palma = np.linalg.norm(P[MUNECA] - P[9])          # muñeca -> base del corazón: escala de la mano
    m = {}
    # 1) ¿Cuánto está estirado cada dedo? Ángulo en la articulación PIP (base-PIP-DIP),
    #    y en la DIP para que un dedo «en gancho» no cuente como estirado: nos quedamos con el menor.
    for nombre, (mcp, pip, dip, tip) in DEDOS.items():
        if nombre == "pulgar":
            m["ang_pulgar"] = angulo(P[mcp], P[pip], P[tip])   # pulgar: ángulo en su MCP
        else:
            m[f"ang_{nombre}"] = min(angulo(P[mcp], P[pip], P[dip]), angulo(P[pip], P[dip], P[tip]))
    # 2) Distancias entre puntas, divididas por el tamaño de la palma (así no dependen del tamaño)
    m["d_pulgar_indice"] = np.linalg.norm(P[4] - P[8]) / palma
    m["d_indice_corazon"] = np.linalg.norm(P[8] - P[12]) / palma
    m["d_pulgar_menique"] = np.linalg.norm(P[4] - P[17]) / palma   # pulgar cruzado sobre la palma
    # 3) Ángulo entre la dirección del pulgar y la del índice (90° en la L, pequeño si van juntos)
    m["ang_pulgar_indice"] = angulo(P[4] - P[2] + P[5], P[5], P[8])
    return m


def estado_dedos(m):
    """Convierte ángulos en 'E' (estirado), 'D' (doblado) o '·' (dudoso) para cada dedo."""
    est = {}
    for nombre in DEDOS:
        a = m[f"ang_{nombre}"]
        if nombre == "pulgar":            # el pulgar casi nunca se dobla tanto: umbrales más laxos
            est[nombre] = "E" if a > 150 else ("D" if a < 130 else "·")
        else:
            est[nombre] = "E" if a > ESTIRADO else ("D" if a < DOBLADO else "·")
    return est


# ------------------------------------------------------------------ reglas -> letra
def letra_por_reglas(m, est):
    """Reglas hechas a mano. Devuelve la letra o '?' si ninguna encaja claramente.

    Los cuatro dedos largos (índice, corazón, anular, meñique) mandan; el pulgar desempata.
    """
    i, c, a, me = est["indice"], est["corazon"], est["anular"], est["menique"]
    pinza = m["d_pulgar_indice"] < TOCAN

    # O (tal como la hacemos en nuestros vídeos): índice doblado formando un círculo con el pulgar
    # y los otros tres dedos estirados. Con la mano de perfil MediaPipe ve las puntas algo
    # separadas (≈0,45), así que pedimos «cerca» (no «tocando») y además que el índice se haya
    # apartado del corazón (en la B van pegados), para no confundirlas.
    if (m["d_pulgar_indice"] < CERCA and m["d_indice_corazon"] > SEPARADO_O
            and c == "E" and a == "E" and me == "E"):
        return "O"
    # O «cerrada»: todas las puntas curvadas y el pulgar tocando el índice (variante de la O)
    if pinza and c != "E" and a != "E" and me != "E" and i != "E":
        return "O"
    # L: índice estirado, el resto doblado y el pulgar bien separado del índice
    if i == "E" and c == "D" and a == "D" and me == "D" and m["d_pulgar_indice"] > PULGAR_FUERA:
        return "L"
    # V: índice y corazón estirados y separados, anular y meñique doblados
    if i == "E" and c == "E" and a == "D" and me == "D" and m["d_indice_corazon"] > JUNTOS:
        return "V"
    # W: índice, corazón y anular estirados, meñique doblado
    if i == "E" and c == "E" and a == "E" and me == "D":
        return "W"
    # B: los cuatro dedos estirados y juntos, pulgar recogido sobre la palma.
    # Toleramos un dedo «dudoso» (con la mano de lado el anular sale a veces a ~130°),
    # pero ninguno claramente doblado.
    largos = [i, c, a, me]
    if ("D" not in largos and largos.count("E") >= 3
            and m["d_indice_corazon"] < JUNTOS and m["d_pulgar_menique"] < 0.9):
        return "B"
    # A: puño cerrado (cuatro dedos doblados) con el pulgar estirado al lado
    if i == c == a == me == "D" and not pinza:
        return "A"
    return "?"


# ------------------------------------------------------------------ dibujo
def dibujar(frame, mano_img, letra, est, palabra, aceptada):
    H, W = frame.shape[:2]
    if mano_img is not None:
        px = [(int(q.x * W), int(q.y * H)) for q in mano_img]
        for a, b in CONEXIONES:
            cv2.line(frame, px[a], px[b], (46, 105, 217), 6, cv2.LINE_AA)
        for p in px:
            cv2.circle(frame, p, 9, (54, 42, 30), -1, cv2.LINE_AA)
    # banda superior con la letra del fotograma, el estado de los dedos y la palabra
    cv2.rectangle(frame, (0, 0), (W, 260), (255, 255, 255), -1)
    color = (0, 160, 0) if letra not in (None, "?") else (0, 0, 200)
    cv2.putText(frame, letra or "-", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 6, color, 14, cv2.LINE_AA)
    if est:
        txt = " ".join(f"{n[:3]}:{e}" for n, e in est.items())
        cv2.putText(frame, txt, (260, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (60, 60, 60), 3, cv2.LINE_AA)
    cv2.putText(frame, f"Palabra: {palabra}", (260, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.4,
                (200, 80, 0) if aceptada else (30, 30, 30), 6, cv2.LINE_AA)
    return frame


def grafica(fichero_png, filas, fps, titulo):
    """Línea de tiempo: arriba el ángulo de cada dedo, abajo la letra de cada fotograma."""
    t = [f["frame"] / fps for f in filas]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    for nombre in DEDOS:
        ax1.plot(t, [f.get(f"ang_{nombre}", np.nan) for f in filas], label=nombre)
    ax1.axhline(ESTIRADO, ls="--", c="gray"); ax1.axhline(DOBLADO, ls=":", c="gray")
    ax1.set_ylabel("ángulo del dedo (°)\n180 = estirado"); ax1.legend(ncol=5, fontsize=8)
    ax1.set_title(titulo)
    letras = sorted({f["letra"] for f in filas if f["letra"]})
    y = {l: k for k, l in enumerate(letras)}
    ax2.scatter(t, [y.get(f["letra"], np.nan) for f in filas], s=6)
    for f in filas:
        if f["aceptada"]:
            ax2.axvline(f["frame"] / fps, c="tab:red", lw=1)
    ax2.set_yticks(range(len(letras))); ax2.set_yticklabels(letras)
    ax2.set_xlabel("tiempo (s)   ·   rojo = letra aceptada en la palabra"); ax2.set_ylabel("letra")
    fig.tight_layout(); fig.savefig(fichero_png, dpi=110); plt.close(fig)


# ------------------------------------------------------------------ bucle principal
def procesar(video, ver=False):
    nombre = os.path.splitext(os.path.basename(video))[0]
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    opciones = vision.HandLandmarkerOptions(
        # Leemos el modelo con Python y le pasamos los bytes: MediaPipe no sabe abrir rutas con
        # tildes en Windows (la carpeta «Visión») y daría FileNotFoundError.
        base_options=mpt.BaseOptions(model_asset_buffer=open(MODELO, "rb").read()),
        running_mode=vision.RunningMode.VIDEO,   # modo vídeo: usa el fotograma anterior para seguir la mano
        num_hands=1,
        min_hand_detection_confidence=0.5, min_tracking_confidence=0.5)

    escritor, filas, palabra = None, [], Palabra()
    with vision.HandLandmarker.create_from_options(opciones) as detector:
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)          # MediaPipe quiere RGB, OpenCV da BGR
            r = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                          int(n * 1000 / fps))     # marca de tiempo en ms (creciente)
            fila = {"frame": n, "mano": bool(r.hand_landmarks), "letra": None, "aceptada": ""}
            est, mano_img = None, None
            if r.hand_landmarks:
                mano_img = r.hand_landmarks[0]
                m = medidas(r.hand_world_landmarks[0])
                est = estado_dedos(m)
                fila.update({k: round(v, 3) for k, v in m.items()})
                fila["dedos"] = "".join(est.values())
                fila["lado"] = r.handedness[0][0].category_name
                fila["letra"] = letra_por_reglas(m, est)
            nueva = palabra.actualizar(fila["letra"])
            fila["aceptada"] = nueva or ""
            fila["palabra"] = palabra.texto
            filas.append(fila)

            vis = dibujar(frame.copy(), mano_img, fila["letra"], est, palabra.texto, bool(nueva))
            vis = cv2.resize(vis, (vis.shape[1] // 2, vis.shape[0] // 2))   # vídeo de salida más ligero
            if escritor is None:
                escritor = cv2.VideoWriter(os.path.join(SALIDA, f"{nombre}_anotado.mp4"),
                                           cv2.VideoWriter_fourcc(*"mp4v"), fps, vis.shape[1::-1])
            escritor.write(vis)
            if ver:
                cv2.imshow("Prueba MediaPipe", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            n += 1
            if n % 50 == 0:
                print(f"  {nombre}: {n}/{total} fotogramas · palabra hasta ahora: {palabra.texto!r}", flush=True)
    cap.release()
    if escritor:
        escritor.release()

    # CSV con todas las medidas: sirve para ver por qué sale (o no) cada letra y ajustar umbrales
    columnas = list(dict.fromkeys(k for f in filas for k in f))
    with open(os.path.join(SALIDA, f"{nombre}_datos.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columnas, delimiter=";")
        w.writeheader(); w.writerows(filas)
    grafica(os.path.join(SALIDA, f"{nombre}_linea_tiempo.png"), filas, fps,
            f"{nombre} · palabra: {palabra.texto}")

    con_mano = sum(f["mano"] for f in filas)
    reparto = {}
    for f in filas:
        if f["letra"]:
            reparto[f["letra"]] = reparto.get(f["letra"], 0) + 1
    print(f"{nombre}: {len(filas)} fotogramas, mano detectada en {con_mano} ({100 * con_mano / max(1, len(filas)):.0f} %)")
    print(f"   fotogramas por letra: {dict(sorted(reparto.items(), key=lambda kv: -kv[1]))}")
    print(f"   PALABRA: {palabra.texto!r}")
    return palabra.texto


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="*", help="vídeos a procesar (por defecto, todos los de signos_gestos/)")
    ap.add_argument("--ver", action="store_true", help="mostrar el vídeo anotado mientras se procesa")
    args = ap.parse_args()

    if not os.path.exists(MODELO):
        print("Descargando el modelo de mano de MediaPipe...")
        urllib.request.urlretrieve(URL_MODELO, MODELO)
    os.makedirs(SALIDA, exist_ok=True)

    videos = args.videos or sorted({f for ext in ("*.MOV", "*.mov", "*.mp4", "*.MP4")
                                    for f in glob.glob(os.path.join(RAIZ, ext))})
    if not videos:
        raise SystemExit("No hay vídeos que procesar.")
    resultados = {os.path.basename(v): procesar(v, args.ver) for v in videos}
    print("\nResumen:")
    for v, p in resultados.items():
        print(f"  {v:16s} -> {p!r}")
    print(f"Resultados en {SALIDA}")


if __name__ == "__main__":
    main()
