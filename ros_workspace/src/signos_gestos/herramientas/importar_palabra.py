#!/usr/bin/env python3
"""Mete vídeos de palabras deletreadas en el dataset (T2) · Responsable: Keltse.

Los vídeos de palabras se graban con cualquier cámara (la de Windows, el móvil...). Este script
los deja como los vídeos de letras de grabar_letras.py:
    datos/palabras/<PALABRA>/<persona>_<fecha>.mp4
reducidos a 720 px de alto como mucho y sin espejo. <fecha> es la de modificación del fichero
original (cuando se grabó), así que el nombre no cambia si se importa dos veces.

Uso, desde la carpeta signos_gestos/ (la del package.xml):
    python herramientas/importar_palabra.py --palabra LOBO --persona ane VID_01.mp4 VID_02.mp4

Ver docs/DATASET.md: qué palabras grabar y en qué condiciones.
"""
import argparse
import os
from datetime import datetime

import cv2

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_PALABRAS = os.path.join(RAIZ, 'datos', 'palabras')
ALTO_MAX = 720          # igual que grabar_letras.py (letras_comun.ALTO_MAX)


def nombre_destino(origen, palabra, persona):
    """datos/palabras/<PALABRA>/<persona>_<fecha de modificación del original>.mp4."""
    fecha = datetime.fromtimestamp(os.path.getmtime(origen))
    return os.path.join(CARPETA_PALABRAS, palabra, f'{persona}_{fecha:%Y%m%d-%H%M%S}.mp4')


def importar(origen, destino):
    """Copia el vídeo fotograma a fotograma, reducido a ALTO_MAX de alto.

    Devuelve cuántos fotogramas ha copiado.
    """
    cap = cv2.VideoCapture(origen)
    if not cap.isOpened():
        raise SystemExit(f'No se pudo abrir {origen}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    escritor, n = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        alto, ancho = frame.shape[:2]
        if alto > ALTO_MAX:
            frame = cv2.resize(frame, (round(ancho * ALTO_MAX / alto), ALTO_MAX),
                               interpolation=cv2.INTER_AREA)
        if escritor is None:
            os.makedirs(os.path.dirname(destino), exist_ok=True)
            escritor = cv2.VideoWriter(destino, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                       frame.shape[1::-1])
        escritor.write(frame)
        n += 1
    cap.release()
    if escritor is not None:
        escritor.release()
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('videos', nargs='+', help='vídeos originales de la misma palabra y persona')
    ap.add_argument('--palabra', required=True, help='palabra que se deletrea (p. ej. LOBO)')
    ap.add_argument('--persona', required=True,
                    help='quién sale en el vídeo: solo letras y números (p. ej. ane)')
    args = ap.parse_args()

    palabra, persona = args.palabra.upper(), args.persona.lower()
    if not palabra.isalpha():
        raise SystemExit('--palabra: solo letras (p. ej. LOBO)')
    if not persona.isalnum():
        # el nombre va delante de «_» y de ahí se saca para separar por persona (evaluar.py)
        raise SystemExit('--persona: solo letras y números, sin «_» ni espacios')
    for origen in args.videos:
        destino = nombre_destino(origen, palabra, persona)
        if os.path.exists(destino):
            print(f'Ya existe {os.path.relpath(destino, RAIZ)}: no se vuelve a importar')
            continue
        n = importar(origen, destino)
        print(f'{os.path.basename(origen)} -> {os.path.relpath(destino, RAIZ)} ({n} fotogramas)')


if __name__ == '__main__':
    main()
