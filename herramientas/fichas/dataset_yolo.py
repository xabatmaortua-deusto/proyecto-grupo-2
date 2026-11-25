"""
Crea o amplía un dataset de YOLO (una sola clase: ficha) con bandejas enderezadas.

YOLO trabaja sobre la enderezada, igual que detectar_dl, así que cada imagen se endereza con
los ArUco (bandeja.py) antes de guardarla. Tres fuentes, que se pueden mezclar:

- --sinteticas N: imágenes de sinteticas.py. Las cajas salen de la verdad.
- --fotos carpeta: fotos reales de la cámara cenital (por ejemplo, los fotogramas de un vídeo).
  Las cajas las propone detectar_clasico. Solo se guardan las fotos en las que todo lo claro
  tiene caja (falta_alguna_ficha): si no, YOLO aprendería que una ficha sin caja es fondo.
  Las guardadas se dibujan en revisar/ y las descartadas en revisar/incompletas/. Aun así,
  HAY QUE REVISARLAS: las que tengan errores se borran (imagen y .txt) antes de entrenar.
- --compuestas N (con --fotos): N imágenes nuevas hechas con las fichas y los fondos reales de
  esas fotos (compuestas.py): fichas recortadas y pegadas al azar, giradas y con otra luz. Así
  hay muchas imágenes reales distintas sin etiquetar nada a mano.

--min-aruco 1 deja usar fotos donde solo se ve un ArUco, como las del 2026-11-23.

Cada llamada añade imágenes a un split (train o val). La validación se separa por vídeo:
los fotogramas de un vídeo (y sus compuestas) van todos a train o todos a val.

Uso, desde la raíz del repo en cualquier PC con requirements.txt instalado (herramientas/ no
se monta en el contenedor; ros_workspace/datasets/ sí, y está fuera de git):
    export PYTHONPATH=ros_workspace/src/signos_fichas
    D=ros_workspace/datasets/fichas/yolo
    python3 herramientas/fichas/dataset_yolo.py $D --sinteticas 300 --semilla 10000
    python3 herramientas/fichas/dataset_yolo.py $D --fotos fotogramas/video3 --split val
    python3 herramientas/fichas/dataset_yolo.py $D --fotos fotogramas/video1 --compuestas 2000

Entrenar (mejor con GPU) y dejar los pesos donde los busca detectar_dl dentro del contenedor:
    yolo detect train data=$D/data.yaml model=yolo11n.pt imgsz=640 epochs=60 project=runs
    cp runs/train/weights/best.pt ros_workspace/modelos/yolo_fichas.pt
"""

import argparse
import glob
import os

import cv2
import numpy as np

from signos_fichas.bandeja import calcular_homografia, detectar_aruco, enderezar
from signos_fichas.compuestas import (borrar_fichas, enderezar_foto, falta_alguna_ficha,
                                      pegar_fichas, recortar_fichas, variar_luz)
from signos_fichas.deteccion import detectar_clasico, dibujar, MARGEN_RECORTE_MM, PX_POR_MM
from signos_fichas.evaluar_deteccion import caja_verdad
from signos_fichas.sinteticas import generar

FICHAS_POR_COMPUESTA = (3, 13)  # entre 3 y 12 fichas en cada imagen compuesta
COMPUESTAS_A_REVISAR = 20  # las primeras se guardan dibujadas en revisar/compuestas/


def caja_de_deteccion(ficha, px_por_mm):
    """Caja (x1, y1, x2, y2) en píxeles de una ficha detectada, a partir de su recorte."""
    alto, ancho = (lado / px_por_mm + 2 * MARGEN_RECORTE_MM for lado in ficha['recorte'].shape[:2])
    return caja_verdad({'x': ficha['x'], 'y': ficha['y'], 'theta': ficha['angulo']}, px_por_mm,
                       alto, ancho)


def guardar(salida, split, nombre, enderezada, cajas):
    """Guarda la imagen y su etiqueta en formato YOLO (clase cx cy w h, normalizados)."""
    alto, ancho = enderezada.shape[:2]
    cv2.imwrite(os.path.join(salida, 'images', split, nombre + '.png'), enderezada)
    with open(os.path.join(salida, 'labels', split, nombre + '.txt'), 'w') as f:
        for x1, y1, x2, y2 in cajas:
            x1, x2 = max(0.0, x1), min(ancho - 1.0, x2)
            y1, y2 = max(0.0, y1), min(alto - 1.0, y2)
            f.write(f'0 {(x1 + x2) / 2 / ancho:.6f} {(y1 + y2) / 2 / alto:.6f} '
                    f'{(x2 - x1) / ancho:.6f} {(y2 - y1) / alto:.6f}\n')


def dibujar_cajas(imagen, cajas):
    """Copia de la imagen con las cajas dibujadas, para revisarlas."""
    dibujo = imagen.copy()
    for x1, y1, x2, y2 in cajas:
        cv2.rectangle(dibujo, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
    return dibujo


def anadir_fotos(salida, split, carpeta, min_aruco, px_por_mm):
    """Añade las fotos completas de la carpeta y devuelve (recortes, fondos) de todas ellas."""
    revisar = os.path.join(salida, 'revisar')
    os.makedirs(os.path.join(revisar, 'incompletas'), exist_ok=True)
    rutas = sorted(r for r in glob.glob(os.path.join(carpeta, '*'))
                   if r.lower().endswith(('.png', '.jpg')))
    recortes, fondos = [], []
    sin_aruco = guardadas = 0
    for ruta in rutas:
        enderezada = enderezar_foto(cv2.imread(ruta), min_aruco, px_por_mm)
        if enderezada is None:
            sin_aruco += 1
            continue
        fichas = detectar_clasico(enderezada, px_por_mm)
        nombre = os.path.splitext(os.path.basename(ruta))[0]
        dibujo = dibujar(enderezada, fichas, px_por_mm)
        if falta_alguna_ficha(enderezada, fichas, px_por_mm):
            cv2.imwrite(os.path.join(revisar, 'incompletas', nombre + '.jpg'), dibujo)
        else:
            guardar(salida, split, nombre, enderezada,
                    [caja_de_deteccion(f, px_por_mm) for f in fichas])
            cv2.imwrite(os.path.join(revisar, nombre + '.jpg'), dibujo)
            guardadas += 1
        recortes += recortar_fichas(enderezada, px_por_mm)
        fondo = borrar_fichas(enderezada, px_por_mm)
        if fondo is not None:
            fondos.append(fondo)
    print(f'{len(rutas)} fotos: {sin_aruco} sin ArUco, {guardadas} guardadas, '
          f'{len(rutas) - sin_aruco - guardadas} incompletas. {len(recortes)} fichas recortadas '
          f'y {len(fondos)} fondos. Revisa las cajas en {revisar} antes de entrenar')
    return recortes, fondos


def anadir_compuestas(salida, split, n, semilla, recortes, fondos, prefijo, px_por_mm):
    """Añade n imágenes compuestas con los recortes y fondos reales."""
    revisar = os.path.join(salida, 'revisar', 'compuestas')
    os.makedirs(revisar, exist_ok=True)
    for k in range(semilla, semilla + n):
        rng = np.random.default_rng(k)
        imagen, cajas = pegar_fichas(fondos[rng.integers(len(fondos))], recortes,
                                     int(rng.integers(*FICHAS_POR_COMPUESTA)), rng, px_por_mm)
        imagen = variar_luz(imagen, rng)
        nombre = f'{prefijo}_compuesta_{k:05d}'
        guardar(salida, split, nombre, imagen, cajas)
        if k - semilla < COMPUESTAS_A_REVISAR:
            cv2.imwrite(os.path.join(revisar, nombre + '.jpg'), dibujar_cajas(imagen, cajas))
    print(f'{n} compuestas en {split}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('salida', help='carpeta del dataset (se crea o se amplía)')
    p.add_argument('--split', choices=['train', 'val'], default='train')
    p.add_argument('--sinteticas', type=int, default=0, help='número de imágenes sintéticas')
    p.add_argument('--semilla', type=int, default=0,
                   help='semilla de la primera sintética o compuesta')
    p.add_argument('--fotos', help='carpeta con fotos reales (.png o .jpg)')
    p.add_argument('--compuestas', type=int, default=0,
                   help='imágenes compuestas con las fichas y fondos de --fotos')
    p.add_argument('--min-aruco', type=int, default=3,
                   help='ArUco que tienen que verse en cada foto (3; 1 para fotos sin bandeja)')
    p.add_argument('--px-por-mm', type=float, default=PX_POR_MM)
    args = p.parse_args()
    salida = os.path.expanduser(args.salida)
    for carpeta in ('images', 'labels'):
        os.makedirs(os.path.join(salida, carpeta, args.split), exist_ok=True)

    for k in range(args.semilla, args.semilla + args.sinteticas):
        imagen, verdad = generar(k)
        enderezada = enderezar(imagen, calcular_homografia(detectar_aruco(imagen)), args.px_por_mm)
        cajas = [caja_verdad(f, args.px_por_mm) for f in verdad['fichas']]
        guardar(salida, args.split, f'sintetica_{k:05d}', enderezada, cajas)

    if args.fotos:
        recortes, fondos = anadir_fotos(salida, args.split, args.fotos, args.min_aruco,
                                        args.px_por_mm)
        if args.compuestas:
            if not recortes or not fondos:
                p.error('no hay fichas recortadas o fondos limpios para hacer compuestas')
            prefijo = os.path.basename(os.path.normpath(args.fotos))
            anadir_compuestas(salida, args.split, args.compuestas, args.semilla, recortes,
                              fondos, prefijo, args.px_por_mm)

    with open(os.path.join(salida, 'data.yaml'), 'w') as f:
        f.write(f'path: {os.path.abspath(salida)}\ntrain: images/train\nval: images/val\n'
                'names:\n  0: ficha\n')
    print(f'Dataset en {salida}')


if __name__ == '__main__':
    main()
