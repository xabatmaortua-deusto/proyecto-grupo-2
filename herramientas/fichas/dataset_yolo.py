"""
Crea o amplía un dataset de YOLO (una sola clase: ficha) con bandejas enderezadas.

YOLO trabaja sobre la enderezada, igual que detectar_dl, así que cada imagen se endereza con
los ArUco (bandeja.py) antes de guardarla. Con --sinteticas N, las imágenes salen de
sinteticas.py y las cajas, de su verdad.

Cada llamada añade imágenes a un split (train o val). La validación se separa por vídeo:
los fotogramas de un vídeo van todos a train o todos a val.

Uso, desde la raíz del repo en cualquier PC con requirements.txt instalado (herramientas/ no
se monta en el contenedor; ros_workspace/datasets/ sí, y está fuera de git):
    export PYTHONPATH=ros_workspace/src/signos_fichas
    D=ros_workspace/datasets/fichas
    python3 herramientas/fichas/dataset_yolo.py $D --sinteticas 300 --semilla 10000
    python3 herramientas/fichas/dataset_yolo.py $D --sinteticas 60 --semilla 20000 --split val

Entrenar (mejor con GPU) y dejar los pesos donde los busca detectar_dl dentro del contenedor:
    yolo detect train data=$D/data.yaml model=yolo11n.pt imgsz=640 epochs=60 project=runs
    cp runs/train/weights/best.pt ros_workspace/modelos/yolo_fichas.pt
"""

import argparse
import os

import cv2

from signos_fichas.bandeja import calcular_homografia, detectar_aruco, enderezar
from signos_fichas.deteccion import PX_POR_MM
from signos_fichas.evaluar_deteccion import caja_verdad
from signos_fichas.sinteticas import generar


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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('salida', help='carpeta del dataset (se crea o se amplía)')
    p.add_argument('--split', choices=['train', 'val'], default='train')
    p.add_argument('--sinteticas', type=int, default=0, help='número de imágenes sintéticas')
    p.add_argument('--semilla', type=int, default=0, help='semilla de la primera sintética')
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

    with open(os.path.join(salida, 'data.yaml'), 'w') as f:
        f.write(f'path: {os.path.abspath(salida)}\ntrain: images/train\nval: images/val\n'
                'names:\n  0: ficha\n')
    print(f'Dataset en {salida}')


if __name__ == '__main__':
    main()
