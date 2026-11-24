"""
Crea o amplía un dataset de YOLO (una sola clase: ficha) con bandejas enderezadas.

YOLO trabaja sobre la enderezada, igual que detectar_dl, así que cada imagen se endereza con
los ArUco (bandeja.py) antes de guardarla. Dos fuentes, que se pueden mezclar:

- --sinteticas N: imágenes de sinteticas.py. Las cajas salen de la verdad.
- --fotos carpeta: fotos reales de la cámara cenital (por ejemplo, los fotogramas de un vídeo).
  Las cajas las propone detectar_clasico, así que HAY QUE REVISARLAS: cada imagen se guarda
  también dibujada en revisar/, y las que tengan errores se borran (imagen y .txt) o se
  corrigen con un etiquetador antes de entrenar.

Cada llamada añade imágenes a un split (train o val). La validación se separa por vídeo:
los fotogramas de un vídeo van todos a train o todos a val.

Uso, desde la raíz del repo en cualquier PC con requirements.txt instalado (herramientas/ no
se monta en el contenedor; ros_workspace/datasets/ sí, y está fuera de git):
    export PYTHONPATH=ros_workspace/src/signos_fichas
    D=ros_workspace/datasets/fichas
    python3 herramientas/fichas/dataset_yolo.py $D --sinteticas 300 --semilla 10000
    python3 herramientas/fichas/dataset_yolo.py $D --sinteticas 60 --semilla 20000 --split val
    python3 herramientas/fichas/dataset_yolo.py $D --fotos fotogramas/video3 --split val

Entrenar (mejor con GPU) y dejar los pesos donde los busca detectar_dl dentro del contenedor:
    yolo detect train data=$D/data.yaml model=yolo11n.pt imgsz=640 epochs=60 project=runs
    cp runs/train/weights/best.pt ros_workspace/modelos/yolo_fichas.pt
"""

import argparse
import glob
import os

import cv2

from signos_fichas.bandeja import calcular_homografia, detectar_aruco, enderezar
from signos_fichas.deteccion import detectar_clasico, dibujar, MARGEN_RECORTE_MM, PX_POR_MM
from signos_fichas.evaluar_deteccion import caja_verdad
from signos_fichas.sinteticas import generar


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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('salida', help='carpeta del dataset (se crea o se amplía)')
    p.add_argument('--split', choices=['train', 'val'], default='train')
    p.add_argument('--sinteticas', type=int, default=0, help='número de imágenes sintéticas')
    p.add_argument('--semilla', type=int, default=0, help='semilla de la primera sintética')
    p.add_argument('--fotos', help='carpeta con fotos reales (.png o .jpg)')
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
        revisar = os.path.join(salida, 'revisar')
        os.makedirs(revisar, exist_ok=True)
        rutas = sorted(r for r in glob.glob(os.path.join(args.fotos, '*'))
                       if r.lower().endswith(('.png', '.jpg')))
        for ruta in rutas:
            imagen = cv2.imread(ruta)
            homografia = calcular_homografia(detectar_aruco(imagen))
            if homografia is None:
                print(f'{ruta}: no se ven 3 ArUco, se salta')
                continue
            enderezada = enderezar(imagen, homografia, args.px_por_mm)
            fichas = detectar_clasico(enderezada, args.px_por_mm)
            nombre = os.path.splitext(os.path.basename(ruta))[0]
            guardar(salida, args.split, nombre, enderezada,
                    [caja_de_deteccion(f, args.px_por_mm) for f in fichas])
            cv2.imwrite(os.path.join(revisar, nombre + '.png'),
                        dibujar(enderezada, fichas, args.px_por_mm))
        print(f'{len(rutas)} fotos: revisa las cajas en {revisar} antes de entrenar')

    with open(os.path.join(salida, 'data.yaml'), 'w') as f:
        f.write(f'path: {os.path.abspath(salida)}\ntrain: images/train\nval: images/val\n'
                'names:\n  0: ficha\n')
    print(f'Dataset en {salida}')


if __name__ == '__main__':
    main()
