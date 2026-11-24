"""
Evaluación de la detección de fichas (etapa 2) con imágenes que traen su verdad.

Cada imagen pasa por ArUco -> enderezada -> detectar_clasico y/o detectar_dl, y cada ficha
real se empareja con la detección más cercana (a 5 mm como mucho). Se mide el tiempo por
imagen de la detección, las fichas perdidas y las falsas, el error del centro (mm) y el del
ángulo de los bordes (grados, módulo 90°: el giro de la letra es cosa de la etapa 3).

La verdad es el .json que guarda sinteticas.py junto a cada imagen:
{'fichas': [{'letra', 'x', 'y', 'theta'}, ...]} en mm y rad de la bandeja. Para fotos reales
vale el mismo formato.

Uso:
    python3 -m signos_fichas.evaluar_deteccion carpeta [--metodo clasico dl]
        [--modelo yolo_fichas.pt] [--dibujos carpeta_salida]
"""

import argparse
import functools
import glob
import json
import math
import os
import time

import cv2
import numpy as np

from signos_fichas import deteccion
from signos_fichas.bandeja import calcular_homografia, detectar_aruco, enderezar

DISTANCIA_MAX_MM = 5.0


def caja_verdad(ficha, px_por_mm=deteccion.PX_POR_MM, largo=deteccion.LARGO_FICHA_MM,
                ancho=deteccion.ANCHO_FICHA_MM):
    """
    Devuelve la caja (x1, y1, x2, y2) en píxeles de la enderezada de una ficha de la verdad.

    El lado largo va en la dirección de theta (la letra es más alta que ancha).
    """
    arriba = np.array([math.cos(ficha['theta']), math.sin(ficha['theta'])])
    derecha = np.array([-arriba[1], arriba[0]])
    centro = np.array([ficha['x'], ficha['y']])
    esquinas = [centro + a * largo / 2 * arriba + d * ancho / 2 * derecha
                for a in (-1, 1) for d in (-1, 1)]
    esquinas = np.array(esquinas) * px_por_mm
    return (*esquinas.min(axis=0), *esquinas.max(axis=0))


def emparejar(detecciones, verdad, distancia_max=DISTANCIA_MAX_MM):
    """
    Empareja detecciones con fichas reales, de la pareja más cercana a la más lejana.

    Devuelve [(i_deteccion, j_verdad, distancia_mm)]; lo que queda sin pareja son fichas
    falsas (detecciones) o perdidas (verdad).
    """
    candidatas = sorted((math.hypot(d['x'] - v['x'], d['y'] - v['y']), i, j)
                        for i, d in enumerate(detecciones) for j, v in enumerate(verdad))
    usadas_d, usadas_v, parejas = set(), set(), []
    for distancia, i, j in candidatas:
        if distancia <= distancia_max and i not in usadas_d and j not in usadas_v:
            usadas_d.add(i)
            usadas_v.add(j)
            parejas.append((i, j, distancia))
    return parejas


def error_angulo_grados(angulo, theta):
    """Error entre el ángulo de los bordes y la theta real, módulo 90° (en grados, 0..45)."""
    return abs(math.degrees(math.remainder(angulo - theta, math.pi / 2)))


def evaluar(imagenes, metodo, px_por_mm=deteccion.PX_POR_MM, modelo=None, dibujos=None):
    """Pasa la detección por las imágenes y devuelve un dict con las métricas."""
    if metodo == 'dl':
        detectar = functools.partial(deteccion.detectar_dl, px_por_mm=px_por_mm,
                                     modelo=modelo or deteccion.MODELO_DL)
    else:
        detectar = functools.partial(deteccion.detectar_clasico, px_por_mm=px_por_mm)
    tiempos, errores_mm, errores_grados = [], [], []
    n_fichas = perdidas = falsas = 0
    for ruta in imagenes:
        imagen = cv2.imread(ruta)
        with open(os.path.splitext(ruta)[0] + '.json', encoding='utf-8') as f:
            verdad = json.load(f)['fichas']
        homografia = calcular_homografia(detectar_aruco(imagen))
        if homografia is None:
            print(f'{ruta}: no se ven 3 ArUco, se salta')
            continue
        enderezada = enderezar(imagen, homografia, px_por_mm)
        if not tiempos:
            detectar(enderezada)  # la primera vez se carga el modelo: no se cuenta
        inicio = time.perf_counter()
        fichas = detectar(enderezada)
        tiempos.append(time.perf_counter() - inicio)
        parejas = emparejar(fichas, verdad)
        n_fichas += len(verdad)
        perdidas += len(verdad) - len(parejas)
        falsas += len(fichas) - len(parejas)
        errores_mm += [d for _, _, d in parejas]
        errores_grados += [error_angulo_grados(fichas[i]['angulo'], verdad[j]['theta'])
                           for i, j, _ in parejas]
        if dibujos:
            nombre = f'{os.path.splitext(os.path.basename(ruta))[0]}_{metodo}.png'
            cv2.imwrite(os.path.join(dibujos, nombre),
                        deteccion.dibujar(enderezada, fichas, px_por_mm))
    errores_mm = np.array(errores_mm or [np.nan])
    errores_grados = np.array(errores_grados or [np.nan])
    return {'metodo': metodo, 'imagenes': len(tiempos), 'fichas': n_fichas,
            'perdidas': perdidas, 'falsas': falsas,
            'error_mm_medio': errores_mm.mean(), 'error_mm_max': errores_mm.max(),
            'error_grados_medio': errores_grados.mean(), 'error_grados_max': errores_grados.max(),
            'ms_por_imagen': 1000 * np.mean(tiempos or [np.nan])}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('carpeta', help='imágenes (.png o .jpg) con su .json de verdad al lado')
    p.add_argument('--metodo', nargs='+', choices=['clasico', 'dl'], default=['clasico'])
    p.add_argument('--modelo', help=f'pesos de YOLO para dl ({deteccion.MODELO_DL})')
    p.add_argument('--px-por-mm', type=float, default=deteccion.PX_POR_MM)
    p.add_argument('--dibujos', help='carpeta donde guardar las detecciones dibujadas')
    args = p.parse_args()
    imagenes = sorted(r for r in glob.glob(os.path.join(args.carpeta, '*'))
                      if r.lower().endswith(('.png', '.jpg'))
                      and os.path.exists(os.path.splitext(r)[0] + '.json'))
    if args.dibujos:
        os.makedirs(args.dibujos, exist_ok=True)
    print('| Método | Imágenes | Fichas | Perdidas | Falsas | Error centro (mm) media / máx '
          '| Error ángulo (°) media / máx | ms por imagen |')
    print('|---|---|---|---|---|---|---|---|')
    for metodo in args.metodo:
        r = evaluar(imagenes, metodo, args.px_por_mm, args.modelo, args.dibujos)
        print(f"| {r['metodo']} | {r['imagenes']} | {r['fichas']} | {r['perdidas']} "
              f"| {r['falsas']} | {r['error_mm_medio']:.2f} / {r['error_mm_max']:.2f} "
              f"| {r['error_grados_medio']:.2f} / {r['error_grados_max']:.2f} "
              f"| {r['ms_por_imagen']:.1f} |")


if __name__ == '__main__':
    main()
