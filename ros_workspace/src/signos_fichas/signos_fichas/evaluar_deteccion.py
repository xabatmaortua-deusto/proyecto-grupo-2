"""
Evaluación de la detección de fichas (etapa 2) con imágenes que traen su verdad.

Cada ficha real se empareja con la detección más cercana (a 5 mm como mucho): lo que
queda sin pareja son fichas perdidas o falsas.
"""

import math

import numpy as np

from signos_fichas import deteccion

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
