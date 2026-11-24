"""Tests de bandeja.py con imágenes sintéticas."""

import cv2
import numpy as np
import pytest

from signos_fichas.bandeja import (ANCHO_MM, calcular_homografia, detectar_aruco, enderezar,
                                   FONDO_MM, px_a_mm)
from signos_fichas.sinteticas import generar


def centros_reales(verdad):
    """Devuelve los centros reales de las fichas en mm y en píxeles de la imagen."""
    mm = np.array([[f['x'], f['y']] for f in verdad['fichas']])
    px = cv2.perspectiveTransform(mm.reshape(-1, 1, 2), np.array(verdad['h_mm_a_px']))
    return mm, px.reshape(-1, 2)


@pytest.mark.parametrize('semilla', range(5))
def test_centros_de_las_fichas_en_mm(semilla):
    imagen, verdad = generar(semilla)
    esquinas = detectar_aruco(imagen)
    assert sorted(esquinas) == [0, 1, 2, 3]
    mm, px = centros_reales(verdad)
    error = np.linalg.norm(px_a_mm(calcular_homografia(esquinas), px) - mm, axis=1)
    assert error.max() < 1.0  # mm


def test_con_un_aruco_tapado_sigue_funcionando():
    imagen, verdad = generar(7)
    esquinas = detectar_aruco(imagen)
    del esquinas[3]
    mm, px = centros_reales(verdad)
    error = np.linalg.norm(px_a_mm(calcular_homografia(esquinas), px) - mm, axis=1)
    assert error.max() < 2.0  # mm
    assert calcular_homografia({0: esquinas[0], 1: esquinas[1]}) is None


def test_enderezada_tiene_madera_en_las_fichas():
    imagen, verdad = generar(4)
    ender = enderezar(imagen, calcular_homografia(detectar_aruco(imagen)), px_por_mm=2.0)
    assert ender.shape[:2] == (FONDO_MM * 2, ANCHO_MM * 2)
    fondo = np.median(ender)
    for f in verdad['fichas']:
        # Cuadrado de 6 x 6 mm en el centro de la ficha: casi todo es madera
        u, v = round(f['x'] * 2), round(f['y'] * 2)
        assert np.percentile(ender[v - 6:v + 6, u - 6:u + 6], 75) > fondo + 40
