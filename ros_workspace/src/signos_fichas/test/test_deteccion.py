"""Tests de deteccion.py y evaluar_deteccion.py con imágenes sintéticas."""

import math
import os

import cv2
import numpy as np
import pytest

from signos_fichas.bandeja import calcular_homografia, detectar_aruco, enderezar
from signos_fichas.deteccion import (detectar_clasico, detectar_dl, MODELO_DL, PX_POR_MM,
                                     refinar_cajas, theta_en_bandeja)
from signos_fichas.evaluar_deteccion import caja_verdad, emparejar, error_angulo_grados
from signos_fichas.sinteticas import generar


def enderezada_sintetica(semilla, n_fichas=8):
    """Genera una imagen sintética y devuelve su enderezada y la verdad de sus fichas."""
    imagen, verdad = generar(semilla, n_fichas)
    homografia = calcular_homografia(detectar_aruco(imagen))
    return enderezar(imagen, homografia, PX_POR_MM), verdad['fichas']


def comprobar(fichas, verdad, max_mm=1.0, max_grados=3.0):
    """Todas las fichas reales encontradas, sin falsas, con poco error de centro y ángulo."""
    parejas = emparejar(fichas, verdad)
    assert len(parejas) == len(verdad) == len(fichas)
    for i, j, distancia in parejas:
        assert distancia < max_mm
        assert error_angulo_grados(fichas[i]['angulo'], verdad[j]['theta']) < max_grados


@pytest.mark.parametrize('semilla', range(10))
def test_clasico_encuentra_todas_las_fichas(semilla):
    enderezada, verdad = enderezada_sintetica(semilla)
    comprobar(detectar_clasico(enderezada), verdad)


def test_clasico_sin_fichas_no_detecta_nada():
    # Los ArUco y su papel blanco también son claros: no pueden salir como fichas
    enderezada, _ = enderezada_sintetica(3, n_fichas=0)
    assert detectar_clasico(enderezada) == []


@pytest.mark.parametrize('semilla', range(3))
def test_recorte_y_giro_dan_la_theta_real(semilla):
    # Con el giro k correcto, np.rot90(recorte, k) deja la raya abajo y theta_en_bandeja
    # da la theta real: es el convenio que usan la etapa de letras y el nodo
    enderezada, verdad = enderezada_sintetica(semilla)
    fichas = detectar_clasico(enderezada)
    for i, j, _ in emparejar(fichas, verdad):
        f, theta = fichas[i], verdad[j]['theta']
        assert -3 * math.pi / 4 <= f['angulo'] <= -math.pi / 4
        giro = round(math.remainder(theta - f['angulo'], 2 * math.pi) / (math.pi / 2)) % 4
        assert abs(math.remainder(theta_en_bandeja(f['angulo'], giro) - theta,
                                  2 * math.pi)) < math.radians(2)
        filas = np.rot90(f['recorte'], giro).mean(axis=(1, 2))
        assert np.argmin(filas) > 0.7 * len(filas)  # la raya es la fila más oscura


def test_theta_en_bandeja():
    arriba = -math.pi / 2  # el "arriba" del recorte apunta a -Y de la bandeja
    assert theta_en_bandeja(arriba, 0) == pytest.approx(-math.pi / 2)
    assert theta_en_bandeja(arriba, 1) == pytest.approx(0)
    assert theta_en_bandeja(arriba, 2) == pytest.approx(math.pi / 2)
    assert abs(theta_en_bandeja(arriba, 3)) == pytest.approx(math.pi)


@pytest.mark.parametrize('semilla', range(3))
def test_refinar_cajas_como_las_de_yolo(semilla):
    # Cajas algo descentradas, como las que daría YOLO: el centro y el ángulo salen de dentro
    enderezada, verdad = enderezada_sintetica(semilla)
    rng = np.random.default_rng(semilla)
    cajas = [np.array(caja_verdad(f)) + rng.uniform(-3, 3, 4) for f in verdad]
    fichas = refinar_cajas(enderezada, cajas, [0.9] * len(cajas))
    comprobar(fichas, verdad)
    assert all(f['confianza'] == 0.9 for f in fichas)


def test_refinar_cajas_quita_el_canto_de_la_ficha():
    # Foto en ángulo: junto a la cara de arriba (18 x 20 mm, clara) se ve el canto (6 mm, más
    # oscuro). Con el primer Otsu salen 18 x 26 mm, que no es una ficha; con el de lo claro, sí
    enderezada = np.full((500, 600, 3), 40, np.uint8)  # en el centro, lejos de los ArUco
    cara = ((300, 250), (18 * PX_POR_MM, 20 * PX_POR_MM), 30)
    canto = ((300 - 13 * PX_POR_MM * math.sin(math.radians(30)),
              250 + 13 * PX_POR_MM * math.cos(math.radians(30))),
             (18 * PX_POR_MM, 6 * PX_POR_MM), 30)
    cv2.fillPoly(enderezada, [np.intp(cv2.boxPoints(canto))], (150, 150, 150))
    cv2.fillPoly(enderezada, [np.intp(cv2.boxPoints(cara))], (220, 220, 220))
    fichas = refinar_cajas(enderezada, [(260, 210, 340, 300)], [0.9])
    assert len(fichas) == 1
    assert math.hypot(fichas[0]['x'] - 150, fichas[0]['y'] - 125) < 0.5  # en mm: px / 2
    assert error_angulo_grados(fichas[0]['angulo'], math.radians(30)) < 3


def test_refinar_descarta_cajas_sin_ficha():
    enderezada, _ = enderezada_sintetica(5, n_fichas=0)
    assert refinar_cajas(enderezada, [(250, 200, 290, 240)], [0.8]) == []


def test_emparejar():
    verdad = [{'x': 0, 'y': 0}, {'x': 50, 'y': 0}]
    detecciones = [{'x': 49, 'y': 1}, {'x': 1, 'y': 0}, {'x': 100, 'y': 100}]
    parejas = emparejar(detecciones, verdad)
    assert sorted((i, j) for i, j, _ in parejas) == [(0, 1), (1, 0)]  # la 2 es falsa


def test_error_angulo_es_modulo_90_grados():
    assert error_angulo_grados(0.0, math.pi / 2 + math.radians(3)) == pytest.approx(3)
    assert error_angulo_grados(math.radians(44), math.radians(-44)) == pytest.approx(2)


@pytest.mark.skipif(not os.path.exists(os.environ.get('MODELO_FICHAS', MODELO_DL)),
                    reason='no hay pesos de YOLO (variable MODELO_FICHAS o MODELO_DL)')
@pytest.mark.parametrize('semilla', [1000, 1001, 1002])
def test_dl_encuentra_todas_las_fichas(semilla):
    pytest.importorskip('ultralytics')
    enderezada, verdad = enderezada_sintetica(semilla)
    comprobar(detectar_dl(enderezada, modelo=os.environ.get('MODELO_FICHAS', MODELO_DL)), verdad)
