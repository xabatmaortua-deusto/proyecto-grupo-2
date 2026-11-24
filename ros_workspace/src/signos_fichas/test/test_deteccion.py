"""Tests de deteccion.py y evaluar_deteccion.py con imágenes sintéticas."""

import math

import numpy as np
import pytest

from signos_fichas.bandeja import calcular_homografia, detectar_aruco, enderezar
from signos_fichas.deteccion import detectar_clasico, PX_POR_MM, theta_en_bandeja
from signos_fichas.evaluar_deteccion import emparejar, error_angulo_grados
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


def test_emparejar():
    verdad = [{'x': 0, 'y': 0}, {'x': 50, 'y': 0}]
    detecciones = [{'x': 49, 'y': 1}, {'x': 1, 'y': 0}, {'x': 100, 'y': 100}]
    parejas = emparejar(detecciones, verdad)
    assert sorted((i, j) for i, j, _ in parejas) == [(0, 1), (1, 0)]  # la 2 es falsa


def test_error_angulo_es_modulo_90_grados():
    assert error_angulo_grados(0.0, math.pi / 2 + math.radians(3)) == pytest.approx(3)
    assert error_angulo_grados(math.radians(44), math.radians(-44)) == pytest.approx(2)
