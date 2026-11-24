"""Tests de deteccion.py y evaluar_deteccion.py con imágenes sintéticas."""

import math

import pytest

from signos_fichas.deteccion import theta_en_bandeja
from signos_fichas.evaluar_deteccion import emparejar, error_angulo_grados


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
