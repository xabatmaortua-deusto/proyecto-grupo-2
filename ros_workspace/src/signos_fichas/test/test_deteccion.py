"""Tests de deteccion.py y evaluar_deteccion.py con imágenes sintéticas."""

import math

import pytest

from signos_fichas.deteccion import theta_en_bandeja


def test_theta_en_bandeja():
    arriba = -math.pi / 2  # el "arriba" del recorte apunta a -Y de la bandeja
    assert theta_en_bandeja(arriba, 0) == pytest.approx(-math.pi / 2)
    assert theta_en_bandeja(arriba, 1) == pytest.approx(0)
    assert theta_en_bandeja(arriba, 2) == pytest.approx(math.pi / 2)
    assert abs(theta_en_bandeja(arriba, 3)) == pytest.approx(math.pi)
