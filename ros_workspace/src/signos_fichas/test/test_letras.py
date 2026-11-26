"""Tests de letras.py con recortes reales de las fichas (test/datos)."""

import glob
import os

import cv2
import numpy as np
import pytest

from signos_fichas.letras import giro_clasico

# Recortes de detectar_clasico a 4 px/mm de las fotos del 23-11. El nombre es
# <letra>_g<giro>_<n>.png: np.rot90(recorte, giro) deja la letra derecha.
DATOS = sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'datos', '*.png')))


def test_hay_recortes_de_prueba():
    assert len(DATOS) >= 10


@pytest.mark.parametrize('ruta', DATOS, ids=os.path.basename)
@pytest.mark.parametrize('k', range(4))
def test_giro_clasico_con_fichas_reales(ruta, k):
    # Girar el recorte k cuartos de vuelta equivale a tener la ficha girada en la bandeja,
    # así cada recorte real prueba los 4 giros
    letra, giro, _ = os.path.basename(ruta).split('_')
    recorte = np.rot90(cv2.imread(ruta), k)
    assert giro_clasico(recorte, letra) == (int(giro[1]) - k) % 4
