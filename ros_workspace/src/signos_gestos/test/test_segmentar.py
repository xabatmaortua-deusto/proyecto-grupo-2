"""Tests de signos_gestos/clasico/segmentar.py.

Comprueba la segmentación mediante imágenes sintéticas dibujadas
con código, sin necesidad de fotografías ni cámara.

Se ejecuta desde la carpeta signos_gestos/ con:
    python3 -m pytest test/test_segmentar.py
"""

import cv2
import numpy as np

from signos_gestos.clasico.segmentar import mascara_mano
from signos_gestos.clasico.medidas import medidas


# Color sintético que cumple los umbrales YCrCb de piel.
COLOR_PIEL = (119, 149, 198)

# Fondo claro y manga oscura, siguiendo las condiciones de captura.
COLOR_FONDO = (240, 240, 240)
COLOR_MANGA = (25, 25, 25)


def lienzo():
    """Imagen BGR de 200 x 200 con fondo claro."""
    imagen = np.zeros((200, 200, 3), dtype=np.uint8)
    imagen[:] = COLOR_FONDO
    return imagen


def test_mascara_es_de_un_canal():
    imagen = lienzo()
    mascara = mascara_mano(imagen)

    assert mascara.shape == (200, 200)


def test_mascara_es_binaria():
    imagen = lienzo()
    cv2.circle(imagen, (100, 100), 50, COLOR_PIEL, -1)

    mascara = mascara_mano(imagen)

    assert mascara.dtype == np.uint8
    assert set(np.unique(mascara)) == {0, 255}


def test_fondo_sin_mano():
    mascara = mascara_mano(lienzo())

    assert np.count_nonzero(mascara) == 0


def test_detecta_color_de_piel():
    imagen = lienzo()
    cv2.circle(imagen, (100, 100), 50, COLOR_PIEL, -1)

    mascara = mascara_mano(imagen)

    assert mascara[100, 100] == 255
    assert mascara[10, 10] == 0


def test_manga_oscura_no_se_segmenta():
    imagen = lienzo()

    cv2.rectangle(imagen, (70, 100), (130, 190), COLOR_MANGA, -1)
    cv2.circle(imagen, (100, 70), 40, COLOR_PIEL, -1)

    mascara = mascara_mano(imagen)

    assert mascara[70, 100] == 255
    assert mascara[170, 100] == 0


def test_conserva_agujero():
    # Simula la letra O mediante una forma con un hueco interior.
    imagen = lienzo()

    cv2.circle(imagen, (100, 100), 60, COLOR_PIEL, -1)
    cv2.circle(imagen, (100, 100), 25, COLOR_FONDO, -1)

    mascara = mascara_mano(imagen)

    assert mascara[100, 100] == 0
    assert medidas(mascara)['tiene_agujero']


def test_compatible_con_medidas():
    imagen = lienzo()
    cv2.circle(imagen, (100, 100), 50, COLOR_PIEL, -1)

    resultado = medidas(mascara_mano(imagen))

    assert resultado['valida']
    assert resultado['n_dedos'] == 0
    assert not resultado['tiene_agujero']