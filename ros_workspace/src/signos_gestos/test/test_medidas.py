"""Tests de signos_gestos/clasico/medidas.py · Responsable: Ane.

Comprueba cada medida con máscaras sencillas dibujadas con código (sin fotos ni cámara).

Se ejecuta desde la carpeta signos_gestos/ con:  python -m pytest test/test_medidas.py
(con «python -m» la carpeta actual entra en el path y se puede importar el paquete).
"""
import cv2
import numpy as np

from signos_gestos.clasico.medidas import medidas


def lienzo():
    """Máscara negra vacía de 200 x 200."""
    return np.zeros((200, 200), dtype=np.uint8)


def test_mascara_vacia_no_es_valida():
    assert medidas(lienzo()) == {'valida': False}


def test_circulo_lleno_no_tiene_agujero():
    m = lienzo()
    cv2.circle(m, (100, 100), 60, 255, -1)
    r = medidas(m)
    assert r['valida']
    assert not r['tiene_agujero']


def test_anillo_tiene_agujero():
    # Como la O: una forma con un hueco grande dentro.
    m = lienzo()
    cv2.circle(m, (100, 100), 60, 255, -1)
    cv2.circle(m, (100, 100), 25, 0, -1)
    assert medidas(m)['tiene_agujero']


def test_agujero_diminuto_es_ruido():
    # Un hueco de pocos píxeles (ruido de la segmentación) no cuenta como agujero.
    m = lienzo()
    cv2.circle(m, (100, 100), 60, 255, -1)
    cv2.circle(m, (100, 100), 2, 0, -1)
    assert not medidas(m)['tiene_agujero']


def test_se_queda_con_la_mancha_mas_grande():
    # Una mancha pequeña con agujero (ruido) no debe confundirse con la mano.
    m = lienzo()
    cv2.circle(m, (70, 100), 50, 255, -1)
    cv2.circle(m, (170, 30), 15, 255, -1)
    cv2.circle(m, (170, 30), 8, 0, -1)
    r = medidas(m)
    assert not r['tiene_agujero']
    assert r['area'] > 7000
