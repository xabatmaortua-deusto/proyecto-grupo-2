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


def mano_con_dedos(n_dedos, separacion=30):
    """Dibuja una «mano» esquemática: una palma rectangular con n dedos separados encima."""
    m = lienzo()
    cv2.rectangle(m, (50, 110), (150, 180), 255, -1)                  # palma
    for i in range(n_dedos):
        x = 60 + i * separacion
        cv2.rectangle(m, (x, 30), (x + 12, 110), 255, -1)             # dedo
    return m


def test_formas_convexas_tienen_solidez_casi_1():
    # Puño (A) o dedos juntos (B): sin huecos, la envolvente coincide con la forma.
    m = lienzo()
    cv2.rectangle(m, (60, 40), (140, 170), 255, -1)
    assert medidas(m)['solidez'] > 0.98


def test_dedos_separados_bajan_la_solidez():
    # Con dedos separados (L, V, W) la envolvente cubre los huecos entre ellos.
    assert medidas(mano_con_dedos(3))['solidez'] < 0.85


def test_mismos_dedos_separados_tienen_menos_solidez_que_juntos():
    # Ojo: la solidez NO cuenta dedos (un solo dedo en un lado deja un hueco grande y la baja
    # mucho). Lo que sí mide: con los mismos dedos, separarlos aumenta los huecos.
    juntos = medidas(mano_con_dedos(4, separacion=12))['solidez']
    separados = medidas(mano_con_dedos(4, separacion=30))['solidez']
    assert separados < juntos


def test_se_queda_con_la_mancha_mas_grande():
    # Una mancha pequeña con agujero (ruido) no debe confundirse con la mano.
    m = lienzo()
    cv2.circle(m, (70, 100), 50, 255, -1)
    cv2.circle(m, (170, 30), 15, 255, -1)
    cv2.circle(m, (170, 30), 8, 0, -1)
    r = medidas(m)
    assert not r['tiene_agujero']
    assert r['area'] > 7000
