"""Tests de signos_gestos/clasico/reconocer.py.

Dos partes:
    1. Las reglas (clasificar_medidas) con medidas inventadas, sin imágenes. Los valores se
       eligen a partir de las constantes UMBRAL_* de reconocer.py, para que los tests sigan
       siendo válidos cuando los umbrales se ajusten con el dataset real.
    2. La cadena completa (reconocer) con imágenes en color dibujadas con código: fondo claro y
       una «mano» del color de piel que usa test_segmentar.py.

Se ejecuta desde la carpeta signos_gestos/ con:
    python -m pytest test/test_reconocer_clasico.py
"""
import math

import cv2
import numpy as np

from signos_gestos.clasico.reconocer import (UMBRAL_ALARGAMIENTO, UMBRAL_ANGULO,
                                              UMBRAL_SOLIDEZ, clasificar_medidas, reconocer)


# --- 1. Reglas, con medidas inventadas ---------------------------------------------------------

def medidas_de(**cambios):
    """Medidas de una mano válida, sin agujero y sin dedos, con los cambios indicados."""
    base = {
        'valida': True,
        'area': 10000.0,
        'tiene_agujero': False,
        'solidez': UMBRAL_SOLIDEZ + 0.05,
        'alargamiento': UMBRAL_ALARGAMIENTO - 0.5,
        'n_dedos': 0,
        'angulo_dedos': None,
    }
    base.update(cambios)
    return base


def test_sin_mano_devuelve_interrogacion():
    assert clasificar_medidas({'valida': False}) == '?'


def test_agujero_es_o():
    assert clasificar_medidas(medidas_de(tiene_agujero=True)) == 'O'


def test_agujero_manda_sobre_el_resto():
    # La O se comprueba la primera: aunque se detecten dedos, si hay agujero es O.
    r = medidas_de(tiene_agujero=True, n_dedos=2, angulo_dedos=UMBRAL_ANGULO + 20)
    assert clasificar_medidas(r) == 'O'


def test_sin_dedos_y_compacta_es_a():
    assert clasificar_medidas(medidas_de(alargamiento=UMBRAL_ALARGAMIENTO - 0.3)) == 'A'


def test_sin_dedos_y_alargada_es_b():
    assert clasificar_medidas(medidas_de(alargamiento=UMBRAL_ALARGAMIENTO + 0.3)) == 'B'


def test_sin_dedos_y_poco_solida_es_interrogacion():
    # Por ejemplo, un solo dedo levantado: no deja huecos, pero la forma no es compacta.
    r = medidas_de(solidez=UMBRAL_SOLIDEZ - 0.1, alargamiento=UMBRAL_ALARGAMIENTO + 0.3)
    assert clasificar_medidas(r) == '?'


def test_dos_dedos_con_angulo_grande_es_l():
    r = medidas_de(n_dedos=2, angulo_dedos=UMBRAL_ANGULO + 20, solidez=0.5)
    assert clasificar_medidas(r) == 'L'


def test_dos_dedos_con_angulo_pequeno_es_v():
    r = medidas_de(n_dedos=2, angulo_dedos=UMBRAL_ANGULO - 40, solidez=0.6)
    assert clasificar_medidas(r) == 'V'


def test_dos_dedos_sin_angulo_es_interrogacion():
    assert clasificar_medidas(medidas_de(n_dedos=2, angulo_dedos=None)) == '?'


def test_tres_dedos_es_w():
    assert clasificar_medidas(medidas_de(n_dedos=3, solidez=0.65)) == 'W'


def test_numero_de_dedos_sin_letra_es_interrogacion():
    # Ninguna de las 6 letras tiene 1, 4 o 5 dedos separados.
    for dedos in (1, 4, 5):
        assert clasificar_medidas(medidas_de(n_dedos=dedos)) == '?'


# --- 2. Cadena completa, con imágenes en color --------------------------------------------------

# Mismos colores que test_segmentar.py: piel dentro de los rangos YCrCb y fondo claro.
COLOR_PIEL = (119, 149, 198)
COLOR_FONDO = (240, 240, 240)


def fotograma():
    """Fotograma BGR de 260 x 260 con fondo claro."""
    imagen = np.zeros((260, 260, 3), dtype=np.uint8)
    imagen[:] = COLOR_FONDO
    return imagen


def foto_mano(angulos_dedos, grosor=18):
    """Fotograma con una mano dibujada: palma redonda y dedos que salen de su centro.

    angulos_dedos: un ángulo por dedo, en grados (0 = hacia arriba, positivo = derecha).
    """
    imagen = fotograma()
    centro, palma, largo = (130, 160), 38, 75
    cv2.circle(imagen, centro, palma, COLOR_PIEL, -1)
    for grados in angulos_dedos:
        r = math.radians(grados)
        punta = (int(centro[0] + (palma + largo) * math.sin(r)),
                 int(centro[1] - (palma + largo) * math.cos(r)))
        cv2.line(imagen, centro, punta, COLOR_PIEL, grosor)
    return imagen


def test_fotograma_vacio_o_inexistente_es_interrogacion():
    assert reconocer(None) == '?'
    assert reconocer(np.zeros((0, 0, 3), dtype=np.uint8)) == '?'


def test_fondo_sin_mano_es_interrogacion():
    assert reconocer(fotograma()) == '?'


def test_reconoce_a():
    assert reconocer(foto_mano([])) == 'A'


def test_reconoce_b():
    imagen = fotograma()
    cv2.rectangle(imagen, (100, 40), (160, 210), COLOR_PIEL, -1)
    assert reconocer(imagen) == 'B'


def test_reconoce_o():
    imagen = fotograma()
    cv2.circle(imagen, (130, 130), 70, COLOR_PIEL, -1)
    cv2.circle(imagen, (130, 130), 28, COLOR_FONDO, -1)
    assert reconocer(imagen) == 'O'


def test_reconoce_l():
    assert reconocer(foto_mano([0, -90])) == 'L'


def test_reconoce_v():
    assert reconocer(foto_mano([-20, 20])) == 'V'


def test_reconoce_w():
    assert reconocer(foto_mano([-25, 0, 25])) == 'W'


def test_un_solo_dedo_es_interrogacion():
    # No es ninguna de las 6 letras: mejor no contestar que confundirlo con la B.
    assert reconocer(foto_mano([0])) == '?'
