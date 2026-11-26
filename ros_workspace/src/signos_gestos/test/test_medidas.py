"""Tests de signos_gestos/clasico/medidas.py · Responsable: Ane.

Comprueba cada medida con máscaras sencillas dibujadas con código (sin fotos ni cámara).

Se ejecuta desde la carpeta signos_gestos/ con:  python -m pytest test/test_medidas.py
(con «python -m» la carpeta actual entra en el path y se puede importar el paquete).
"""
import math

import cv2
import numpy as np

from signos_gestos.clasico.medidas import (ANGULO_MAX_HUECO, FRACCION_MIN_PROFUNDIDAD, angulo,
                                           contorno_mano, defectos_convexidad,
                                           huecos_entre_dedos, mayor_fraccion_agujero, medidas)


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


def rectangulo_girado(largo, corto, grados):
    """Máscara con un rectángulo relleno de largo x corto, girado los grados indicados."""
    m = lienzo()
    esquinas = cv2.boxPoints(((100, 100), (largo, corto), grados))
    cv2.fillPoly(m, [np.int32(esquinas)], 255)
    return m


def test_forma_compacta_alargamiento_cercano_a_1():
    # Como el puño de la A.
    m = lienzo()
    cv2.circle(m, (100, 100), 60, 255, -1)
    assert abs(medidas(m)['alargamiento'] - 1) < 0.1


def test_forma_alargada():
    # Como la B: tres veces más larga que ancha.
    assert abs(medidas(rectangulo_girado(150, 50, 0))['alargamiento'] - 3) < 0.15


def test_alargamiento_no_cambia_al_inclinar_la_mano():
    # Por eso se usa minAreaRect y no boundingRect: la B se hace de lado.
    recta = medidas(rectangulo_girado(150, 50, 0))['alargamiento']
    inclinada = medidas(rectangulo_girado(150, 50, 35))['alargamiento']
    assert abs(recta - inclinada) < 0.15


def mano(angulos_dedos, grosor=18):
    """Dibuja una mano más realista: palma redonda con dedos que salen de su centro.

    angulos_dedos: un ángulo por dedo, en grados (0 = hacia arriba, positivo = hacia la derecha).
    Sin antebrazo, como debe llegar la máscara de segmentar.py.
    """
    m = np.zeros((260, 260), dtype=np.uint8)
    centro, palma, largo = (130, 160), 38, 75
    cv2.circle(m, centro, palma, 255, -1)
    for grados in angulos_dedos:
        r = math.radians(grados)
        punta = (int(centro[0] + (palma + largo) * math.sin(r)),
                 int(centro[1] - (palma + largo) * math.cos(r)))
        cv2.line(m, centro, punta, 255, grosor)
    return m


def test_angulo_recto():
    assert abs(angulo((1, 0), (0, 0), (0, 1)) - 90) < 1e-6


def test_puno_sin_dedos():
    # A: puño cerrado.
    r = medidas(mano([]))
    assert r['n_dedos'] == 0
    assert r['angulo_dedos'] is None


def test_dedos_juntos_cuentan_como_cero():
    # B: los dedos juntos no dejan huecos entre ellos.
    assert medidas(mano([-6, 0, 6], grosor=22))['n_dedos'] == 0


def test_v_dos_dedos_con_angulo_pequeno():
    r = medidas(mano([-20, 20]))
    assert r['n_dedos'] == 2
    assert r['angulo_dedos'] < 60


def test_l_dos_dedos_con_angulo_grande():
    # Índice hacia arriba y pulgar hacia un lado.
    r = medidas(mano([0, -90]))
    assert r['n_dedos'] == 2
    assert r['angulo_dedos'] > 80


def test_l_inclinada_sigue_siendo_l():
    r = medidas(mano([20, -70]))
    assert r['n_dedos'] == 2
    assert r['angulo_dedos'] > 80


def test_w_tres_dedos():
    r = medidas(mano([-25, 0, 25]))
    assert r['n_dedos'] == 3
    assert r['angulo_dedos'] is None


def test_fraccion_de_agujero_en_bruto():
    # Valores en bruto que usa el cuaderno de análisis para elegir FRACCION_MIN_AGUJERO.
    lleno = lienzo()
    cv2.circle(lleno, (100, 100), 60, 255, -1)
    contorno, agujeros = contorno_mano(lleno)
    assert mayor_fraccion_agujero(agujeros, cv2.contourArea(contorno)) == 0.0

    anillo = lleno.copy()
    cv2.circle(anillo, (100, 100), 30, 0, -1)          # hueco de radio 30 en una mano de radio 60
    contorno, agujeros = contorno_mano(anillo)
    fraccion = mayor_fraccion_agujero(agujeros, cv2.contourArea(contorno))
    assert 0.2 < fraccion < 0.4                          # (30 / 60)² = 0,25 del círculo exterior


def test_defectos_en_bruto_incluyen_los_descartados():
    # El cuaderno necesita ver también los defectos que el filtro descarta.
    contorno, _ = contorno_mano(mano([0, -90]))
    todos = defectos_convexidad(contorno)
    huecos = huecos_entre_dedos(contorno)
    assert len(todos) > len(huecos)
    for d in todos:
        assert d['profundidad'] >= 0
        assert 0 <= d['angulo'] <= 180
    aceptados = [d for d in todos
                 if d['profundidad'] >= FRACCION_MIN_PROFUNDIDAD
                 and d['angulo'] < ANGULO_MAX_HUECO]
    assert len(aceptados) == len(huecos)


def test_se_queda_con_la_mancha_mas_grande():
    # Una mancha pequeña con agujero (ruido) no debe confundirse con la mano.
    m = lienzo()
    cv2.circle(m, (70, 100), 50, 255, -1)
    cv2.circle(m, (170, 30), 15, 255, -1)
    cv2.circle(m, (170, 30), 8, 0, -1)
    r = medidas(m)
    assert not r['tiene_agujero']
    assert r['area'] > 7000
