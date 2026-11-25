"""Tests de compuestas.py con imágenes sintéticas."""

import cv2
import numpy as np

from signos_fichas.bandeja import (calcular_homografia, CENTROS_MM, detectar_aruco, enderezar,
                                   esquinas_mm, LADO_ARUCO_MM)
from signos_fichas.compuestas import (borrar_fichas, enderezar_foto, falta_alguna_ficha,
                                      pegar_fichas, recortar_fichas, variar_luz)
from signos_fichas.deteccion import detectar_clasico, MARGEN_ARUCO_MM, PX_POR_MM
from signos_fichas.sinteticas import generar


def enderezada_sintetica(semilla, n_fichas=8):
    """Genera una imagen sintética y devuelve su enderezada y la verdad."""
    imagen, verdad = generar(semilla, n_fichas)
    return enderezar(imagen, calcular_homografia(detectar_aruco(imagen)), PX_POR_MM), verdad


def test_enderezar_foto_con_un_aruco_rellena_los_que_faltan():
    # Como en las fotos del 2026-11-23: solo se ve el ArUco 2. Se tapan los otros con fondo
    imagen, verdad = generar(5)
    for i in (0, 1, 3):
        centro = np.array(CENTROS_MM[i])
        tapa = (esquinas_mm(i) - centro) * 1.3 + centro  # el marcador y su papel blanco
        tapa = cv2.perspectiveTransform(tapa.reshape(-1, 1, 2), np.array(verdad['h_mm_a_px']))
        cv2.fillConvexPoly(imagen, np.int32(tapa.reshape(-1, 2)), (30, 30, 30))
    assert enderezar_foto(imagen) is None
    enderezada = enderezar_foto(imagen, minimo=1)
    assert enderezada.shape[:2] == (500, 600)
    m = round((LADO_ARUCO_MM / 2 + MARGEN_ARUCO_MM) * PX_POR_MM)
    for i, (cx, cy) in CENTROS_MM.items():
        u, v = round(cx * PX_POR_MM), round(cy * PX_POR_MM)
        zona = enderezada[max(0, v - m):v + m, max(0, u - m):u + m]
        assert (zona.std() > 50) == (i == 2)  # solo queda el ArUco 2, el resto es fondo liso


def test_falta_alguna_ficha():
    enderezada, _ = enderezada_sintetica(2)
    fichas = detectar_clasico(enderezada)
    assert not falta_alguna_ficha(enderezada, fichas)
    assert falta_alguna_ficha(enderezada, fichas[1:])


def test_borrar_fichas_deja_el_fondo_sin_fichas():
    enderezada, _ = enderezada_sintetica(1)
    fondo = borrar_fichas(enderezada)
    assert detectar_clasico(fondo) == []
    assert not falta_alguna_ficha(fondo, [])


def test_borrar_fichas_descarta_lo_que_no_puede_borrar():
    # Una mancha clara grande (una mano, un papel) no se borra: ese fondo no sirve
    enderezada, _ = enderezada_sintetica(1)
    cv2.rectangle(enderezada, (150, 150), (330, 330), (200, 200, 200), -1)
    assert borrar_fichas(enderezada) is None


def test_recortes_de_las_fichas():
    enderezada, verdad = enderezada_sintetica(3)
    recortes = recortar_fichas(enderezada)
    assert len(recortes) == len(verdad['fichas'])
    area = 18 * 20 * PX_POR_MM ** 2
    for r in recortes:
        assert r['imagen'].shape[:2] == r['alfa'].shape
        assert 0.8 * area < (r['alfa'] > 0.5).sum() < 1.25 * area


def test_las_fichas_pegadas_las_encuentra_el_clasico_en_sus_cajas():
    fondo = borrar_fichas(enderezada_sintetica(1)[0])
    recortes = recortar_fichas(enderezada_sintetica(2)[0])
    imagen, cajas = pegar_fichas(fondo, recortes, 6, np.random.default_rng(0))
    fichas = detectar_clasico(imagen)
    assert len(cajas) == len(fichas) == 6
    for f in fichas:
        u, v = f['x'] * PX_POR_MM, f['y'] * PX_POR_MM
        assert sum(x1 < u < x2 and y1 < v < y2 for x1, y1, x2, y2 in cajas) == 1


def test_variar_luz():
    enderezada, _ = enderezada_sintetica(0)
    a = variar_luz(enderezada, np.random.default_rng(1))
    b = variar_luz(enderezada, np.random.default_rng(2))
    assert a.shape == enderezada.shape and a.dtype == np.uint8
    assert not np.array_equal(a, b)
