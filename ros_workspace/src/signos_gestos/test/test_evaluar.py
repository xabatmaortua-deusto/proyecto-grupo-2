"""Tests de signos_gestos/comun/evaluar.py · Responsable: Keltse.

Las métricas se comprueban con listas inventadas, y el recorrido de los vídeos con vídeos
pequeños generados con código (cada color es una «letra»), sin MediaPipe ni modelos.

Se ejecuta desde la carpeta signos_gestos/ con:  python -m pytest test/test_evaluar.py
"""
import os

import cv2
import numpy as np

from signos_gestos.comun.evaluar import (buscar_videos, evaluar, evaluar_palabras,
                                         formar_palabra, matriz_confusion, persona_de, resumen)


def test_matriz_confusion_cuenta_por_fila_real():
    reales = ['A', 'A', 'A', 'B', 'B']
    predichas = ['A', 'B', '?', 'B', 'B']
    m, columnas = matriz_confusion(reales, predichas, ['A', 'B'])
    assert columnas == ['A', 'B', '?']
    assert m.tolist() == [[1, 1, 1], [0, 2, 0]]


def test_letra_fuera_del_conjunto_tiene_su_columna():
    m, columnas = matriz_confusion(['A', 'A'], ['A', 'CH'], ['A', 'B'])
    assert columnas == ['A', 'B', 'CH', '?']
    assert m.tolist() == [[1, 0, 1, 0], [0, 0, 0, 0]]


def test_resumen_acierto_y_porcentaje_de_nada():
    reales = ['A'] * 4 + ['B'] * 2
    m, columnas = matriz_confusion(reales, ['A', 'A', '?', 'B', 'B', '?'], ['A', 'B'])
    filas = {f['letra']: f for f in resumen(m, ['A', 'B'], columnas)}
    assert filas['A']['acierto'] == 0.5
    assert filas['A']['porc_nada'] == 0.25
    assert abs(filas['A']['acierto_sin_nada'] - 2 / 3) < 1e-9
    assert filas['B']['acierto'] == 0.5
    assert filas['total']['fotogramas'] == 6
    assert filas['total']['acierto'] == 3 / 6
    assert filas['total']['porc_nada'] == 2 / 6


def test_letra_sin_fotogramas_no_divide_por_cero():
    m, columnas = matriz_confusion([], [], ['A'])
    assert resumen(m, ['A'], columnas)[0]['acierto'] == 0.0


def test_persona_sale_del_nombre():
    assert persona_de(os.path.join('datos', 'videos', 'A', 'Ane_20261125-101500.mp4')) == 'ane'


COLORES = {'A': (0, 0, 255), 'B': (0, 255, 0)}       # BGR: rojo = A, verde = B


def grabar(ruta, color, n=10):
    grabar_secuencia(ruta, [(color, n)])


def grabar_secuencia(ruta, tramos):
    """Vídeo de 64 x 64 hecho de tramos (color, n_fotogramas)."""
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    w = cv2.VideoWriter(ruta, cv2.VideoWriter_fourcc(*'mp4v'), 30, (64, 64))
    for color, n in tramos:
        for _ in range(n):
            w.write(np.full((64, 64, 3), color, dtype=np.uint8))
    w.release()


def por_color(frame):
    """Reconocedor de mentira: mira el color medio del fotograma."""
    b, g, r = frame.reshape(-1, 3).mean(0)
    return 'A' if r > 128 else ('B' if g > 128 else '?')


def test_evaluar_recorre_videos_por_letra_y_persona(tmp_path):
    grabar(str(tmp_path / 'A' / 'ane_1.mp4'), COLORES['A'])
    grabar(str(tmp_path / 'B' / 'ane_1.mp4'), COLORES['B'])
    grabar(str(tmp_path / 'B' / 'xabi_1.mp4'), COLORES['A'])    # una B mal hecha: parece A
    grabar(str(tmp_path / 'B' / 'xabi_2.mp4'), (0, 0, 0))       # sin nada que reconocer

    assert len(buscar_videos(str(tmp_path), ['A', 'B'], ['ane'])) == 2
    videos = buscar_videos(str(tmp_path), ['A', 'B'])
    assert len(videos) == 4

    m, columnas, filas = evaluar(lambda: por_color, videos, ['A', 'B'])
    assert columnas == ['A', 'B', '?']
    assert m.tolist() == [[10, 0, 0], [10, 10, 10]]
    total = filas[-1]
    assert total['fotogramas'] == 40
    assert total['acierto'] == 0.5


# ------------------------------------------------------------------ palabras (T2)
def test_formar_palabra_usa_la_clase_palabra():
    letras = ['L'] * 12 + ['?'] * 3 + ['O'] * 12 + ['B'] * 12 + ['O'] * 12
    assert formar_palabra(letras) == 'LOBO'
    assert formar_palabra(['A'] * 5) == ''               # menos de ESTABLE fotogramas
    assert formar_palabra(['A'] * 5, estable=5) == 'A'


def test_buscar_videos_sin_etiquetas_coge_todas_las_carpetas(tmp_path):
    grabar(str(tmp_path / 'BOLA' / 'ane_1.mp4'), COLORES['A'])
    grabar(str(tmp_path / 'OLA' / 'xabi_1.mp4'), COLORES['A'])
    assert [e for e, _ in buscar_videos(str(tmp_path))] == ['BOLA', 'OLA']
    assert [e for e, _ in buscar_videos(str(tmp_path), personas=['xabi'])] == ['OLA']


def test_evaluar_palabras_compara_real_y_reconocida(tmp_path):
    negro = (0, 0, 0)
    grabar_secuencia(str(tmp_path / 'AB' / 'ane_1.mp4'),
                     [(COLORES['A'], 12), (COLORES['B'], 12)])
    grabar_secuencia(str(tmp_path / 'AB' / 'xabi_1.mp4'),
                     [(COLORES['A'], 12), (negro, 10), (COLORES['A'], 12)])  # dice AA
    filas = evaluar_palabras(lambda: por_color, buscar_videos(str(tmp_path)))
    assert [(f['persona'], f['reconocida'], f['correcta']) for f in filas] == [
        ('ane', 'AB', True), ('xabi', 'AA', False)]
