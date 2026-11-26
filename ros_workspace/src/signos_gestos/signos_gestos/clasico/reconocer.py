"""Reconocimiento de letras por reglas (vía clásica).

Qué hace
--------
Ofrece la función común a las dos vías:

    reconocer(frame_bgr) -> str

Recibe un fotograma de OpenCV (BGR) y devuelve 'A', 'B', 'L',
'O', 'V', 'W' o '?' si no reconoce ninguna letra.

Por dentro:
    1. Obtiene la máscara de la mano con clasico/segmentar.py.
    2. Calcula sus medidas con clasico/medidas.py.
    3. Aplica un árbol de reglas sobre esas medidas.

Árbol de reglas
---------------
    ¿hay agujero?                  → O
    0 dedos:  alargada             → B
              compacta             → A
    2 dedos:  ángulo grande        → L
              ángulo pequeño       → V
    3 dedos                        → W
    cualquier otro caso            → '?'

La solidez se utiliza como comprobación adicional para descartar
formas poco compactas cuando no se detectan dedos separados.

Umbrales
--------
Los valores iniciales se han elegido a partir de las imágenes
sintéticas analizadas en herramientas/analisis_medidas.ipynb.

Se ajustarán SOLO con las imágenes reales de entrenamiento,
nunca con las de test.

Pruebas
-------
test/test_reconocer_clasico.py: las reglas con medidas inventadas y la cadena completa con
imágenes en color dibujadas con código.
"""
from signos_gestos.clasico.segmentar import mascara_mano
from signos_gestos.clasico.medidas import medidas


# Valores iniciales obtenidos del análisis de las imágenes sintéticas (paso 7).

# La A tiene un alargamiento de 1.061 y la B de 2.625.
# El umbral permite distinguir el puño cerrado de los dedos extendidos.
UMBRAL_ALARGAMIENTO = 1.8

# La L presenta un ángulo de 115.184° y la V de 67.291°.
# El umbral separa las dos letras según la apertura entre los dedos.
UMBRAL_ANGULO = 90

# La A y la B presentan una solidez cercana a 1.
# El gesto de un solo dedo presenta una solidez de 0.815.
# Se descartan formas poco compactas cuando no hay dedos separados.
UMBRAL_SOLIDEZ = 0.9


def clasificar_medidas(resultado):
    """Decide la letra a partir de las medidas geométricas de la mano.

    Aplica las reglas en orden, empezando por las características
    que permiten separar las letras con mayor claridad.

    Devuelve '?' si no hay un contorno válido o no se reconoce el gesto.
    """
    if not resultado['valida']:
        return '?'

    # La O es la única letra que esperamos que tenga un agujero interior.
    if resultado['tiene_agujero']:
        return 'O'

    dedos = resultado['n_dedos']

    # A y B: no presentan separaciones entre los dedos.
    if dedos == 0:
        if resultado['solidez'] < UMBRAL_SOLIDEZ:
            return '?'
        if resultado['alargamiento'] > UMBRAL_ALARGAMIENTO:
            return 'B'
        return 'A'

    # L y V: presentan dos dedos separados y se distinguen por el ángulo.
    if dedos == 2:
        angulo = resultado['angulo_dedos']
        if angulo is None:
            return '?'
        if angulo > UMBRAL_ANGULO:
            return 'L'
        return 'V'

    # W: presenta tres dedos separados.
    if dedos == 3:
        return 'W'

    return '?'


def reconocer(frame_bgr):
    """Reconoce una letra a partir de un fotograma de OpenCV (BGR).

    Segmenta la mano, calcula sus medidas geométricas y aplica
    las reglas de clasificación.

    Devuelve 'A', 'B', 'L', 'O', 'V', 'W' o '?'.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return '?'

    mascara = mascara_mano(frame_bgr)
    resultado = medidas(mascara)
    return clasificar_medidas(resultado)
