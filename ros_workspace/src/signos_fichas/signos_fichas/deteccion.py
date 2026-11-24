"""
Detección de fichas en la bandeja enderezada: centro (mm), ángulo y recorte alineado.

Como la ficha (18 x 20 mm) es casi cuadrada, aquí no se sabe dónde está la parte de arriba de
la letra: de sus 4 lados, el "arriba" del recorte es el que mira más hacia -Y, así que el
recorte se ve como en la enderezada, girado menos de 45°. La etapa de letras decide el giro k
(0..3) tal que np.rot90(recorte, k) deja la letra derecha, y theta_en_bandeja(angulo, k) da
la theta del contrato (dirección raya -> parte de arriba de la letra).
"""

import math

import cv2
import numpy as np

ANCHO_FICHA_MM = 18.0
LARGO_FICHA_MM = 20.0
TOLERANCIA_MM = 4.0  # diferencia admitida con las medidas de la ficha
RELLENO_MIN = 0.85  # área del contorno / área de su rectángulo: las fichas son rectángulos
MARGEN_RECORTE_MM = 0.5  # el recorte se hace un poco hacia dentro para no meter fondo


def theta_en_bandeja(angulo, giro):
    """Devuelve theta (rad, en [-pi, pi]) con el ángulo de la detección y el giro de la letra."""
    return math.remainder(angulo + giro * math.pi / 2, 2 * math.pi)


def _rectangulo_de_ficha(contorno, px_por_mm):
    """Devuelve el minAreaRect del contorno si tiene el tamaño y la forma de una ficha, o None."""
    rect = cv2.minAreaRect(contorno)
    corto, largo = (lado / px_por_mm for lado in sorted(rect[1]))
    if abs(corto - ANCHO_FICHA_MM) > TOLERANCIA_MM or abs(largo - LARGO_FICHA_MM) > TOLERANCIA_MM:
        return None
    if cv2.contourArea(contorno) < RELLENO_MIN * rect[1][0] * rect[1][1]:
        return None
    return rect


def _ficha(enderezada, rect, px_por_mm, confianza):
    """Pasa un minAreaRect (en píxeles de la enderezada) al dict de una ficha detectada."""
    (cu, cv), _, _ = rect
    esquinas = cv2.boxPoints(rect)
    lado1, lado2 = esquinas[1] - esquinas[0], esquinas[2] - esquinas[1]
    # Los 4 lados apuntan a a1 + k * 90°: el "arriba" es el más cercano a -Y (-90°)
    a1 = math.atan2(lado1[1], lado1[0])
    k = round((-math.pi / 2 - a1) / (math.pi / 2))
    angulo = a1 + k * math.pi / 2
    alto, ancho = np.hypot(*lado1), np.hypot(*lado2)  # alto: el lado paralelo al "arriba"
    if k % 2:
        alto, ancho = ancho, alto

    # Transformación afín recorte -> enderezada: el "arriba" del recorte (0, -1) va a angulo
    margen = 2 * MARGEN_RECORTE_MM * px_por_mm
    ancho_r, alto_r = max(1, round(ancho - margen)), max(1, round(alto - margen))
    arriba = np.array([math.cos(angulo), math.sin(angulo)])
    derecha = np.array([-arriba[1], arriba[0]])
    origen = np.array([cu, cv]) - (ancho_r - 1) / 2 * derecha + (alto_r - 1) / 2 * arriba
    afin = np.column_stack([derecha, -arriba, origen])
    recorte = cv2.warpAffine(enderezada, afin, (ancho_r, alto_r),
                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    return {'x': cu / px_por_mm, 'y': cv / px_por_mm, 'angulo': angulo, 'recorte': recorte,
            'confianza': confianza}
