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

ANCHO_FICHA_MM = 18.0
LARGO_FICHA_MM = 20.0
TOLERANCIA_MM = 4.0  # diferencia admitida con las medidas de la ficha
RELLENO_MIN = 0.85  # área del contorno / área de su rectángulo: las fichas son rectángulos


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
