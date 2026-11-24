"""
Detección de fichas en la bandeja enderezada: centro (mm), ángulo y recorte alineado.

Como la ficha (18 x 20 mm) es casi cuadrada, aquí no se sabe dónde está la parte de arriba de
la letra: de sus 4 lados, el "arriba" del recorte es el que mira más hacia -Y, así que el
recorte se ve como en la enderezada, girado menos de 45°. La etapa de letras decide el giro k
(0..3) tal que np.rot90(recorte, k) deja la letra derecha, y theta_en_bandeja(angulo, k) da
la theta del contrato (dirección raya -> parte de arriba de la letra).
"""

import math


def theta_en_bandeja(angulo, giro):
    """Devuelve theta (rad, en [-pi, pi]) con el ángulo de la detección y el giro de la letra."""
    return math.remainder(angulo + giro * math.pi / 2, 2 * math.pi)
