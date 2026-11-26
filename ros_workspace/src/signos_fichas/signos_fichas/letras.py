"""
Etapa 3: qué letra tiene un recorte de ficha y cuántos cuartos de vuelta la dejan derecha.

Las fichas reales son de Scrabble: no tienen raya, así que el giro sale de la forma de la letra.
Se compara la letra del recorte, en sus 4 giros, con una plantilla de la letra derecha dibujada
con cv2.putText, y gana el giro que más se le parece.
"""

import cv2
import numpy as np

LADO = 32  # px del cuadrado donde se comparan letra y plantilla


def giro_clasico(recorte, letra):
    """Devuelve el giro k (0..3) tal que np.rot90(recorte, k) deja `letra` derecha."""
    mascara = normalizar(mascara_letra(recorte))
    plantilla = normalizar(plantilla_letra(letra))
    parecidos = [iou(np.rot90(mascara, k), plantilla) for k in range(4)]
    return int(np.argmax(parecidos))


def mascara_letra(recorte):
    """
    Máscara de la letra: la mancha oscura más grande que no toca el borde del recorte.

    Así se quedan fuera el número pequeño de Scrabble y el canto oscuro que a veces entra por
    los bordes.
    """
    gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    _, oscuro = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(oscuro)
    alto, ancho = oscuro.shape
    mejor, area_mejor = 0, 0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        toca_borde = x == 0 or y == 0 or x + w == ancho or y + h == alto
        if not toca_borde and area > area_mejor:
            mejor, area_mejor = i, area
    return (etiquetas == mejor).astype(np.uint8) * 255 if mejor else np.zeros_like(oscuro)


def plantilla_letra(letra):
    """Máscara de la letra derecha dibujada con una tipografía sencilla, como la de las fichas."""
    lienzo = np.zeros((120, 120), np.uint8)
    cv2.putText(lienzo, letra, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 8, cv2.LINE_AA)
    return lienzo


def normalizar(mascara):
    """Encaja la letra en un cuadrado de LADO px sin deformarla, para comparar sus formas."""
    x, y, w, h = cv2.boundingRect(mascara)
    escala = (LADO - 2) / max(w, h, 1)
    letra = cv2.resize(mascara[y:y + h, x:x + w],
                       (max(1, round(w * escala)), max(1, round(h * escala))))
    cuadrado = np.zeros((LADO, LADO), np.uint8)
    y0, x0 = (LADO - letra.shape[0]) // 2, (LADO - letra.shape[1]) // 2
    cuadrado[y0:y0 + letra.shape[0], x0:x0 + letra.shape[1]] = letra
    return cuadrado > 127


def iou(a, b):
    """Intersección entre unión de dos máscaras: 1 si son iguales, 0 si no se tocan."""
    return (a & b).sum() / max((a | b).sum(), 1)
