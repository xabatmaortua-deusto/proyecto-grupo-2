"""Segmentación de la mano (vía clásica).

Qué hace
--------
Recibe un fotograma en color (BGR) y genera una máscara binaria:
la mano aparece en blanco y el fondo en negro.

La máscara se utiliza en clasico/medidas.py para calcular las
características geométricas de la mano.

Función principal
-----------------
    mascara_mano(frame_bgr) -> mascara

Técnicas
--------
Vistas en clase (tema 4 y Práctica 4):
    cv2.GaussianBlur para reducir el ruido.
    cv2.cvtColor para convertir la imagen a YCrCb.
    Umbralización de los canales Cr y Cb mediante NumPy.

Condiciones de captura
----------------------
Cámara fija, fondo claro y manga larga oscura para evitar
que el antebrazo aparezca en la máscara.
"""

import cv2
import numpy as np

# Tamaño inicial del filtro gaussiano para reducir el ruido
# sin perder demasiado detalle en los bordes de los dedos.
TAM_KERNEL = (5, 5)

# Rangos de color de piel en YCrCb (Práctica 4).
# Se ajustarán con las imágenes reales del dataset.
CR_MIN = 133
CR_MAX = 173

CB_MIN = 77
CB_MAX = 127


def mascara_mano(frame_bgr):
    """Devuelve una máscara binaria de los píxeles de piel."""
    # Reducir el ruido antes de aplicar el umbral.
    suavizada = cv2.GaussianBlur(frame_bgr, TAM_KERNEL, 0)

    # Separar la luminosidad de la información de color.
    imagen_ycrcb = cv2.cvtColor(suavizada, cv2.COLOR_BGR2YCrCb)

    # Extraer los canales de crominancia.
    cr = imagen_ycrcb[:, :, 1]
    cb = imagen_ycrcb[:, :, 2]

    # Seleccionar los píxeles que cumplen los rangos de piel.
    es_piel = (cr >= CR_MIN) & (cr <= CR_MAX) & \
              (cb >= CB_MIN) & (cb <= CB_MAX)

    # Convertir los valores booleanos en una máscara binaria.
    mascara = es_piel.astype(np.uint8) * 255

    return mascara