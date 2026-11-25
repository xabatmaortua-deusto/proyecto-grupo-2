"""
Bandeja: detectar los ArUco, enderezar la imagen y pasar de píxel a mm.

Es la base común de los métodos clásico y DL, y no depende de ROS.
Sigue el material de ArUco de clase (DICT_4X4_50, marcadores de 80 mm, ArucoDetector)
y su recomendación: homografía con cv2.findHomography usando las esquinas de todos los
marcadores frente a sus esquinas reales, y cv2.warpPerspective para enderezar.

Colocación: 4 ArUco sueltos en las esquinas de la bandeja, vistos desde la cámara:

    0 ---- 1
    |      |
    2 ---- 3

Coordenadas de la bandeja (mm): origen en el centro del ArUco 0, +X hacia el ArUco 1
y +Y hacia el ArUco 2. Pasar al marco world del robot es la calibración B3.
"""

import cv2
import numpy as np

LADO_ARUCO_MM = 80.0  # lado del marcador, borde negro incluido
ANCHO_MM = 300.0  # entre los centros de los ArUco 0 y 1 (PROVISIONAL: falta medir)
FONDO_MM = 250.0  # entre los centros de los ArUco 0 y 2 (PROVISIONAL: falta medir)
CENTROS_MM = {0: (0.0, 0.0), 1: (ANCHO_MM, 0.0), 2: (0.0, FONDO_MM), 3: (ANCHO_MM, FONDO_MM)}

DETECTOR = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))


def esquinas_mm(id_aruco):
    """Devuelve las 4 esquinas reales de un ArUco en el orden de OpenCV (sup-izq, sup-der...)."""
    cx, cy = CENTROS_MM[id_aruco]
    m = LADO_ARUCO_MM / 2
    return np.array([[cx - m, cy - m], [cx + m, cy - m],
                     [cx + m, cy + m], [cx - m, cy + m]], np.float32)


def detectar_aruco(imagen):
    """Detecta los ArUco de la bandeja y devuelve {id: esquinas (4, 2) en píxeles}."""
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    esquinas, ids, _ = DETECTOR.detectMarkers(gris)
    if ids is None:
        return {}
    return {int(i): e.reshape(4, 2) for i, e in zip(ids.flatten(), esquinas)
            if int(i) in CENTROS_MM}


def calcular_homografia(esquinas_px, minimo=3):
    """
    Calcula la homografía píxel -> mm con las esquinas de todos los ArUco que se ven.

    Con 3 ArUco (12 esquinas) sigue siendo fiable; con menos devuelve None. minimo=1 solo
    para hacer datasets con fotos donde se ve un ArUco: lejos de él el error crece.
    """
    if len(esquinas_px) < minimo:
        return None
    ids = sorted(esquinas_px)
    px = np.concatenate([esquinas_px[i] for i in ids])
    mm = np.concatenate([esquinas_mm(i) for i in ids])
    homografia, _ = cv2.findHomography(px, mm)
    return homografia


def px_a_mm(homografia, puntos_px):
    """Convierte puntos (N, 2) de píxeles de la imagen a mm de la bandeja."""
    puntos = np.float32(puntos_px).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(puntos, homografia).reshape(-1, 2)


def enderezar(imagen, homografia, px_por_mm=2.0):
    """
    Devuelve la bandeja vista desde arriba, entre los centros de los ArUco.

    En la imagen enderezada, el píxel (u, v) es el punto (u / px_por_mm, v / px_por_mm) en mm.
    """
    escala = np.array([[px_por_mm, 0, 0], [0, px_por_mm, 0], [0, 0, 1]])
    tamano = (round(ANCHO_MM * px_por_mm), round(FONDO_MM * px_por_mm))
    return cv2.warpPerspective(imagen, escala @ homografia, tamano)
