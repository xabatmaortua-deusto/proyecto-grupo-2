"""
Imágenes sintéticas de la bandeja, para desarrollar y probar sin la cámara real.

Se dibuja la bandeja vista desde arriba (fondo oscuro, ArUco y fichas de madera con letra
y raya) y luego se deforma con una perspectiva aleatoria, como si la viera la cámara.
Junto a cada imagen se guarda la verdad: letra, posición (mm) y theta de cada ficha.

theta: dirección raya -> parte de arriba de la letra, en rad, medida desde +X de la bandeja.

Uso:
    python3 -m signos_fichas.sinteticas carpeta_salida --n 10
"""

import argparse
import json
import math
import os

import cv2
import numpy as np

from signos_fichas.bandeja import ANCHO_MM, CENTROS_MM, FONDO_MM, LADO_ARUCO_MM

LETRAS = 'ABLOVW'
PX_MM = 4  # resolución del dibujo cenital
MARGEN_MM = LADO_ARUCO_MM / 2 + 20  # alrededor de los centros de los ArUco
TAMANO_CAMARA = (1280, 960)


def mm_a_lienzo(x, y):
    """Pasa un punto en mm de la bandeja a píxeles del dibujo cenital."""
    return (x + MARGEN_MM) * PX_MM, (y + MARGEN_MM) * PX_MM


def dibujar_ficha(letra, rng):
    """Dibuja una ficha derecha de 18 x 20 mm: letra negra y raya debajo."""
    ancho, largo = 18 * PX_MM, 20 * PX_MM
    madera = np.clip(np.array([115, 165, 205]) + rng.integers(-15, 16, 3), 0, 255)
    ficha = np.zeros((largo, ancho, 3), np.uint8)
    ficha[:] = madera
    (tw, _), _ = cv2.getTextSize(letra, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)
    cv2.putText(ficha, letra, ((ancho - tw) // 2, round(13.5 * PX_MM)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (25, 25, 25), 4, cv2.LINE_AA)
    y_raya = round(16.5 * PX_MM)
    cv2.line(ficha, (4 * PX_MM, y_raya), (ancho - 4 * PX_MM, y_raya), (25, 25, 25), 5)
    return ficha


def elegir_posiciones(n, rng, separacion_mm=35.0):
    """Elige n centros dentro de la bandeja, sin pisar los ArUco y separados entre sí."""
    m = LADO_ARUCO_MM / 2 + 15
    centros = []
    while len(centros) < n:
        p = rng.uniform([m, m], [ANCHO_MM - m, FONDO_MM - m])
        if all(np.linalg.norm(p - c) >= separacion_mm for c in centros):
            centros.append(p)
    return centros


def generar(semilla=0, n_fichas=8):
    """
    Genera una imagen de cámara sintética.

    Devuelve (imagen BGR, verdad), con verdad = {'fichas': [{letra, x, y, theta}],
    'h_mm_a_px': homografía 3x3 de mm de la bandeja a píxeles de la imagen}.
    """
    rng = np.random.default_rng(semilla)
    ancho = round((ANCHO_MM + 2 * MARGEN_MM) * PX_MM)
    alto = round((FONDO_MM + 2 * MARGEN_MM) * PX_MM)
    lienzo = np.full((alto, ancho, 3), 30, np.uint8)  # fondo oscuro mate

    # ArUco con su papel blanco alrededor
    diccionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    lado = round(LADO_ARUCO_MM * PX_MM)
    for id_aruco, (cx, cy) in CENTROS_MM.items():
        u, v = (round(c) for c in mm_a_lienzo(cx, cy))
        b = round(lado * 0.6)
        cv2.rectangle(lienzo, (u - b, v - b), (u + b, v + b), (235, 235, 235), -1)
        marcador = cv2.aruco.generateImageMarker(diccionario, id_aruco, lado)
        lienzo[v - lado // 2:v + lado // 2, u - lado // 2:u + lado // 2] = marcador[:, :, None]

    # Fichas: se giran con getRotationMatrix2D y se llevan a su sitio
    fichas = []
    for x, y in elegir_posiciones(n_fichas, rng):
        letra = str(rng.choice(list(LETRAS)))
        theta = float(rng.uniform(-math.pi, math.pi))
        ficha = dibujar_ficha(letra, rng)
        h, w = ficha.shape[:2]
        # El "arriba" de la ficha (0, -1) tiene que apuntar a theta. En la imagen Y va hacia
        # abajo, y getRotationMatrix2D gira en sentido antihorario con ángulos positivos.
        angulo = -math.degrees(theta + math.pi / 2)
        giro = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angulo, 1.0)
        u, v = mm_a_lienzo(x, y)
        giro[:, 2] += [u - (w - 1) / 2, v - (h - 1) / 2]
        girada = cv2.warpAffine(ficha, giro, (ancho, alto))
        mascara = cv2.warpAffine(np.full((h, w), 255, np.uint8), giro, (ancho, alto))
        lienzo[mascara > 127] = girada[mascara > 127]
        fichas.append({'letra': letra, 'x': float(x), 'y': float(y), 'theta': theta})

    # Cámara: la bandeja en el centro de la imagen, con perspectiva aleatoria
    W, H = TAMANO_CAMARA
    escala = 0.75 * min(W / ancho, H / alto)
    origen = np.float32([[0, 0], [ancho, 0], [ancho, alto], [0, alto]])
    destino = (origen - [ancho / 2, alto / 2]) * escala + [W / 2, H / 2]
    destino += rng.uniform(-0.06, 0.06, (4, 2)) * [W, H]
    camara = cv2.getPerspectiveTransform(origen, np.float32(destino))
    imagen = cv2.warpPerspective(lienzo, camara, (W, H), borderValue=(90, 90, 90))

    # Luz, desenfoque y ruido
    imagen = cv2.GaussianBlur(imagen, (3, 3), 0).astype(np.float32)
    imagen = imagen * rng.uniform(0.8, 1.2) + rng.normal(0, 3, imagen.shape)
    imagen = np.clip(imagen, 0, 255).astype(np.uint8)

    lienzo_mm = np.array([[PX_MM, 0, MARGEN_MM * PX_MM], [0, PX_MM, MARGEN_MM * PX_MM], [0, 0, 1]])
    return imagen, {'fichas': fichas, 'h_mm_a_px': (camara @ lienzo_mm).tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('carpeta')
    p.add_argument('--n', type=int, default=10, help='número de imágenes (10)')
    p.add_argument('--semilla', type=int, default=0, help='semilla de la primera imagen (0)')
    args = p.parse_args()
    os.makedirs(args.carpeta, exist_ok=True)
    for k in range(args.semilla, args.semilla + args.n):
        imagen, verdad = generar(k)
        base = os.path.join(args.carpeta, f'sintetica_{k:03d}')
        cv2.imwrite(base + '.png', imagen)
        with open(base + '.json', 'w', encoding='utf-8') as f:
            json.dump(verdad, f, indent=2)
    print(f'{args.n} imágenes en {args.carpeta}')


if __name__ == '__main__':
    main()
