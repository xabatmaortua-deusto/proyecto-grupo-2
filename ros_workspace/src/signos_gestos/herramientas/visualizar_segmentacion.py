"""Visualización de la segmentación de la mano (vía clásica).

Qué hace
--------
Muestra la imagen original, la máscara binaria y la mano segmentada.
Permite comprobar visualmente el funcionamiento de segmentar.py.

Uso
---
    python3 herramientas/visualizar_segmentacion.py ruta_imagen.png

Técnicas vistas en clase (tema 4):
    cv2.imread, cv2.imshow y cv2.bitwise_and.
"""

import argparse

import cv2

from signos_gestos.clasico.segmentar import mascara_mano
from signos_gestos.clasico.medidas import medidas


def visualizar(frame):
    """Muestra la imagen original, la máscara y la mano segmentada."""
    mascara = mascara_mano(frame)

    mano = cv2.bitwise_and(frame, frame, mask=mascara)

    print("Medidas geométricas:")
    for nombre, valor in medidas(mascara).items():
        print(f"  {nombre}: {valor}")

    cv2.imshow("Imagen original", frame)
    cv2.imshow("Mascara binaria", mascara)
    cv2.imshow("Mano segmentada", mano)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """Carga la imagen indicada y visualiza su segmentación."""
    parser = argparse.ArgumentParser()
    parser.add_argument("imagen", help="Ruta de la imagen")
    args = parser.parse_args()

    frame = cv2.imread(args.imagen)

    if frame is None:
        raise SystemExit("No se ha podido cargar la imagen.")

    visualizar(frame)


if __name__ == "__main__":
    main()