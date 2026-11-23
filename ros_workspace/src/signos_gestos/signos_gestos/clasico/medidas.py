"""Medidas geométricas de la silueta de la mano (vía clásica). Tarea C5 · Responsable: Ane.

Qué hace
--------
Recibe la máscara de la mano (imagen en blanco y negro: mano en blanco, fondo en negro), que genera
clasico/segmentar.py, y la resume en unos pocos números que describen su forma. Con esos números,
clasico/reconocer.py decide qué letra es. Este fichero NO segmenta la mano ni decide la letra.

Función principal
-----------------
    medidas(mascara) -> dict

Medidas
-------
Todas son relativas (cocientes o ángulos), para que no dependan de la distancia a la cámara.

    valida         False si no hay mano en la máscara (las demás medidas no tienen sentido).
    area           Área del contorno, en píxeles.                     → descartar ruido
    tiene_agujero  Hay un hueco interior grande (círculo índice-pulgar). → O
    n_dedos        (pendiente) Dedos estirados, por los huecos entre ellos. → A/B 0, L/V 2, W 3
    alargamiento   (pendiente) Lado largo / lado corto del rectángulo mínimo. → B frente a A
    solidez        (pendiente) Área del contorno / área de la envolvente. → comprobación extra
    angulo_dedos   (pendiente) Ángulo entre las puntas de los dos dedos.  → L frente a V

Técnicas
--------
Vistas en clase (tema 4 y sesión 5):
    cv2.findContours con RETR_TREE (contorno exterior y agujeros), cv2.contourArea, cv2.convexHull,
    cv2.minAreaRect y el cálculo de ángulos entre puntos de la sesión de MediaPipe.

No vista en clase (justificada):
    cv2.convexityDefects: para contar los dedos. Es la continuación directa de cv2.convexHull:
    devuelve los puntos del contorno más alejados de la envolvente, es decir, los huecos entre
    los dedos.

Pruebas
-------
test/test_medidas.py comprueba cada medida con máscaras sencillas dibujadas con código.
"""
import cv2
import numpy as np

# Un hueco interior cuenta como agujero de la O si su área es al menos esta fracción del área de la
# mano. Los huecos más pequeños son ruido de la segmentación. Valor inicial: se ajustará con las
# gráficas de herramientas/analisis_medidas.ipynb (tarea C6).
FRACCION_MIN_AGUJERO = 0.02


def contorno_mano(mascara):
    """Devuelve el contorno exterior más grande de la máscara (la mano) y sus agujeros.

    Usa cv2.findContours con RETR_TREE (tema 4): además de los contornos exteriores devuelve
    los interiores, y en la jerarquía indica de qué contorno es hijo cada uno. Los hijos del
    contorno de la mano son sus agujeros.

    Devuelve (contorno, agujeros), donde agujeros es una lista de contornos. Si la máscara
    está vacía, devuelve (None, []).
    """
    binaria = (np.asarray(mascara) > 0).astype(np.uint8) * 255
    contornos, jerarquia = cv2.findContours(binaria, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None, []

    # jerarquia[0][i] = [siguiente, anterior, primer_hijo, padre]. Exterior = sin padre (-1).
    jerarquia = jerarquia[0]
    exteriores = [i for i in range(len(contornos)) if jerarquia[i][3] == -1]
    i_mano = max(exteriores, key=lambda i: cv2.contourArea(contornos[i]))
    agujeros = [contornos[i] for i in range(len(contornos)) if jerarquia[i][3] == i_mano]
    return contornos[i_mano], agujeros


def tiene_agujero(agujeros, area_mano):
    """Indica si alguno de los agujeros es lo bastante grande para ser el círculo de la O."""
    if area_mano <= 0:
        return False
    return any(cv2.contourArea(a) >= FRACCION_MIN_AGUJERO * area_mano for a in agujeros)


def medidas(mascara):
    """Calcula las medidas geométricas de la mano a partir de su máscara.

    mascara: imagen de un canal; la mano es todo lo que no es 0.
    Devuelve un diccionario con las medidas descritas al principio del fichero.
    """
    contorno, agujeros = contorno_mano(mascara)
    if contorno is None:
        return {'valida': False}

    area = cv2.contourArea(contorno)
    return {
        'valida': True,
        'area': area,
        'tiene_agujero': tiene_agujero(agujeros, area),
    }
