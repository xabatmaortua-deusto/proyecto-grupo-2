"""
Detección de fichas en la bandeja enderezada: centro (mm), ángulo y recorte alineado.

Es la etapa 2 de docs/PLAN_FICHAS.md, con dos versiones que devuelven lo mismo:

- detectar_clasico: solo OpenCV. La madera es clara y el fondo de la bandeja oscuro, así que
  basta un umbral (Otsu) sobre el brillo, findContours y minAreaRect, filtrando por tamaño.
- detectar_dl: YOLO (una clase, "ficha") da una caja por ficha y, dentro de cada caja, el
  mismo minAreaRect da el centro y el ángulo exactos.

Cada ficha detectada es un dict:
    x, y       centro en mm de la bandeja (origen en el centro del ArUco 0, +X al 1, +Y al 2)
    angulo     rad, medido desde +X de la bandeja: hacia dónde apunta el "arriba" del recorte
    recorte    imagen BGR de la ficha, girada para que sus bordes queden rectos
    confianza  0..1 (1.0 en el clásico; la de YOLO en el DL)

Como la ficha (18 x 20 mm) es casi cuadrada, aquí no se sabe dónde está la parte de arriba de
la letra: de sus 4 lados, el "arriba" del recorte es el que mira más hacia -Y, así que el
recorte se ve como en la enderezada, girado menos de 45°. La etapa de letras decide el giro k
(0..3) tal que np.rot90(recorte, k) deja la letra derecha, y theta_en_bandeja(angulo, k) da
la theta del contrato (dirección raya -> parte de arriba de la letra).
"""

import math
import os

import cv2
import numpy as np

from signos_fichas.bandeja import CENTROS_MM, LADO_ARUCO_MM

PX_POR_MM = 2.0  # resolución de la enderezada (la de bandeja.enderezar por defecto)
ANCHO_FICHA_MM = 18.0
LARGO_FICHA_MM = 20.0
TOLERANCIA_MM = 4.0  # diferencia admitida con las medidas de la ficha
RELLENO_MIN = 0.85  # área del contorno / área de su rectángulo: las fichas son rectángulos
CONTRASTE_MIN = 40  # la madera tiene que ser al menos así de más clara que el fondo (0..255)
MARGEN_ARUCO_MM = 10.0  # papel blanco alrededor de cada ArUco: ahí no se buscan fichas
MARGEN_RECORTE_MM = 0.5  # el recorte se hace un poco hacia dentro para no meter fondo
MARGEN_CAJA_MM = 3.0  # DL: se amplía la caja de YOLO antes de buscar la ficha dentro
CONFIANZA_MIN_DL = 0.5
MODELO_DL = os.path.expanduser('~/ros_workspace/modelos/yolo_fichas.pt')

_modelos = {}  # modelos YOLO ya cargados, por ruta


def theta_en_bandeja(angulo, giro):
    """Devuelve theta (rad, en [-pi, pi]) con el ángulo de la detección y el giro de la letra."""
    return math.remainder(angulo + giro * math.pi / 2, 2 * math.pi)


def detectar_clasico(enderezada, px_por_mm=PX_POR_MM):
    """Detecta las fichas con un umbral sobre el brillo: la madera es clara y el fondo oscuro."""
    brillo = cv2.cvtColor(enderezada, cv2.COLOR_BGR2HSV)[:, :, 2]
    util = _fuera_de_los_aruco(brillo.shape, px_por_mm)
    otsu, _ = cv2.threshold(brillo[util].reshape(-1, 1), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Sin fichas, Otsu separaría el ruido del fondo: se exige un contraste mínimo
    umbral = max(otsu, np.median(brillo[util]) + CONTRASTE_MIN)
    binaria = ((brillo > umbral) & util).astype(np.uint8) * 255
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fichas = []
    for contorno in contornos:
        rect = _rectangulo_de_ficha(contorno, px_por_mm)
        if rect is not None:
            fichas.append(_ficha(enderezada, rect, px_por_mm, 1.0))
    return fichas


def detectar_dl(enderezada, px_por_mm=PX_POR_MM, modelo=MODELO_DL,
                confianza_min=CONFIANZA_MIN_DL):
    """Detecta las fichas con YOLO y saca su centro y su ángulo dentro de cada caja."""
    if modelo not in _modelos:
        from ultralytics import YOLO  # solo hace falta con el método DL
        _modelos[modelo] = YOLO(modelo)
    resultado = _modelos[modelo].predict(enderezada, conf=confianza_min, verbose=False)[0]
    return refinar_cajas(enderezada, resultado.boxes.xyxy.cpu().numpy(),
                         resultado.boxes.conf.cpu().numpy(), px_por_mm)


def refinar_cajas(enderezada, cajas, confianzas, px_por_mm=PX_POR_MM):
    """
    Busca la ficha dentro de cada caja (x1, y1, x2, y2 en píxeles de la enderezada).

    Dentro de la caja, ampliada un poco, Otsu separa la madera del fondo y minAreaRect del
    contorno más grande da el centro y el ángulo. Las cajas sin una ficha válida se descartan.
    """
    alto, ancho = enderezada.shape[:2]
    util = _fuera_de_los_aruco(enderezada.shape, px_por_mm)
    margen = MARGEN_CAJA_MM * px_por_mm
    fichas = []
    for (x1, y1, x2, y2), confianza in zip(cajas, confianzas):
        u1, v1 = max(0, int(x1 - margen)), max(0, int(y1 - margen))
        u2, v2 = min(ancho, int(x2 + margen) + 1), min(alto, int(y2 + margen) + 1)
        brillo = cv2.cvtColor(enderezada[v1:v2, u1:u2], cv2.COLOR_BGR2HSV)[:, :, 2]
        _, binaria = cv2.threshold(brillo, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binaria[~util[v1:v2, u1:u2]] = 0  # una ficha junto a un ArUco no se junta con su papel
        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            continue
        rect = _rectangulo_de_ficha(max(contornos, key=cv2.contourArea), px_por_mm)
        if rect is not None:
            (cu, cv), tamano, angulo_rect = rect  # el centro, en píxeles del trozo de la caja
            fichas.append(_ficha(enderezada, ((cu + u1, cv + v1), tamano, angulo_rect),
                                 px_por_mm, float(confianza)))
    return fichas


def dibujar(enderezada, fichas, px_por_mm=PX_POR_MM):
    """Dibuja cada ficha: su rectángulo, el número y una flecha hacia el "arriba" del recorte."""
    imagen = enderezada.copy()
    margen = 2 * MARGEN_RECORTE_MM * px_por_mm
    for i, f in enumerate(fichas):
        centro = np.array([f['x'], f['y']]) * px_por_mm
        alto, ancho = f['recorte'].shape[:2]
        rect = (tuple(centro), (ancho + margen, alto + margen), math.degrees(f['angulo']) + 90)
        cv2.polylines(imagen, [np.intp(np.round(cv2.boxPoints(rect)))], True, (0, 255, 0), 1)
        punta = centro + 0.8 * alto * np.array([math.cos(f['angulo']), math.sin(f['angulo'])])
        cv2.arrowedLine(imagen, tuple(np.intp(centro)), tuple(np.intp(punta)), (0, 0, 255), 1,
                        tipLength=0.3)
        cv2.putText(imagen, str(i), tuple(np.intp(centro + [alto * 0.6, alto * 0.6])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    return imagen


def _fuera_de_los_aruco(forma, px_por_mm):
    """Máscara (bool) de la enderezada sin los ArUco ni su papel, que también son claros."""
    util = np.ones(forma[:2], bool)
    m = (LADO_ARUCO_MM / 2 + MARGEN_ARUCO_MM) * px_por_mm
    for cx, cy in CENTROS_MM.values():
        u, v = cx * px_por_mm, cy * px_por_mm
        util[max(0, round(v - m)):round(v + m), max(0, round(u - m)):round(u + m)] = False
    return util


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
