"""
Imágenes compuestas para entrenar YOLO: muchas imágenes nuevas hechas con fichas y fondos reales.

De cada foto real enderezada se sacan:
- recortes: cada ficha suelta que encuentra detectar_clasico, con la máscara de su silueta
  (madera y canto) para pegarla sin su fondo;
- un fondo: la misma imagen con las fichas borradas (cv2.inpaint), que conserva la luz, las
  sombras y la textura reales. Si queda algo claro que no se puede borrar (una mano, un papel,
  muchas fichas amontonadas), ese fondo no se usa.

pegar_fichas pone recortes sobre un fondo, girados al azar y con su sombra, y variar_luz cambia la
exposición, el degradado de luz, el color, el enfoque, el ruido y la compresión de toda la
imagen. Las cajas de YOLO salen de dónde se ha pegado cada ficha: no hay que etiquetar a mano.

El giro de las fichas pegadas va de 0 a 360°: la caja de YOLO no tiene orientación, así que aquí no
hace falta saber hacia dónde mira la letra (en las fichas de Scrabble lo marca el número pequeño
de abajo a la derecha, y lo decide la etapa de letras).

También está aquí lo necesario para meter fotos reales en un dataset: enderezar_foto (con un solo
ArUco si se pide) y falta_alguna_ficha, para no guardar fotos con fichas sin etiquetar.
"""

import cv2
import numpy as np

from signos_fichas.bandeja import (calcular_homografia, CENTROS_MM, detectar_aruco, enderezar,
                                   LADO_ARUCO_MM)
from signos_fichas.deteccion import (ANCHO_FICHA_MM, detectar_clasico, fuera_de_los_aruco,
                                     LARGO_FICHA_MM, MARGEN_ARUCO_MM, PX_POR_MM, zonas_claras)

AREA_FICHA_MM2 = ANCHO_FICHA_MM * LARGO_FICHA_MM
MAX_FICHAS_JUNTAS = 6  # en el fondo se borran manchas claras de hasta 6 fichas que se tocan
RESTO_MIN = 0.3  # una zona clara de 0,3 fichas sin ficha detectada encima cuenta como ficha
BORDE_BORRADO_MM = 5.0  # al borrar una ficha del fondo, también su canto y su sombra
SEPARACION_MM = 3.0  # hueco mínimo entre fichas pegadas (así el clásico las separa)
SOMBRA_MM = 4.0  # hasta dónde llega la sombra de una ficha pegada


def enderezar_foto(imagen, minimo=3, px_por_mm=PX_POR_MM):
    """
    Endereza una foto real para un dataset, o devuelve None si no se ven `minimo` ArUco.

    Lo que cae fuera de la foto y los sitios de los ArUco que no se ven se rellenan con el color
    del fondo: el detector no busca fichas ahí, así que tampoco puede haber fichas sin etiquetar.
    """
    esquinas = detectar_aruco(imagen)
    homografia = calcular_homografia(esquinas, minimo)
    if homografia is None:
        return None
    enderezada = enderezar(imagen, homografia, px_por_mm)
    relleno = enderezar(np.full(imagen.shape[:2], 255, np.uint8), homografia, px_por_mm) < 128
    m = (LADO_ARUCO_MM / 2 + MARGEN_ARUCO_MM) * px_por_mm
    for id_aruco, (cx, cy) in CENTROS_MM.items():
        if id_aruco not in esquinas:
            u, v = cx * px_por_mm, cy * px_por_mm
            relleno[max(0, round(v - m)):round(v + m), max(0, round(u - m)):round(u + m)] = True
    if relleno.all():
        return None
    enderezada[relleno] = np.median(enderezada[~relleno], axis=0)
    return enderezada


def falta_alguna_ficha(enderezada, fichas, px_por_mm=PX_POR_MM):
    """Dice si queda alguna zona clara de al menos RESTO_MIN fichas sin ficha detectada encima."""
    resto = zonas_claras(enderezada, px_por_mm)
    radio = round((np.hypot(ANCHO_FICHA_MM, LARGO_FICHA_MM) / 2 + 2) * px_por_mm)
    for f in fichas:
        cv2.circle(resto, (round(f['x'] * px_por_mm), round(f['y'] * px_por_mm)), radio, 0, -1)
    contornos, _ = cv2.findContours(resto, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimo = RESTO_MIN * AREA_FICHA_MM2 * px_por_mm ** 2
    return any(cv2.contourArea(c) >= minimo for c in contornos)


def borrar_fichas(enderezada, px_por_mm=PX_POR_MM):
    """
    Devuelve la enderezada con las fichas borradas, o None si queda algo claro sin borrar.

    Se borra cada mancha clara (hasta MAX_FICHAS_JUNTAS fichas) y BORDE_BORRADO_MM alrededor, para
    llevarse también el canto y la sombra, y se rellena con el fondo de alrededor suavizado (una
    media que no cuenta lo borrado ni los ArUco). cv2.inpaint dejaba manchas con forma de ficha.
    """
    contornos, _ = cv2.findContours(zonas_claras(enderezada, px_por_mm), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    area_max = MAX_FICHAS_JUNTAS * AREA_FICHA_MM2 * px_por_mm ** 2
    borrar = np.zeros(enderezada.shape[:2], np.uint8)
    for contorno in contornos:
        if cv2.contourArea(contorno) <= area_max:
            cv2.drawContours(borrar, [contorno], -1, 255, -1)
    lado = 2 * round(BORDE_BORRADO_MM * px_por_mm) + 1
    borrar = cv2.dilate(borrar, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lado, lado))) > 0
    libre = (~borrar & fuera_de_los_aruco(enderezada.shape, px_por_mm)).astype(np.float32)
    sigma = 8 * px_por_mm
    peso = cv2.GaussianBlur(libre, (0, 0), sigma)[:, :, None]
    suave = cv2.GaussianBlur(enderezada * libre[:, :, None], (0, 0), sigma)
    fondo = enderezada.copy()
    fondo[borrar] = np.clip(suave / np.maximum(peso, 1e-3), 0, 255)[borrar]
    return None if falta_alguna_ficha(fondo, [], px_por_mm) else fondo


def recortar_fichas(enderezada, px_por_mm=PX_POR_MM):
    """
    Recorta las fichas sueltas que encuentra detectar_clasico: [{'imagen': BGR, 'alfa': 0..1}].

    alfa es la silueta clara de la ficha (madera y canto, con la letra dentro), con el borde
    suavizado para que al pegarla no se note el corte.
    """
    contornos, _ = cv2.findContours(zonas_claras(enderezada, px_por_mm), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    alto, ancho = enderezada.shape[:2]
    b = round(px_por_mm)  # 1 mm de margen alrededor de la silueta
    recortes = []
    for f in detectar_clasico(enderezada, px_por_mm):
        centro = (f['x'] * px_por_mm, f['y'] * px_por_mm)
        contorno = next((c for c in contornos if cv2.pointPolygonTest(c, centro, False) >= 0),
                        None)
        if contorno is None:
            continue
        x, y, w, h = cv2.boundingRect(contorno)
        x0, y0, x1, y1 = max(0, x - b), max(0, y - b), min(ancho, x + w + b), min(alto, y + h + b)
        mascara = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.drawContours(mascara, [contorno - [x0, y0]], -1, 255, -1)
        recortes.append({'imagen': enderezada[y0:y1, x0:x1].copy(),
                         'alfa': cv2.GaussianBlur(mascara, (3, 3), 0).astype(np.float32) / 255})
    return recortes


def pegar_fichas(fondo, recortes, n, rng, px_por_mm=PX_POR_MM):
    """
    Pega n recortes al azar sobre una copia del fondo: girados, con sombra y sin tocarse.

    Devuelve (imagen, cajas), con una caja (x1, y1, x2, y2) en píxeles por ficha pegada. Si alguna
    no cabe, se pegan menos. Sobre los ArUco no se pega nada.
    """
    imagen = fondo.astype(np.float32)
    alto, ancho = fondo.shape[:2]
    ocupado = ~fuera_de_los_aruco(fondo.shape, px_por_mm)
    hueco = round(SEPARACION_MM * px_por_mm)
    halo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * hueco + 1, 2 * hueco + 1))
    # Una sola luz para toda la imagen: las sombras caen todas hacia el mismo lado
    direccion = rng.uniform(0, 2 * np.pi)
    dx, dy = rng.uniform(0.5, 3) * px_por_mm * np.array([np.cos(direccion), np.sin(direccion)])
    oscuridad = rng.uniform(0.15, 0.5)
    mover = np.float32([[1, 0, dx], [0, 1, dy]])
    cajas = []
    for _ in range(n):
        ficha, alfa = _girar(recortes[rng.integers(len(recortes))], rng.uniform(0, 360),
                             rng.uniform(0.93, 1.07), hueco + round(SOMBRA_MM * px_por_mm))
        lado = alfa.shape[0]
        if lado >= min(alto, ancho):
            continue
        cuerpo = alfa > 0.05
        for _ in range(50):
            x0, y0 = int(rng.integers(0, ancho - lado)), int(rng.integers(0, alto - lado))
            if not ocupado[y0:y0 + lado, x0:x0 + lado][cuerpo].any():
                break
        else:
            continue
        # La sombra es la silueta movida y difuminada; cabe en el lienzo de la ficha girada
        sombra = cv2.GaussianBlur(cv2.warpAffine(alfa, mover, (lado, lado)), (0, 0), px_por_mm)
        zona = imagen[y0:y0 + lado, x0:x0 + lado]
        a = alfa[:, :, None]
        zona *= 1 - oscuridad * sombra[:, :, None]
        zona[:] = zona * (1 - a) + ficha * rng.uniform(0.85, 1.15) * a
        ocupado[y0:y0 + lado, x0:x0 + lado] |= cv2.dilate(cuerpo.astype(np.uint8), halo) > 0
        filas, columnas = np.nonzero(alfa > 0.5)
        cajas.append((x0 + columnas.min(), y0 + filas.min(),
                      x0 + columnas.max() + 1, y0 + filas.max() + 1))
    return np.clip(imagen, 0, 255).astype(np.uint8), cajas


def variar_luz(imagen, rng):
    """Cambia toda la imagen como cambia en el laboratorio: luz, color, enfoque, ruido y JPEG."""
    alto, ancho = imagen.shape[:2]
    x = imagen.astype(np.float32)
    # Degradado en una dirección al azar (sombra del robot, luz de la ventana...)
    angulo = rng.uniform(0, 2 * np.pi)
    filas, columnas = np.mgrid[0:alto, 0:ancho]
    t = np.cos(angulo) * columnas / ancho + np.sin(angulo) * filas / alto
    t = (t - t.min()) / (t.max() - t.min())
    x *= (1 - rng.uniform(0, 0.5) * t)[:, :, None]
    # Exposición (a veces quemada, como muchas fotos del 23) y balance de color
    x *= rng.uniform(0.6, 1.6) * rng.uniform(0.9, 1.1, 3)
    if rng.random() < 0.5:  # velo: poco contraste y fondo gris, como las fotos lavadas del 23
        velo = rng.uniform(0.1, 0.5)
        x = x * (1 - velo) + velo * rng.uniform(120, 230)
    if rng.random() < 0.3:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 1.5))
    elif rng.random() < 0.3:  # imagen movida
        largo = int(rng.integers(3, 10))
        nucleo = np.zeros((largo, largo), np.float32)
        nucleo[largo // 2] = 1
        giro = cv2.getRotationMatrix2D(((largo - 1) / 2, (largo - 1) / 2), rng.uniform(0, 180), 1)
        nucleo = cv2.warpAffine(nucleo, giro, (largo, largo))
        x = cv2.filter2D(x, -1, nucleo / nucleo.sum())
    x += rng.normal(0, rng.uniform(0, 6), x.shape)
    salida = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        calidad = int(rng.integers(35, 95))
        _, jpg = cv2.imencode('.jpg', salida, [cv2.IMWRITE_JPEG_QUALITY, calidad])
        salida = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    return salida


def _girar(recorte, grados, escala, borde):
    """Gira y escala un recorte en un lienzo cuadrado donde cabe entero, con `borde` px más."""
    h, w = recorte['alfa'].shape
    lado = int(np.ceil(np.hypot(h, w) * escala)) + 2 * borde
    giro = cv2.getRotationMatrix2D((w / 2, h / 2), grados, escala)
    giro[:, 2] += [lado / 2 - w / 2, lado / 2 - h / 2]
    imagen = cv2.warpAffine(recorte['imagen'], giro, (lado, lado), borderMode=cv2.BORDER_REPLICATE)
    return imagen.astype(np.float32), cv2.warpAffine(recorte['alfa'], giro, (lado, lado))
