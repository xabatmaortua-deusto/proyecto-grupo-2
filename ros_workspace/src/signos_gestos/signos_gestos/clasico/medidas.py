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
    n_dedos        Dedos separados, contando los huecos entre ellos.  → A/B 0, L/V 2, W 3
    alargamiento   Lado largo / lado corto del rectángulo de área mínima (≥ 1). → B frente a A
    solidez        Área del contorno / área de la envolvente convexa (0-1). → cerrada o abierta
    angulo_dedos   Ángulo entre los dos dedos, en grados (None si no hay 2). → L frente a V

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
import math

import cv2
import numpy as np

# Valores iniciales. Se ajustarán con las gráficas de herramientas/analisis_medidas.ipynb (paso 7).

# Un hueco interior cuenta como agujero de la O si su área es al menos esta fracción del área de la
# mano. Los huecos más pequeños son ruido de la segmentación.
FRACCION_MIN_AGUJERO = 0.02

# Un defecto de convexidad es un hueco entre dos dedos si su profundidad supera esta fracción del
# radio del círculo mínimo que encierra la mano (así no depende de la distancia a la cámara)...
FRACCION_MIN_PROFUNDIDAD = 0.25
# ...y si es estrecho: el ángulo en su punto más profundo es menor que este. Las curvas anchas son
# los lados de la palma, no huecos entre dedos. En la L el hueco índice-pulgar mide unos 105-110°
# (se mide desde la base hasta las puntas de dos dedos gruesos), así que el límite queda por
# encima, con margen. La V mide unos 35-55°.
ANGULO_MAX_HUECO = 120


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


def mayor_fraccion_agujero(agujeros, area_mano):
    """Área del agujero más grande dividida entre el área de la mano (0 si no hay agujeros).

    Es el valor en bruto con el que se decide tiene_agujero(). Lo usa el cuaderno de análisis
    para elegir FRACCION_MIN_AGUJERO.
    """
    if area_mano <= 0 or not agujeros:
        return 0.0
    return max(cv2.contourArea(a) for a in agujeros) / area_mano


def tiene_agujero(agujeros, area_mano):
    """Indica si alguno de los agujeros es lo bastante grande para ser el círculo de la O."""
    if area_mano <= 0 or not agujeros:
        return False
    return mayor_fraccion_agujero(agujeros, area_mano) >= FRACCION_MIN_AGUJERO


def solidez(contorno, area):
    """Área del contorno dividida entre el área de su envolvente convexa (de 0 a 1).

    La envolvente convexa (cv2.convexHull, tema 4) es como una goma elástica que rodea la mano.
    Con la mano cerrada (puño, A) o los dedos juntos (B) la goma queda pegada y la solidez es
    cercana a 1. Con dedos separados (L, V, W) la goma cubre los huecos entre ellos y la solidez
    baja.
    """
    area_envolvente = cv2.contourArea(cv2.convexHull(contorno))
    if area_envolvente <= 0:
        return 0.0
    return area / area_envolvente


def alargamiento(contorno):
    """Lado largo dividido entre lado corto del rectángulo de área mínima (1 o más).

    cv2.minAreaRect (tema 4) da el rectángulo girado más pequeño que encierra la mano. Se usa en
    lugar de cv2.boundingRect porque no cambia si la mano está inclinada (la B se hace de lado).
    La B (dedos juntos y estirados) es alargada; la A (puño) es compacta, cercana a 1.

    Depende de que la máscara no incluya el antebrazo: si lo incluye, todo parece alargado
    (lo recorta clasico/segmentar.py, tarea C3).
    """
    (_, _), (ancho, alto), _ = cv2.minAreaRect(contorno)
    corto, largo = sorted((ancho, alto))
    if corto <= 0:
        return 0.0
    return largo / corto


def angulo(a, b, c):
    """Ángulo en b (grados) formado por los puntos a-b-c.

    Es el mismo cálculo (producto escalar) que la función angulo() de la sesión 5 y la Práctica 5,
    aplicado a puntos del contorno en lugar de puntos de MediaPipe.
    """
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    normas = math.hypot(*v1) * math.hypot(*v2)
    if normas == 0:
        return 0.0
    coseno = (v1[0] * v2[0] + v1[1] * v2[1]) / normas
    return math.degrees(math.acos(max(-1.0, min(1.0, coseno))))


def defectos_convexidad(contorno):
    """Devuelve TODOS los defectos de convexidad del contorno, sin filtrar.

    Técnica NO vista en clase: cv2.convexityDefects. Justificación: es la continuación directa de
    cv2.convexHull (tema 4). Devuelve los tramos donde el contorno se separa de la envolvente
    convexa, con su punto más profundo y su profundidad; es decir, los posibles huecos entre
    los dedos. Ninguna medida vista en clase cuenta dedos: la solidez no sirve (test_medidas.py).

    Cada defecto es un diccionario con:
        punta_1, base, punta_2   puntos (x, y) del contorno: los dos extremos y el más profundo
        profundidad              profundidad / radio del círculo mínimo que encierra la mano
                                 (cv2.minEnclosingCircle, tema 4), para no depender de la distancia
        angulo                   ángulo en la base, en grados

    Son los valores en bruto con los que huecos_entre_dedos() decide qué es un hueco entre dedos.
    Los usa el cuaderno de análisis para elegir FRACCION_MIN_PROFUNDIDAD y ANGULO_MAX_HUECO.
    """
    if len(contorno) < 4:
        return []
    indices_envolvente = cv2.convexHull(contorno, returnPoints=False)
    if indices_envolvente is None or len(indices_envolvente) < 3:
        return []
    try:
        defectos = cv2.convexityDefects(contorno, indices_envolvente)
    except cv2.error:
        # Ocurre con contornos que se cortan a sí mismos (máscaras muy irregulares).
        return []
    if defectos is None:
        return []

    _, radio = cv2.minEnclosingCircle(contorno)
    if radio <= 0:
        return []
    # reshape: OpenCV 4 devuelve las formas (N, 1, 4) y (N, 1, 2); OpenCV 5, (N, 4) y (N, 2).
    puntos = contorno.reshape(-1, 2)
    resultado = []
    for inicio, fin, mas_profundo, profundidad in defectos.reshape(-1, 4):
        punta_1 = tuple(int(v) for v in puntos[inicio])
        punta_2 = tuple(int(v) for v in puntos[fin])
        base = tuple(int(v) for v in puntos[mas_profundo])
        resultado.append({
            'punta_1': punta_1,
            'base': base,
            'punta_2': punta_2,
            # OpenCV da la profundidad multiplicada por 256 (en coma fija).
            'profundidad': profundidad / 256.0 / radio,
            'angulo': angulo(punta_1, base, punta_2),
        })
    return resultado


def huecos_entre_dedos(contorno):
    """Devuelve los huecos entre dedos como una lista de (punta_1, base, punta_2).

    De todos los defectos de convexidad, solo cuentan los profundos y estrechos (constantes del
    principio del fichero).

    Depende de que la máscara no incluya el antebrazo: el hueco entre el pulgar de la L y la
    muñeca se cuenta como un dedo más (comprobado con manos dibujadas).
    """
    return [(d['punta_1'], d['base'], d['punta_2'])
            for d in defectos_convexidad(contorno)
            if d['profundidad'] >= FRACCION_MIN_PROFUNDIDAD and d['angulo'] < ANGULO_MAX_HUECO]


def n_dedos(huecos):
    """Número de dedos separados: huecos + 1 si hay alguno, 0 si no hay (A, B)."""
    return len(huecos) + 1 if huecos else 0


def angulo_dedos(huecos):
    """Ángulo entre los dos dedos, medido en la base del hueco que los separa.

    Solo tiene sentido con 2 dedos (un único hueco): la L ronda los 90° y la V los 30-50°.
    Con otro número de dedos devuelve None.
    """
    if len(huecos) != 1:
        return None
    punta_1, base, punta_2 = huecos[0]
    return angulo(punta_1, base, punta_2)


def medidas(mascara):
    """Calcula las medidas geométricas de la mano a partir de su máscara.

    mascara: imagen de un canal; la mano es todo lo que no es 0.
    Devuelve un diccionario con las medidas descritas al principio del fichero.
    """
    contorno, agujeros = contorno_mano(mascara)
    if contorno is None:
        return {'valida': False}

    area = cv2.contourArea(contorno)
    huecos = huecos_entre_dedos(contorno)
    return {
        'valida': True,
        'area': area,
        'tiene_agujero': tiene_agujero(agujeros, area),
        'solidez': solidez(contorno, area),
        'alargamiento': alargamiento(contorno),
        'n_dedos': n_dedos(huecos),
        'angulo_dedos': angulo_dedos(huecos),
    }
