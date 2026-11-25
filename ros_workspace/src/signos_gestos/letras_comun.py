"""Piezas comunes para reconocer el abecedario completo con MediaPipe + un clasificador.

Lo usan los tres scripts:
  grabar_letras.py      graba ejemplos de cada letra (los 21 puntos de la mano, fotograma a fotograma)
  entrenar_letras.py    entrena el clasificador con esos ejemplos
  reconocer_letras.py   reconoce letras y palabras en directo (webcam) o en vídeos

Idea: con las reglas escritas a mano (herramientas/pruebas/prueba_mediapipe_letras.py) cada letra nueva necesita
umbrales nuevos y rompe las anteriores. En vez de eso, MediaPipe nos da los 21 puntos y un
CLASIFICADOR aprende de ejemplos grabados por nosotros qué forma tiene cada letra.

Como algunas letras de la LSE llevan MOVIMIENTO (no son una postura quieta), no clasificamos un
fotograma suelto sino una VENTANA de los últimos fotogramas (≈0,5 s): la forma de la mano al
principio y al final de la ventana y el recorrido de la muñeca y de la punta del índice.
En una letra quieta el recorrido es ≈0 y el clasificador se fija en la forma; en una letra con
movimiento aprende también el recorrido. Así no hay que decir a mano qué letras se mueven.
"""
import os
import ssl
import urllib.error
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

AQUI = os.path.dirname(os.path.abspath(__file__))
MODELO_MANO = os.path.join(AQUI, "hand_landmarker.task")          # el mismo de la prueba inicial
URL_MODELO_MANO = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                   "hand_landmarker/float16/1/hand_landmarker.task")
CARPETA_DATOS = os.path.join(AQUI, "dataset_letras")              # una subcarpeta por letra
MODELO_LETRAS = os.path.join(AQUI, "modelo_letras.joblib")        # lo crea entrenar_letras.py

# Lo que pesa (vídeos e imágenes) va a datos/, que NO se sube a git: se comparte por Drive.
# Cada grabación tiene el mismo nombre en las tres carpetas:
#   dataset_letras/<LETRA>/<id>.npz      puntos de la mano (git)
#   datos/videos/<LETRA>/<id>.mp4        vídeo limpio, sin dibujos ni espejo (vía clásica)
#   datos/recortes/<LETRA>/<id>_<n>.jpg  recorte cuadrado de la mano (vía deep learning)
# El fotograma n del vídeo corresponde a puntos[n] del .npz.
CARPETA_VIDEOS = os.path.join(AQUI, "datos", "videos")
CARPETA_RECORTES = os.path.join(AQUI, "datos", "recortes")
ALTO_MAX = 720         # los fotogramas se reducen a 720 px de alto como mucho (los del móvil son 4K)

# Alfabeto dactilológico de la LSE. Las letras dobles (CH, LL, RR) tienen su propio signo.
# Si el profesor pide otro conjunto, basta con cambiar esta lista.
ALFABETO = ["A", "B", "C", "CH", "D", "E", "F", "G", "H", "I", "J", "K", "L", "LL", "M",
            "N", "Ñ", "O", "P", "Q", "R", "RR", "S", "T", "U", "V", "W", "X", "Y", "Z"]
NADA = "nada"          # clase extra: mano en reposo o pasando de una letra a otra -> «?»

VENTANA = 16           # fotogramas por ejemplo (≈0,5 s a 30 fps)
PASO = 2               # al entrenar, una ventana nueva cada 2 fotogramas de la grabación
PUNTOS_RECORRIDO = 6   # el recorrido de la ventana se resume en 6 posiciones

CONEXIONES = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


# ------------------------------------------------------------------ MediaPipe
def descargar_modelo_mano():
    """Descarga hand_landmarker.task. El Python de python.org en macOS a veces no tiene los
    certificados SSL instalados: en ese caso lo intentamos con los de certifi."""
    print("Descargando el modelo de mano de MediaPipe...")
    try:
        datos = urllib.request.urlopen(URL_MODELO_MANO, timeout=60).read()
    except urllib.error.URLError as e:
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
            datos = urllib.request.urlopen(URL_MODELO_MANO, timeout=60, context=ctx).read()
        except Exception:
            raise SystemExit(f"No se pudo descargar el modelo ({e}).\nDescárgalo a mano de\n  {URL_MODELO_MANO}\n"
                             f"y guárdalo como {MODELO_MANO}")
    with open(MODELO_MANO, "wb") as f:
        f.write(datos)


def crear_detector():
    """HandLandmarker en modo vídeo, una sola mano (igual que en la prueba inicial)."""
    if not os.path.exists(MODELO_MANO):
        descargar_modelo_mano()
    opciones = vision.HandLandmarkerOptions(
        # Le pasamos los bytes: MediaPipe no abre rutas con tildes en Windows (la carpeta «Visión»)
        base_options=mpt.BaseOptions(model_asset_buffer=open(MODELO_MANO, "rb").read()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5, min_tracking_confidence=0.5)
    return vision.HandLandmarker.create_from_options(opciones)


def detectar(detector, frame_bgr, t_ms):
    """Devuelve (puntos, lado, mano_img) o (None, None, None) si no hay mano.

    puntos: array (21, 3) en PÍXELES (x, y, z). Usamos píxeles y no las coordenadas 0-1 de
    MediaPipe porque esas están divididas por el ancho y el alto por separado y deforman la mano
    si la imagen no es cuadrada.
    """
    H, W = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)                  # MediaPipe quiere RGB
    r = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t_ms))
    if not r.hand_landmarks:
        return None, None, None
    lm = r.hand_landmarks[0]
    puntos = np.array([[q.x * W, q.y * H, q.z * W] for q in lm], dtype=np.float32)
    return puntos, r.handedness[0][0].category_name, lm


def texto(img, s, pos, escala, color=(255, 255, 255), grosor=2):
    """putText con borde negro (se lee sobre cualquier fondo).

    Las fuentes de OpenCV solo tienen ASCII: la Ñ se escribe «N~» y el resto de tildes se quitan.
    """
    s = s.replace("Ñ", "N~").replace("ñ", "n~").replace("·", "-")
    s = s.translate(str.maketrans("áéíóúÁÉÍÓÚ¿¡", "aeiouAEIOU  "))
    grosor = max(1, int(grosor))
    cv2.putText(img, s, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), grosor + 3, cv2.LINE_AA)
    cv2.putText(img, s, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


def dibujar_mano(frame, puntos):
    """Esqueleto de la mano sobre el fotograma (puntos en píxeles)."""
    if puntos is None:
        return
    grosor = max(2, frame.shape[0] // 240)
    px = [(int(x), int(y)) for x, y, _ in puntos]
    for a, b in CONEXIONES:
        cv2.line(frame, px[a], px[b], (46, 105, 217), grosor, cv2.LINE_AA)
    for p in px:
        cv2.circle(frame, p, grosor + 2, (54, 42, 30), -1, cv2.LINE_AA)


# ------------------------------------------------------------------ vídeo y recortes
def reducir(frame, alto_max=ALTO_MAX):
    """Reduce el fotograma a alto_max píxeles de alto como mucho (sin deformarlo)."""
    H, W = frame.shape[:2]
    if H <= alto_max:
        return frame
    return cv2.resize(frame, (round(W * alto_max / H), alto_max), interpolation=cv2.INTER_AREA)


def recorte_mano(frame_bgr, puntos, tam=224, margen=0.25):
    """Recorte CUADRADO de la mano, de tam x tam píxeles, a partir de los 21 puntos en píxeles.

    El cuadrado rodea los puntos y deja un margen alrededor (0,25 = un 25 % del tamaño de la mano
    por cada lado), porque los puntos son articulaciones y el borde de la palma queda por fuera.
    Si la mano está cerca del borde de la imagen, lo que falta se rellena de negro: así la mano
    no se deforma ni se descentra.
    """
    H, W = frame_bgr.shape[:2]
    (x0, y0), (x1, y1) = puntos[:, :2].min(0), puntos[:, :2].max(0)
    L = int(round(max(x1 - x0, y1 - y0) * (1 + 2 * margen)))
    L = max(L, 8)
    xi = int(round((x0 + x1) / 2 - L / 2))
    yi = int(round((y0 + y1) / 2 - L / 2))
    borde = max(0, -xi, -yi, xi + L - W, yi + L - H)
    if borde:
        frame_bgr = cv2.copyMakeBorder(frame_bgr, borde, borde, borde, borde, cv2.BORDER_CONSTANT, value=0)
        xi, yi = xi + borde, yi + borde
    recorte = frame_bgr[yi:yi + L, xi:xi + L]
    return cv2.resize(recorte, (tam, tam), interpolation=cv2.INTER_AREA if L > tam else cv2.INTER_LINEAR)


# ------------------------------------------------------------------ características
def tamano_palma(puntos):
    """Muñeca -> base del corazón, en el plano de la imagen: la escala de la mano."""
    return float(np.linalg.norm(puntos[9, :2] - puntos[0, :2])) + 1e-6


def forma(puntos, lado):
    """Forma de la mano: los 21 puntos respecto a la muñeca y divididos por el tamaño de la palma.

    No depende de dónde esté la mano ni de lo lejos que esté de la cámara, pero SÍ de su
    orientación (hay letras que solo se diferencian en hacia dónde apuntan los dedos).
    La mano izquierda se pasa a «derecha» como en un espejo, para que sirvan las dos.
    """
    p = (puntos - puntos[0]) / tamano_palma(puntos)
    if lado == "Left":
        p = p * np.array([-1, 1, 1], dtype=np.float32)
    return p


def caracteristicas(ventana, lado):
    """Vector de números de una ventana (T, 21, 3) de puntos en píxeles.

    - forma de la mano en el ÚLTIMO fotograma (63 números)
    - forma de la mano en el PRIMER fotograma (63): si la mano cambia de forma, se nota aquí
    - RECORRIDO de la muñeca y de la punta del índice (6 posiciones x 2 puntos x 2 coordenadas),
      respecto a la muñeca del primer fotograma y en tamaños de palma
    """
    ventana = np.asarray(ventana, dtype=np.float32)
    escala = np.mean([tamano_palma(p) for p in ventana])
    idx = np.linspace(0, len(ventana) - 1, PUNTOS_RECORRIDO).round().astype(int)
    recorrido = (ventana[idx][:, [0, 8], :2] - ventana[0, 0, :2]) / escala
    if lado == "Left":
        recorrido = recorrido * np.array([-1, 1], dtype=np.float32)
    return np.concatenate([forma(ventana[-1], lado).ravel(),
                           forma(ventana[0], lado).ravel(),
                           recorrido.ravel()])


def ventanas_de_grabacion(puntos, lado):
    """Corta una grabación (n, 21, 3) en ventanas de VENTANA fotogramas, una cada PASO.

    Los fotogramas sin mano están guardados como NaN: las ventanas que tengan alguno se saltan.
    """
    con_mano = ~np.isnan(puntos).any(axis=(1, 2))
    return [caracteristicas(puntos[i:i + VENTANA], lado)
            for i in range(0, len(puntos) - VENTANA + 1, PASO)
            if con_mano[i:i + VENTANA].all()]
