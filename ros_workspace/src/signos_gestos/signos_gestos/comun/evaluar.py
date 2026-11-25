"""Evaluación común de cualquier reconocer(frame_bgr) -> str. Tareas T1 y T2 · Responsable: Keltse.

LETRAS (T1). Pasa los vídeos de letras por un reconocedor y calcula, fotograma a fotograma:
    - el acierto por letra (y el global),
    - el porcentaje de '?' (el reconocedor no se atreve a decir nada),
    - el acierto sin contar los '?' (cuando dice algo, ¿acierta?),
    - la matriz de confusión: filas = letra real, columnas = lo que dijo el reconocedor.

Los vídeos son los limpios que guarda grabar_letras.py: datos/videos/<LETRA>/<persona>_<fecha>.mp4.
Todos los fotogramas de un vídeo tienen como etiqueta su letra, también los que no tienen mano
(cuentan como '?' si el reconocedor no dice nada).

PALABRAS (T2). Pasa cada vídeo de una palabra por el reconocedor y por la clase Palabra (la
misma que usará el nodo ROS) y compara la palabra reconocida con la real. Los vídeos van en
datos/palabras/<PALABRA>/<persona>_<fecha>.mp4: la carpeta es la palabra real. Se graban con
cualquier cámara (sin espejo) y se renombran así a mano.

Uso, desde la carpeta signos_gestos/ (la del package.xml):
    python -m signos_gestos.comun.evaluar --via rf --personas ane          # Random Forest
    python -m signos_gestos.comun.evaluar --via reglas --personas ane      # MediaPipe + reglas
    python -m signos_gestos.comun.evaluar --via clasico --personas ane     # vía clásica
    python -m signos_gestos.comun.evaluar --via rf --letras A B L O V W --carpeta otra/carpeta
    python -m signos_gestos.comun.evaluar --via rf --palabras --personas ane   # palabras (T2)

--personas elige el conjunto de test (por defecto, todos los vídeos: solo para probar).
El conjunto de test NO se usa para ajustar umbrales ni para elegir el modelo (REPARTO_TAREAS.md).

Resultados en resultados_evaluacion/<via>/:
    letras:   resumen.csv, matriz.csv y matriz.png
    palabras: palabras.csv (vídeo, persona, palabra real, reconocida y si es correcta)
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

from signos_gestos.comun.palabra import ESTABLE, HUECO, Palabra

LETRAS = ['A', 'B', 'L', 'O', 'V', 'W']   # el mínimo de CLAUDE.md
NADA = '?'
RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CARPETA_VIDEOS = os.path.join(RAIZ, 'datos', 'videos')
CARPETA_PALABRAS = os.path.join(RAIZ, 'datos', 'palabras')
SALIDA = os.path.join(RAIZ, 'resultados_evaluacion')


# ------------------------------------------------------------------ datos
def persona_de(ruta):
    """'datos/videos/A/ane_20261125-101500.mp4' -> 'ane'.

    grabar_letras.py no deja poner «_» en el nombre de la persona: lo que hay antes es la persona.
    """
    return os.path.basename(ruta).split('_')[0].lower()


def buscar_videos(carpeta, etiquetas=None, personas=None):
    """Lista de (etiqueta, ruta) de carpeta/<ETIQUETA>/*.mp4.

    La etiqueta es la letra (T1) o la palabra (T2). Sin etiquetas, todas las subcarpetas.
    Con personas, solo los vídeos de esas personas.
    """
    if etiquetas is None:
        etiquetas = sorted(d for d in os.listdir(carpeta)
                           if os.path.isdir(os.path.join(carpeta, d)))
    personas = {p.lower() for p in personas} if personas else None
    videos = []
    for etiqueta in etiquetas:
        sub = os.path.join(carpeta, etiqueta)
        if not os.path.isdir(sub):
            continue
        for nombre in sorted(os.listdir(sub)):
            ruta = os.path.join(sub, nombre)
            if not nombre.lower().endswith('.mp4'):
                continue
            if personas is None or persona_de(ruta) in personas:
                videos.append((etiqueta, ruta))
    return videos


def predecir_video(reconocer, ruta):
    """Lista con lo que devuelve reconocer() en cada fotograma del vídeo."""
    cap = cv2.VideoCapture(ruta)
    if not cap.isOpened():
        raise FileNotFoundError(f'No se pudo abrir {ruta}')
    predichas = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        predichas.append(reconocer(frame))
    cap.release()
    return predichas


# ------------------------------------------------------------------ métricas
def matriz_confusion(reales, predichas, letras):
    """Matriz de confusión y nombres de sus columnas.

    Filas: letras (la real). Columnas: letras, luego cualquier otra letra que haya dicho el
    reconocedor (p. ej. el Random Forest conoce el abecedario entero) y al final '?'.
    """
    otras = sorted({p for p in predichas if p not in letras and p != NADA})
    columnas = list(letras) + otras + [NADA]
    fila = {l: i for i, l in enumerate(letras)}
    col = {l: j for j, l in enumerate(columnas)}
    m = np.zeros((len(letras), len(columnas)), dtype=int)
    for r, p in zip(reales, predichas):
        m[fila[r], col[p if p in col else NADA]] += 1
    return m, columnas


def resumen(m, letras, columnas):
    """Métricas por letra (y una fila 'total') a partir de la matriz de confusión."""
    j_nada = columnas.index(NADA)
    filas = []
    for i, letra in enumerate(list(letras) + ['total']):
        cuenta = m.sum(axis=0) if letra == 'total' else m[i]
        n = int(cuenta.sum())
        bien = int(np.trace(m)) if letra == 'total' else int(m[i, i])
        nada = int(cuenta[j_nada])
        filas.append({
            'letra': letra,
            'fotogramas': n,
            'acierto': bien / n if n else 0.0,
            'porc_nada': nada / n if n else 0.0,
            'acierto_sin_nada': bien / (n - nada) if n > nada else 0.0,
        })
    return filas


def evaluar(crear_reconocer, videos, letras):
    """Pasa cada vídeo por un reconocedor NUEVO (los de la vía DL guardan los últimos fotogramas).

    crear_reconocer() devuelve una función reconocer(frame_bgr) -> str.
    Devuelve la matriz de confusión, sus columnas y las métricas por letra.
    """
    reales, predichas = [], []
    for letra, ruta in videos:
        p = predecir_video(crear_reconocer(), ruta)
        bien = sum(x == letra for x in p)
        print(f'  {letra}  {os.path.basename(ruta)}: {bien}/{len(p)} bien', flush=True)
        reales += [letra] * len(p)
        predichas += p
    m, columnas = matriz_confusion(reales, predichas, letras)
    return m, columnas, resumen(m, letras, columnas)


def formar_palabra(predichas, estable=ESTABLE, hueco=HUECO):
    """Pasa las letras de cada fotograma por Palabra y devuelve el texto final."""
    palabra = Palabra(estable, hueco)
    for letra in predichas:
        palabra.actualizar(letra)
    return palabra.texto


def evaluar_palabras(crear_reconocer, videos):
    """Una fila por vídeo: vídeo, persona, palabra real, reconocida y si es correcta.

    Igual que en evaluar(), cada vídeo empieza con un reconocedor nuevo.
    """
    filas = []
    for real, ruta in videos:
        reconocida = formar_palabra(predecir_video(crear_reconocer(), ruta))
        filas.append({'video': os.path.basename(ruta), 'persona': persona_de(ruta),
                      'real': real, 'reconocida': reconocida, 'correcta': reconocida == real})
        print(f'  {real:8s} {os.path.basename(ruta)}: {reconocida!r}'
              f' {"bien" if reconocida == real else "MAL"}', flush=True)
    return filas


# ------------------------------------------------------------------ guardar
def guardar(carpeta, m, columnas, filas, titulo):
    os.makedirs(carpeta, exist_ok=True)
    with open(os.path.join(carpeta, 'resumen.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]), delimiter=';')
        w.writeheader()
        w.writerows(filas)
    letras = [f['letra'] for f in filas[:-1]]
    with open(os.path.join(carpeta, 'matriz.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(['real \\ dicha'] + columnas)
        for letra, fila in zip(letras, m):
            w.writerow([letra] + list(fila))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Normalizada por filas: así se comparan letras con distinto número de fotogramas
    norm = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(columnas), 1 + 0.6 * len(letras)))
    ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    for i in range(len(letras)):
        for j in range(len(columnas)):
            if m[i, j]:
                ax.text(j, i, f'{norm[i, j]:.0%}', ha='center', va='center', fontsize=8,
                        color='white' if norm[i, j] > 0.5 else 'black')
    ax.set_xticks(range(len(columnas)), columnas)
    ax.set_yticks(range(len(letras)), letras)
    ax.set_xlabel('lo que dice el reconocedor')
    ax.set_ylabel('letra real')
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta, 'matriz.png'), dpi=120)
    plt.close(fig)


def guardar_palabras(carpeta, filas):
    os.makedirs(carpeta, exist_ok=True)
    with open(os.path.join(carpeta, 'palabras.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]), delimiter=';')
        w.writeheader()
        w.writerows(filas)


def imprimir(filas):
    print(f'\n{"letra":>6} {"fotog.":>7} {"acierto":>8} {"% ?":>6} {"acierto sin ?":>14}')
    for f in filas:
        print(f'{f["letra"]:>6} {f["fotogramas"]:>7} {f["acierto"]:>8.1%} '
              f'{f["porc_nada"]:>6.1%} {f["acierto_sin_nada"]:>14.1%}')


# ------------------------------------------------------------------ vías
def _scripts_de_la_raiz():
    """Pone la carpeta signos_gestos/ (la del package.xml) en el path de Python.

    Desde ahí se importan reconocer_letras.py y letras_comun.py, que siguen en esa carpeta
    (PROVISIONAL: cuando se muevan a signos_gestos/dl/, ESTRUCTURA.md, se importan del paquete),
    y herramientas/pruebas/prueba_mediapipe_letras.py.
    """
    if RAIZ not in sys.path:
        sys.path.insert(0, RAIZ)


def crear_rf():
    """Vía DL actual: MediaPipe + Random Forest (reconocer_letras.py)."""
    _scripts_de_la_raiz()
    from reconocer_letras import ReconocedorLetras
    return ReconocedorLetras().reconocer


def crear_reglas():
    """Referencia (T4): MediaPipe + reglas escritas a mano (prueba_mediapipe_letras.py)."""
    _scripts_de_la_raiz()
    from herramientas.pruebas.prueba_mediapipe_letras import (estado_dedos, letra_por_reglas,
                                                              medidas)
    import mediapipe as mp
    from letras_comun import crear_detector

    detector = crear_detector()
    t = {'ms': 0}

    def reconocer(frame_bgr):
        t['ms'] += 33                                     # MediaPipe exige tiempos crecientes
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        r = detector.detect_for_video(imagen, t['ms'])
        if not r.hand_world_landmarks:
            return NADA
        m = medidas(r.hand_world_landmarks[0])
        return letra_por_reglas(m, estado_dedos(m))
    return reconocer


def crear_clasico():
    """Vía clásica (clasico/reconocer.py, de Ane): no guarda nada entre fotogramas."""
    from signos_gestos.clasico.reconocer import reconocer
    return reconocer


VIAS = {'rf': crear_rf, 'reglas': crear_reglas, 'clasico': crear_clasico}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--via', choices=VIAS, required=True)
    ap.add_argument('--personas', nargs='+',
                    help='personas del conjunto de test (por defecto, todas)')
    ap.add_argument('--letras', nargs='+', default=LETRAS,
                    help=f'letras a evaluar ({" ".join(LETRAS)})')
    ap.add_argument('--palabras', action='store_true',
                    help='evaluar palabras completas (T2) en vez de letras sueltas')
    ap.add_argument('--carpeta',
                    help='carpeta con <LETRA>/<id>.mp4 o <PALABRA>/<id>.mp4 '
                         '(por defecto datos/videos o datos/palabras)')
    args = ap.parse_args()

    if args.carpeta is None:
        args.carpeta = CARPETA_PALABRAS if args.palabras else CARPETA_VIDEOS
    if not os.path.isdir(args.carpeta):
        raise SystemExit(f'No existe la carpeta {args.carpeta}')
    videos = buscar_videos(args.carpeta, None if args.palabras else args.letras, args.personas)
    if not videos:
        raise SystemExit(f'No hay vídeos en {args.carpeta}')
    if not args.personas:
        print('AVISO: sin --personas se evalúa con TODOS los vídeos, '
              'también los de entrenamiento.')
    print(f'{len(videos)} vídeos · vía {args.via}')
    carpeta = os.path.join(SALIDA, args.via)

    if args.palabras:
        filas = evaluar_palabras(VIAS[args.via], videos)
        bien = sum(f['correcta'] for f in filas)
        print(f'\nPalabras correctas: {bien}/{len(filas)} ({bien / len(filas):.0%})')
        guardar_palabras(carpeta, filas)
        print(f'Resultados en {carpeta}')
        return

    m, columnas, filas = evaluar(VIAS[args.via], videos, args.letras)
    imprimir(filas)
    personas = ', '.join(args.personas) if args.personas else 'todas'
    guardar(carpeta, m, columnas, filas, f'{args.via} · personas: {personas}')
    print(f'\nResultados en {carpeta}')


if __name__ == '__main__':
    main()
