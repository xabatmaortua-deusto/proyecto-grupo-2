#!/usr/bin/env python3
"""Saca recortes de la mano de las grabaciones, para entrenar la red (vía deep learning).

Usa lo que guarda grabar_letras.py: el vídeo limpio (datos/videos/<LETRA>/<id>.mp4) y los puntos
de MediaPipe (dataset_letras/<LETRA>/<id>.npz). No vuelve a pasar MediaPipe: el fotograma n del
vídeo corresponde a puntos[n], así que el recorte sale directamente de los puntos guardados.

Uso:
  python extraer_recortes.py                    # todas las grabaciones, 1 de cada 3 fotogramas
  python extraer_recortes.py --cada 5 --tam 160 # menos imágenes y más pequeñas
  python extraer_recortes.py --letras A B L O V W
  python extraer_recortes.py --rehacer          # borra los recortes que haya y los vuelve a sacar

Resultado (estructura de ImageFolder, una carpeta por letra; datos/ NO va a git):
  datos/recortes/<LETRA>/<persona>_<fecha>_<fotograma>.jpg
  resultados_recortes/muestra.jpg    un recorte de cada letra, para revisarlos a ojo

La persona va delante del primer «_» del nombre: con eso se separa train/val/test POR PERSONA.
Se saltan los fotogramas en los que MediaPipe no vio la mano. Con --cada 1 saldrían fotos casi
iguales seguidas (30 por segundo), que no aportan nada a la red: por defecto se coge 1 de cada 3.
"""
import argparse
import glob
import os
import shutil
from collections import Counter

import cv2
import numpy as np

from letras_comun import AQUI, ALFABETO, CARPETA_DATOS, CARPETA_RECORTES, NADA, recorte_mano, texto

SALIDA = os.path.join(AQUI, "resultados_recortes")


def ruta_video(npz, d):
    """El vídeo de una grabación: el que dice el .npz, o el del mismo nombre en datos/videos/."""
    if "video" in d.files and str(d["video"]):
        return os.path.join(AQUI, *str(d["video"]).split("/"))
    letra = os.path.basename(os.path.dirname(npz))
    return os.path.join(AQUI, "datos", "videos", letra, os.path.splitext(os.path.basename(npz))[0] + ".mp4")


def extraer(npz, args):
    """Saca los recortes de una grabación. Devuelve cuántos ha guardado (None si no hay vídeo)."""
    d = np.load(npz)
    puntos = d["puntos"]
    video = ruta_video(npz, d)
    if not os.path.exists(video):
        return None
    letra = os.path.basename(os.path.dirname(npz))
    ident = os.path.splitext(os.path.basename(npz))[0]
    carpeta = os.path.join(CARPETA_RECORTES, letra)
    os.makedirs(carpeta, exist_ok=True)

    cap = cv2.VideoCapture(video)
    n = guardados = 0
    while True:
        ok, frame = cap.read()
        if not ok or n >= len(puntos):
            break
        p = puntos[n]
        if n % args.cada == 0 and not np.isnan(p).any():
            destino = os.path.join(carpeta, f"{ident}_{n:04d}.jpg")
            if not os.path.exists(destino):
                cv2.imwrite(destino, recorte_mano(frame, p, args.tam, args.margen),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
            guardados += 1
        n += 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if abs(total - len(puntos)) > 1:
        # si no cuadran, los puntos no corresponden a los fotogramas y los recortes salen movidos
        print(f"  Aviso: {os.path.relpath(video, AQUI)} tiene {total} fotogramas y el .npz {len(puntos)}")
    return guardados


def muestra(letras, tam=160):
    """Mosaico con un recorte de cada letra (el del medio de su carpeta), para revisarlos a ojo."""
    fotos = []
    for letra in letras:
        ficheros = sorted(glob.glob(os.path.join(CARPETA_RECORTES, letra, "*.jpg")))
        if not ficheros:
            continue
        img = cv2.resize(cv2.imread(ficheros[len(ficheros) // 2]), (tam, tam))
        texto(img, letra, (6, 30), 0.9, (255, 255, 255), 2)
        fotos.append(img)
    if not fotos:
        return None
    columnas = min(8, len(fotos))
    fotos += [np.zeros_like(fotos[0])] * (-len(fotos) % columnas)
    filas = [np.hstack(fotos[i:i + columnas]) for i in range(0, len(fotos), columnas)]
    os.makedirs(SALIDA, exist_ok=True)
    ruta = os.path.join(SALIDA, "muestra.jpg")
    cv2.imwrite(ruta, np.vstack(filas))
    return ruta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cada", type=int, default=3, help="coger 1 de cada N fotogramas (3)")
    ap.add_argument("--tam", type=int, default=224, help="lado del recorte en píxeles (224)")
    ap.add_argument("--margen", type=float, default=0.25, help="margen alrededor de la mano (0.25)")
    ap.add_argument("--letras", nargs="+", help="solo estas letras (por defecto, todas)")
    ap.add_argument("--rehacer", action="store_true", help="borrar los recortes de esas letras y sacarlos de nuevo")
    args = ap.parse_args()

    letras = args.letras or ALFABETO + [NADA]
    ficheros = [f for l in letras for f in sorted(glob.glob(os.path.join(CARPETA_DATOS, l, "*.npz")))]
    if not ficheros:
        raise SystemExit(f"No hay grabaciones en {CARPETA_DATOS}. Graba primero con grabar_letras.py")
    if args.rehacer:
        for l in letras:
            shutil.rmtree(os.path.join(CARPETA_RECORTES, l), ignore_errors=True)

    por_letra, sin_video = Counter(), []
    for npz in ficheros:
        r = extraer(npz, args)
        if r is None:
            sin_video.append(os.path.relpath(npz, AQUI))
        else:
            por_letra[os.path.basename(os.path.dirname(npz))] += r

    print(f"Recortes de {args.tam}x{args.tam} px (1 de cada {args.cada} fotogramas) en "
          f"{os.path.relpath(CARPETA_RECORTES, AQUI)}/:")
    print("  " + "  ".join(f"{l}={por_letra[l]}" for l in letras if l in por_letra))
    if sin_video:
        print(f"Sin vídeo ({len(sin_video)} grabaciones; se grabaron con --sin-video o falta bajarlo de Drive):")
        print("  " + "\n  ".join(sin_video))
    m = muestra([l for l in letras if l in por_letra])
    if m:
        print(f"Muestra para revisar a ojo: {os.path.relpath(m, AQUI)}")


if __name__ == "__main__":
    main()
