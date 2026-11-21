#!/usr/bin/env python3
"""Entrena el clasificador de letras con las grabaciones de dataset_letras/ (paso 2 de 3).

Uso:
  python entrenar_letras.py                       # valida con una grabación de cada letra
  python entrenar_letras.py --val-persona ane     # valida con TODO lo de una persona (más exigente)

Qué hace:
  1. Corta cada grabación en ventanas de 16 fotogramas y saca sus características
     (forma de la mano al principio y al final + recorrido; ver letras_comun.py).
  2. Separa validación POR GRABACIÓN (o por persona): las ventanas de una misma grabación son
     casi iguales, y si unas fueran a entrenamiento y otras a validación el acierto saldría
     inflado (las «fotos gemelas» de la práctica de SAM).
  3. Entrena un Random Forest, saca el acierto por letra y la matriz de confusión.
  4. Vuelve a entrenar con TODOS los datos y guarda modelo_letras.joblib.

Resultados en resultados_letras/: informe.txt y matriz_confusion.png
"""
import argparse
import glob
import os
from collections import Counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from letras_comun import (AQUI, ALFABETO, CARPETA_DATOS, MODELO_LETRAS, NADA, VENTANA,
                          ventanas_de_grabacion)

SALIDA = os.path.join(AQUI, "resultados_letras")


def cargar():
    """Devuelve una lista de grabaciones: (letra, persona, fichero, matriz de características)."""
    grabaciones = []
    for ruta in sorted(glob.glob(os.path.join(CARPETA_DATOS, "*", "*.npz"))):
        d = np.load(ruta)
        letra = os.path.basename(os.path.dirname(ruta))
        X = ventanas_de_grabacion(d["puntos"], str(d["lado"]))
        if X:
            grabaciones.append((letra, str(d["persona"]), ruta, np.array(X)))
        else:
            print(f"Aviso: {os.path.relpath(ruta)} no tiene {VENTANA} fotogramas seguidos con mano; se ignora")
    return grabaciones


def juntar(grabaciones):
    X = np.vstack([g[3] for g in grabaciones])
    y = np.concatenate([[g[0]] * len(g[3]) for g in grabaciones])
    return X, y


def nuevo_modelo():
    return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-persona", help="usar todas las grabaciones de esta persona como validación")
    args = ap.parse_args()

    grabaciones = cargar()
    if not grabaciones:
        raise SystemExit(f"No hay grabaciones en {CARPETA_DATOS}. Graba primero con grabar_letras.py")
    os.makedirs(SALIDA, exist_ok=True)

    por_letra = Counter(g[0] for g in grabaciones)
    personas = sorted({g[1] for g in grabaciones})
    faltan = [l for l in ALFABETO + [NADA] if l not in por_letra]
    print(f"{len(grabaciones)} grabaciones · personas: {', '.join(personas)}")
    print("Grabaciones por letra: " + "  ".join(f"{l}={por_letra[l]}" for l in ALFABETO + [NADA] if l in por_letra))
    if faltan:
        print(f"Aviso: sin grabaciones de {' '.join(faltan)} -> el modelo no podrá reconocerlas")

    # ---- validación por grabación (o por persona)
    if args.val_persona:
        if args.val_persona not in personas:
            raise SystemExit(f"No hay grabaciones de {args.val_persona!r}. Personas: {', '.join(personas)}")
        val = [g for g in grabaciones if g[1] == args.val_persona]
        tr = [g for g in grabaciones if g[1] != args.val_persona]
        como = f"persona {args.val_persona}"
    else:
        # la última grabación de cada letra que tenga al menos 2 va a validación
        ultima = {}
        for g in grabaciones:
            if por_letra[g[0]] >= 2:
                ultima[g[0]] = g[2]
        val = [g for g in grabaciones if ultima.get(g[0]) == g[2]]
        tr = [g for g in grabaciones if ultima.get(g[0]) != g[2]]
        sin_val = [l for l in por_letra if por_letra[l] < 2]
        como = "una grabación de cada letra"
        if sin_val:
            print(f"Aviso: {' '.join(sin_val)} solo tienen 1 grabación: entran en entrenamiento pero no se validan")

    informe = []
    if val and tr:
        Xtr, ytr = juntar(tr)
        Xva, yva = juntar(val)
        modelo = nuevo_modelo().fit(Xtr, ytr)
        pred = modelo.predict(Xva)
        etiquetas = [l for l in ALFABETO + [NADA] if l in set(yva) | set(pred)]
        rep = classification_report(yva, pred, labels=etiquetas, digits=2, zero_division=0)
        informe += [f"Validación con {como}: {len(Xtr)} ventanas de entrenamiento, {len(Xva)} de validación",
                    f"Acierto total: {np.mean(pred == yva):.1%}", "", rep]
        print("\n" + "\n".join(informe))

        fig, ax = plt.subplots(figsize=(max(6, len(etiquetas) * 0.45),) * 2)
        ConfusionMatrixDisplay(confusion_matrix(yva, pred, labels=etiquetas), display_labels=etiquetas).plot(
            ax=ax, cmap="Oranges", colorbar=False, xticks_rotation=90)
        ax.set_title(f"Matriz de confusión (validación: {como})")
        fig.tight_layout(); fig.savefig(os.path.join(SALIDA, "matriz_confusion.png"), dpi=110); plt.close(fig)

        confusiones = Counter((r, p) for r, p in zip(yva, pred) if r != p).most_common(8)
        if confusiones:
            informe += ["Confusiones más frecuentes (real -> dijo: ventanas):"]
            informe += [f"  {r} -> {p}: {n}" for (r, p), n in confusiones]
            print("\n".join(informe[-len(confusiones) - 1:]))
    else:
        print("\nNo hay datos suficientes para separar validación (hace falta ≥2 grabaciones de alguna letra).")

    # ---- modelo final con todo
    X, y = juntar(grabaciones)
    modelo = nuevo_modelo().fit(X, y)
    joblib.dump({"modelo": modelo, "ventana": VENTANA, "letras": sorted(set(y))}, MODELO_LETRAS)
    informe += ["", f"Modelo final entrenado con {len(X)} ventanas de {len(set(y))} clases -> {os.path.basename(MODELO_LETRAS)}"]
    with open(os.path.join(SALIDA, "informe.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(informe) + "\n")
    print(f"\n{informe[-1]}\nInforme y matriz de confusión en {os.path.relpath(SALIDA)}/")


if __name__ == "__main__":
    main()
