"""Tests de signos_gestos/clasico/reconocer.py · Responsable: Ane.

PENDIENTE DE IMPLEMENTAR. Este fichero describe lo que contendrá.

Comprueba el árbol de reglas a partir de medidas inventadas, sin necesidad de imágenes:
    - con agujero                         → 'O'
    - 0 dedos y alargada / compacta       → 'B' / 'A'
    - 2 dedos con ángulo ~90° / estrecho  → 'L' / 'V'
    - 3 dedos                             → 'W'
    - medidas que no encajan en ninguna   → '?'

Y, cuando haya fotos del dataset, que reconocer(frame) acierta con unas pocas imágenes de ejemplo.

Se ejecuta con:  pytest test/test_reconocer_clasico.py
"""
