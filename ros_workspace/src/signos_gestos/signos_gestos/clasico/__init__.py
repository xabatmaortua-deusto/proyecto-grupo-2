"""Vía clásica: reconocimiento de letras solo con OpenCV, sin modelos entrenados.

Cadena de procesado:
    fotograma → segmentar.py (máscara de la mano) → medidas.py (medidas geométricas)
              → reconocer.py (reglas) → letra o '?'

Responsables: Ainhoa (segmentar.py) y Ane (medidas.py y reconocer.py).
"""
