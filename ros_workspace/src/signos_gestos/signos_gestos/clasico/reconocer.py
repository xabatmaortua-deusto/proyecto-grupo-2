"""Reconocimiento de letras por reglas (vía clásica). Tareas C7 y C8 · Responsable: Ane.

PENDIENTE DE IMPLEMENTAR. Este fichero describe lo que contendrá.

Qué hace
--------
Ofrece la función común a las dos vías:

    reconocer(frame_bgr) -> str
        Recibe un fotograma de OpenCV (BGR) y devuelve 'A', 'B', 'L', 'O', 'V', 'W' o '?'.

Por dentro:
    1. Obtiene la máscara de la mano con clasico/segmentar.py (Ainhoa).
    2. Calcula sus medidas con clasico/medidas.py.
    3. Aplica un árbol de reglas sobre esas medidas.

Árbol de reglas previsto (primera versión)
------------------------------------------
    ¿hay agujero?                  → O
    0 dedos:  alargada             → B
              compacta             → A
    2 dedos:  ángulo ≈ 90°         → L
              ángulo < 50°         → V
    3 dedos                        → W
    cualquier otro caso            → '?'   (no adivinar)

Umbrales
--------
Se eligen con las gráficas de herramientas/analisis_medidas.ipynb (tarea C6) y se ajustan SOLO con los
vídeos de entrenamiento, nunca con los de test (tarea C8). Los valores finales quedarán como constantes
al principio de este fichero, cada una con un comentario que explique de dónde sale.

Pruebas
-------
test/test_reconocer_clasico.py.
"""
