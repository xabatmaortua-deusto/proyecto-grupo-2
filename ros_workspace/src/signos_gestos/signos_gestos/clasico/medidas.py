"""Medidas geométricas de la silueta de la mano (vía clásica). Tarea C5 · Responsable: Ane.

PENDIENTE DE IMPLEMENTAR. Este fichero describe lo que contendrá.

Qué hace
--------
Recibe la máscara de la mano (imagen en blanco y negro: mano en blanco, fondo en negro), que genera
clasico/segmentar.py, y la resume en unos pocos números que describen su forma. Con esos números,
clasico/reconocer.py decide qué letra es. Este fichero NO segmenta la mano ni decide la letra.

Función principal prevista
--------------------------
    medidas(mascara) -> dict

Medidas que devolverá
---------------------
Todas son relativas (cocientes o ángulos), para que no dependan de la distancia a la cámara.

    tiene_agujero   Hay un hueco interior grande (el círculo índice-pulgar).       → O
    n_dedos         Número de dedos estirados, contando los huecos entre ellos.     → A/B (0), L/V (2), W (3)
    alargamiento    Lado largo / lado corto del rectángulo de área mínima.          → B (alargada) frente a A (compacta)
    solidez         Área del contorno / área de su envolvente convexa.              → comprobación extra (mano cerrada o abierta)
    angulo_dedos    Ángulo entre las puntas de los dos dedos, desde el centro.      → L (~90°) frente a V (~30-50°)
    area            Área del contorno, en píxeles.                                   → descartar ruido o manos demasiado pequeñas

Técnicas
--------
Vistas en clase (tema 4 y sesión 5):
    cv2.findContours con RETR_TREE (contorno exterior y agujeros), cv2.contourArea, cv2.convexHull,
    cv2.minAreaRect y el cálculo de ángulos entre puntos de la sesión de MediaPipe.

No vista en clase (justificada):
    cv2.convexityDefects: para contar los dedos. Es la continuación directa de cv2.convexHull: devuelve
    los puntos del contorno más alejados de la envolvente, es decir, los huecos entre los dedos.

Pruebas
-------
test/test_medidas.py comprueba cada medida con máscaras sencillas dibujadas con código.
"""
