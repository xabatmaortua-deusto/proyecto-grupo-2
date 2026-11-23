# Estructura de `signos_gestos` para la entrega

Equipo 4 (gestos): Ainhoa, Ane, Keltse y Xabier.

Este documento describe cómo tiene que quedar **nuestra carpeta** (`ros_workspace/src/signos_gestos/`) el día de la entrega.
Es una propuesta: si alguien no está de acuerdo, lo hablamos y se cambia aquí antes de mover nada.

> Todo lo nuestro de visión vive en esta carpeta. Fuera de ella solo tocamos los archivos compartidos con el resto del grupo
> (`requirements.txt` de la raíz, `signos_interfaces`, `CLAUDE.md`), y siempre con un pull request.

---

## Árbol final

```
signos_gestos/
├── package.xml                  # ROS: descripción del paquete (se añade en la Fase 0)
├── setup.py, setup.cfg          # ROS: instalación del paquete Python
├── resource/signos_gestos       # ROS: fichero vacío obligatorio de ament_python
├── .gitignore
│
├── docs/                        # Notas, decisiones y documentación del equipo
│   ├── ESTRUCTURA.md            # Este documento
│   ├── PLAN.md                  # Plan técnico (movido desde la raíz de la carpeta)
│   ├── REPARTO_TAREAS.md        # Reparto de tareas (movido desde la raíz de la carpeta)
│   └── figuras/                 # Imágenes para la memoria: máscaras, gráficas, matrices de confusión
│
├── signos_gestos/               # El código. Se llama igual que el paquete porque ROS lo exige
│   ├── __init__.py
│   ├── comun/                   # Lo que usan las DOS vías
│   │   ├── __init__.py
│   │   ├── palabra.py           # Clase Palabra: letras estables → palabra
│   │   └── evaluar.py           # Acierto por letra, matriz de confusión y % de '?' de cualquier reconocer()
│   ├── clasico/                 # Vía clásica: solo OpenCV, sin modelos entrenados
│   │   ├── __init__.py
│   │   ├── segmentar.py         # Máscara de la mano y recorte del antebrazo
│   │   ├── medidas.py           # Medidas geométricas de la máscara
│   │   └── reconocer.py         # Reglas → reconocer(frame_bgr)
│   ├── dl/                      # Vía deep learning: MediaPipe + clasificador
│   │   ├── __init__.py
│   │   ├── mano.py              # Detector de MediaPipe y características (hoy letras_comun.py)
│   │   └── reconocer.py         # ReconocedorLetras → reconocer(frame_bgr)
│   └── nodo_gestos.py           # Nodo ROS: cámara → reconocer (clasico|dl) → Palabra → /signos/palabra
│
├── herramientas/                # Scripts que se ejecutan a mano, fuera de ROS
│   ├── grabar_letras.py         # Grabar ejemplos de cada letra
│   ├── entrenar_letras.py       # Entrenar el clasificador de la vía DL
│   ├── analisis_medidas.ipynb   # Gráficas de medidas por letra para elegir umbrales (vía clásica)
│   └── pruebas/                 # Pruebas iniciales que queremos conservar (prueba_mediapipe_letras.py, Práctica 5)
│
├── dataset_letras/              # Puntos de MediaPipe en .npz (pendiente de decidir si va en git, ver abajo)
│
└── test/                        # Tests con pytest (los pide CLAUDE.md para toda lógica no trivial)
    ├── test_palabra.py
    ├── test_medidas.py
    └── test_reconocer_clasico.py
```

---

## La regla que une todo: misma función en las dos vías

Las dos vías ofrecen **exactamente la misma función**:

```python
def reconocer(frame_bgr) -> str:
    """Recibe un fotograma de OpenCV (BGR) y devuelve 'A', 'B', 'L', 'O', 'V', 'W' o '?'."""
```

- Vía clásica: `signos_gestos/clasico/reconocer.py`
- Vía DL: `signos_gestos/dl/reconocer.py`

Gracias a eso, `comun/palabra.py`, `comun/evaluar.py` y el nodo ROS funcionan con cualquiera de las dos sin cambiar nada.
El nodo elige la vía con el parámetro `metodo:=clasico|dl`.

Cómo se importa desde cualquier sitio:

```python
from signos_gestos.clasico.medidas import medidas
from signos_gestos.comun.palabra import Palabra
```

---

## De dónde sale cada archivo

| Hoy (después de integrar TrabajoVision) | En la entrega | Responsable | Estado |
|---|---|---|---|
| `PLAN.md`, `REPARTO_TAREAS.md` | `docs/` | — | Mover |
| Clase `Palabra` dentro de `prueba_mediapipe_letras.py` | `signos_gestos/comun/palabra.py` | Keltse (autora) | Mover. Así la vía clásica no depende de MediaPipe |
| — | `signos_gestos/comun/evaluar.py` | Keltse (T1) | Por hacer |
| — | `signos_gestos/clasico/segmentar.py` | Ainhoa (C1-C4) | Por hacer |
| — | `signos_gestos/clasico/medidas.py` | Ane (C5) | Por hacer |
| — | `signos_gestos/clasico/reconocer.py` | Ane (C7-C8) | Por hacer |
| `letras_comun.py` | `signos_gestos/dl/mano.py` | Ainhoa | Mover y renombrar: pese al nombre, solo lo usa la vía DL |
| `reconocer_letras.py` (clase `ReconocedorLetras`) | `signos_gestos/dl/reconocer.py` | Ainhoa | Mover. La parte de webcam/vídeo puede quedarse como herramienta |
| `grabar_letras.py`, `entrenar_letras.py` | `herramientas/` | Ainhoa | Mover |
| — | `herramientas/analisis_medidas.ipynb` | Ane (C6) | Por hacer |
| `prueba_mediapipe_letras.py`, `Practica5_reto_mediapipe.ipynb` | `herramientas/pruebas/` | Keltse | Mover |
| — | `signos_gestos/nodo_gestos.py` | Xabier (T5-T6, A5) | Por hacer |
| `requirements.txt` | Se junta con el `requirements.txt` de la raíz del repo | Quien haga la Fase 0 | Por hacer |

**Al mover archivos, usad `git mv`**, nunca copiar y borrar: así se conserva el historial de quién escribió cada línea.

---

## Qué va en git y qué no

| Sí va en git | No va en git (Drive del grupo) |
|---|---|
| Código, tests, notebooks sin salidas pesadas | Vídeos (`*.mp4`, `*.MOV`, `*.avi`) |
| `docs/` y `docs/figuras/` (solo las que salgan en la memoria) | Modelos (`*.joblib`, `*.pt`, `*.task`) |
| | Carpetas de resultados (`resultados_*/`) |

En el repo solo se documenta **dónde están** en el Drive y **cómo se generan**.

---

## Técnicas: lo visto en clase tiene que prevalecer

Se pueden usar técnicas no vistas en clase si se justifica su uso y son minoría.
Cada vez que usemos una, la apuntamos en un comentario del código y en la memoria. Hasta ahora:

| Técnica no vista en clase | Dónde | Por qué |
|---|---|---|
| `cv2.convexityDefects` | `clasico/medidas.py` | Contar dedos: es la continuación directa de `convexHull`, que sí se vio |
| Random Forest (sklearn) | vía DL | Clasificar los puntos de MediaPipe. Alternativa vista en clase: CNN con transfer learning (prácticas 2 y 3) |

Visto en clase y útil para nosotros: YCrCb para la piel (Práctica 4), `findContours` con `RETR_TREE`,
`convexHull`, `minAreaRect`, `approxPolyDP`, Canny, ángulos entre puntos (sesión 5 de MediaPipe)
y la evaluación con `classification_report` y matriz de confusión (Práctica 3).

---

## Decisiones pendientes

- [ ] ¿`dataset_letras/*.npz` va en git (como decía TrabajoVision) o en el Drive (como dice `CLAUDE.md`)? Hay que hablarlo con todo el grupo.
- [ ] Mover `Palabra` a `comun/` (lo hace Keltse o le pedimos permiso).
- [ ] `CLAUDE.md` dice que las herramientas van en `herramientas/` de la raíz del repo; nosotros las dejamos dentro de `signos_gestos/` para tenerlo todo junto. Avisar al grupo y actualizar `CLAUDE.md` con un pull request.
