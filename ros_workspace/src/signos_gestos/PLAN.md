# Plan de trabajo · Reconocimiento de letras en lengua de signos

Equipo 4: Ainhoa García Puebla, Keltse Santisteban Ortiz, Xabier Unzilla Higuero y Ane Aizpuru Ruiloba.

Este documento explica **qué hay hecho, cómo se hace la solución clásica ("manual") y cuáles son los próximos pasos**. El reparto de tareas está en [REPARTO_TAREAS.md](REPARTO_TAREAS.md).

---

## 1. Dónde estamos

Hay una **prueba inicial con MediaPipe**: [herramientas/pruebas/prueba_mediapipe_letras.py](herramientas/pruebas/prueba_mediapipe_letras.py).

- MediaPipe detecta la mano y devuelve **21 puntos** (muñeca y 4 por dedo).
- A partir de los puntos se calcula el **ángulo de cada dedo** y la **distancia entre las puntas**.
- **Reglas escritas a mano** convierten esas medidas en una letra, o en `?` si no encaja ninguna.
- Una **máquina de estados** acepta una letra solo si se mantiene estable unos 0,33 s, y así forma la palabra.

Resultado con los tres vídeos de prueba (los tres deletrean LOBO):

| Vídeo | Mano detectada | Palabra |
|---|---|---|
| IMG_2485 | 98 % | LOBO |
| IMG_2486 | 84 % | LLOBO (la mano sale de cuadro y repite la L) |
| IMG_2487 | 100 % | LOBO |

Por cada vídeo se generan en `resultados_mediapipe/` un vídeo anotado, una gráfica de la línea de tiempo y un CSV con las medidas.

**Cómo ejecutarlo** (desde la carpeta `signos_gestos/`, la del `package.xml`; los vídeos van en esa misma carpeta):

```powershell
python herramientas/pruebas/prueba_mediapipe_letras.py            # todos los vídeos
python herramientas/pruebas/prueba_mediapipe_letras.py --ver      # viéndolo en directo
```

**Límites de esta prueba:**
- Los umbrales están ajustados con **solo 3 vídeos, de una misma palabra**. Las reglas de A, V y W están escritas pero **sin probar**.
- MediaPipe es un **modelo entrenado**, así que **no cuenta como solución clásica**. Hay que preguntar al profesor si puede usarse como referencia o como ayuda (por ejemplo, para recortar la mano).

### 1.1 Abecedario completo: MediaPipe + clasificador

Las reglas a mano no escalan a todo el abecedario: muchas letras se parecen y cada regla nueva rompe otra. Por eso hay una segunda versión en la que **MediaPipe da los 21 puntos** y un **clasificador (Random Forest) aprende cada letra a partir de ejemplos grabados por nosotros**.

- Letras: las 30 del alfabeto dactilológico de la LSE (con CH, LL, Ñ y RR), más una clase `nada` para la mano en reposo o cambiando de letra. La lista está en `ALFABETO`, dentro de [letras_comun.py](letras_comun.py).
- Algunas letras llevan **movimiento**. Por eso no se clasifica un fotograma suelto, sino una **ventana de 16 fotogramas** (unos 0,5 s): la forma de la mano al principio y al final, y el recorrido de la muñeca y de la punta del índice. No hace falta decir a mano qué letras se mueven.
- La palabra se construye con la clase común `Palabra` (`signos_gestos/comun/palabra.py`).
- Ofrece la **interfaz común** `ReconocedorLetras().reconocer(frame_bgr)`.

| Paso | Script | Qué hace |
|---|---|---|
| 1 | [grabar_letras.py](grabar_letras.py) | Graba con la webcam unos 3 s de cada letra. Guarda los puntos (unos 25 KB) en `dataset_letras/<LETRA>/<persona>_<fecha>.npz` y **el vídeo limpio** (sin dibujos ni espejo) en `datos/videos/<LETRA>/<persona>_<fecha>.mp4`. También admite un vídeo ya grabado (`--video` y `--letra`). |
| 1b | [extraer_recortes.py](extraer_recortes.py) | Saca de cada vídeo el **recorte cuadrado de la mano** (224 px, 1 de cada 3 fotogramas) usando los puntos ya guardados, en `datos/recortes/<LETRA>/`. Tiene la estructura de `ImageFolder` y sirve para la red. |
| 2 | [entrenar_letras.py](entrenar_letras.py) | Entrena y valida **por grabación** o **por persona** (`--val-persona`), para que no haya ventanas gemelas. Saca el acierto por letra y la matriz de confusión en `resultados_letras/` y guarda `modelo_letras.joblib`. |
| 3 | [reconocer_letras.py](reconocer_letras.py) | Reconoce letras y palabras en directo con la webcam, o en vídeos (guarda el vídeo anotado). |

```powershell
..\.venvVision\Scripts\python.exe grabar_letras.py --persona ainhoa     # ESPACIO graba, d/a cambia de letra
..\.venvVision\Scripts\python.exe entrenar_letras.py                    # o --val-persona ane
..\.venvVision\Scripts\python.exe reconocer_letras.py                   # webcam (o: reconocer_letras.py IMG_2485.MOV)
```

**Dataset:** las grabaciones y los `.npz` se comparten fuera de Git, en la carpeta compartida del grupo. Objetivo: **3 o más grabaciones por letra y persona**, con alguna luz distinta. El modelo `.joblib` no se sube: cada uno lo regenera con el paso 2.

**Estado:** el equipo ha probado el reconocedor con vídeos reales del abecedario completo y confirma que funciona. Los vídeos no se incluyen en Git; se conservan fuera del repositorio. El script permite generar un CSV diagnóstico por fotograma para distinguir fallos de detección de mano, ventana incompleta y rechazo/clasificación.

El modelo `HandLandmarker` se descarga desde una ruta versionada (`float16/1`) para que las instalaciones nuevas no dependan del alias cambiante `latest`. Para revisar una ejecución, se puede añadir `--diagnostico` al reconocedor; el CSV se guarda en `resultados_letras/` e incluye detección, estado, predicción y confianza por fotograma, pero no imágenes ni identidad de personas. Los estados distinguen `sin_mano`, `ventana_incompleta`, `nada`, `confianza_baja` y `reconocida`.

**Una sola grabación sirve para las tres vías.** La misma grabación tiene el mismo nombre en las tres carpetas, y el fotograma *n* del vídeo corresponde a `puntos[n]` del `.npz`:

| Carpeta | Contenido | Quién lo usa | ¿En git? |
|---|---|---|---|
| `dataset_letras/<LETRA>/<id>.npz` | Puntos de MediaPipe | MediaPipe + Random Forest | No, por Drive (pendiente de decidir, ver `docs/ESTRUCTURA.md`) |
| `datos/videos/<LETRA>/<id>.mp4` | Vídeo limpio, como mucho 720 px de alto | Vía clásica (segmentación sobre el fotograma entero) | No, por Drive |
| `datos/recortes/<LETRA>/<id>_<n>.jpg` | Recorte de la mano | Vía deep learning | No: se regenera con `extraer_recortes.py` |

- `<id>` es `<persona>_<fecha>`, así que la persona sale del nombre de fichero (lo que hay antes del primer `_`) y se puede **separar train/val/test por persona** en todas las vías.
- Para compartir: subir al Drive del grupo `dataset_letras/` y `datos/videos/`. Los recortes no hace falta compartirlos: cada uno los regenera.

```powershell
..\.venvVision\Scripts\python.exe extraer_recortes.py                  # todas las letras
..\.venvVision\Scripts\python.exe extraer_recortes.py --letras A B L O V W --cada 5
```

Después hay que revisar a ojo `resultados_recortes/muestra.jpg`.

---

## 2. Cómo es la solución clásica ("manual")

Es la vía del apartado 3.A del hito: **solo OpenCV, sin modelos entrenados**. La diferencia con la prueba es que nadie nos da los puntos de la mano: todo sale de la **silueta**.

```
fotograma → 1. segmentar la mano → 2. quitar el brazo → 3. contorno
          → 4. medidas geométricas → 5. reglas → letra o "?" → 6. palabra
```

### 2.1 Segmentar la mano

El objetivo es una máscara en blanco y negro, con la mano en blanco.

- **Por color de piel:** convertir a **YCrCb** y quedarse con `Cr ∈ [133, 173]` y `Cb ∈ [77, 127]`. Son rangos típicos que hay que ajustar a nuestra luz. Funciona mejor con cambios de luz que RGB.
- **Alternativa con nuestro fondo:** la pared blanca casi no tiene saturación y la piel sí, así que puede bastar un umbral en el canal **S** de HSV, o Otsu.
- **Limpieza:** apertura y cierre morfológicos (`cv2.morphologyEx`) y quedarse con el **componente conexo más grande**.
- **Riesgo:** la ropa de color parecido a la piel. Hay que **grabar con manga larga oscura**.

### 2.2 Quitar el antebrazo

En nuestros vídeos el brazo entra por abajo y estropea las medidas.

- La **transformada de distancia** (`cv2.distanceTransform`) tiene su máximo en el **centro de la palma**, y ese valor es el **radio de la palma** *r*.
- Se elimina todo lo que quede a más de unos 1,5·*r* del centro, hacia el brazo.

### 2.3 Contorno

`cv2.findContours(mascara, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)`

Usamos `RETR_CCOMP` porque también devuelve los **agujeros**, y los necesitamos para la O.

### 2.4 Medidas geométricas

| Medida | Cómo se calcula | Para qué sirve |
|---|---|---|
| **Nº de dedos estirados** | `cv2.convexHull` + `cv2.convexityDefects`. Cada hueco profundo (más de 0,5·*r*) con ángulo menor de 90° está entre dos dedos, así que dedos = huecos + 1. Otra opción: contar cuántas veces el contorno corta un círculo de radio ~1,8·*r* alrededor de la palma. | A/B (0), L y V (2), W (3) |
| **Agujero interior** | Contorno hijo en la jerarquía de `RETR_CCOMP` con área suficiente | **O**: el círculo índice-pulgar deja un agujero |
| **Solidez** = área / área de la envolvente convexa | `cv2.contourArea` del contorno y de la envolvente | Puño y mano plana ≈ 0,9. Con dedos separados, menos. |
| **Alargamiento** | `cv2.minAreaRect` o `cv2.fitEllipse`: eje largo / eje corto | **B** (alargada) frente a **A** (compacta) |
| **Ángulo entre dedos** | Vectores del centro de la palma a cada punta | **L** (~90°) frente a **V** (30-50°) |

Hay que usar medidas que **no cambien al girar la mano** (número de dedos, agujeros, solidez, alargamiento), porque la B, por ejemplo, la hacemos de lado.

### 2.5 Reglas (árbol de decisión, primera versión)

```
¿hay agujero?                 → O
dedos = 0:  alargada          → B
            compacta          → A
dedos = 2:  ángulo ≈ 90°      → L
            ángulo < 50°      → V
dedos = 3                     → W
otro caso                     → ?   (no adivinar)
```

Los umbrales se eligen **con datos**, como se hizo con MediaPipe:
1. Se calculan las medidas en vídeos de cada letra.
2. Se dibujan, una gráfica por medida y color por letra.
3. El corte se pone donde se separan los grupos.

### 2.6 Construcción de la palabra

La clase `Palabra` (`signos_gestos/comun/palabra.py`, sacada de `prueba_mediapipe_letras.py`) **sirve para las dos vías**, porque solo recibe una letra por fotograma, venga del método que venga:
- Una letra se acepta si aparece **10 fotogramas seguidos** (unos 0,33 s a 30 fps).
- No se repite mientras se mantiene el gesto.
- Para escribir una letra doble hay que **soltar el gesto** al menos 8 fotogramas y volver a hacerlo.

---

## 3. Próximos pasos

1. **Grabar un dataset etiquetado.** El plan detallado (qué grabar, condiciones, paso a paso y registro de sesiones)
   está en [docs/DATASET.md](docs/DATASET.md).
   - Letras (A, B, L, O, V, W y `nada`) y palabras, hechas por **las cuatro personas** y con dos iluminaciones.
   - Cámara fija, fondo liso y manga larga oscura.
   - Es el **conjunto común** que pide el apartado 5 del hito, y lo usan los dos métodos.
2. **Separar ya el conjunto de test**, por ejemplo una persona entera o una iluminación, y no usarlo para ajustar nada hasta la evaluación final.
3. **Grabar también palabras completas** (LOBO, BOLA, OLA...) para probar la construcción de palabras.
4. **Vía clásica:** primero la segmentación, revisando las máscaras a ojo. Después medidas, gráficas, umbrales y reglas.
5. **Vía deep learning:** transfer learning partiendo del notebook de la práctica 3 ([Copy_of_Practica3_clasificador.ipynb](../Copy_of_Practica3_clasificador.ipynb)), que ya usa `ImageFolder` con una carpeta por clase.
6. **Evaluación común:**
   - Acierto por letra y **matriz de confusión** (`sklearn`).
   - Palabras reconocidas correctamente.
   - Robustez con otra persona y otra luz.
7. **Integración con Robótica:** acordar con el otro equipo cómo se envía la palabra (texto por fichero, socket, ROS...).

---

## 4. Avisos importantes

- **Nuestras letras son de la LSE (española), no de la ASL (americana).** Nuestra **O** (tres dedos arriba y el círculo índice-pulgar) es en ASL la **F**, y la O de la ASL es la mano entera curvada. Si se entrena con *ASL Alphabet* o *Sign Language MNIST*, algunas letras quedarán **mal etiquetadas** para nosotros. Antes de usar un dataset público hay que **comparar letra a letra** y quedarse solo con las que coinciden, o completar con imágenes propias. Hay que explicarlo en la memoria.
- **MediaPipe no abre ficheros si la ruta tiene tildes** (la carpeta `Visión`). Hay que leer el modelo con Python y pasarle los bytes: `BaseOptions(model_asset_buffer=open(ruta, "rb").read())`.
- **Entorno:** todo se ejecuta con `.venvVision`, que ya tiene OpenCV, MediaPipe, PyTorch (CPU), torchvision y scikit-learn. La GPU del portátil (GTX 1050) no es compatible con el PyTorch actual, así que el entrenamiento irá en CPU, o en Colab si hace falta GPU.
