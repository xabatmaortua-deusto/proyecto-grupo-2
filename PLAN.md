# Plan de trabajo · Reconocimiento de letras en lengua de signos

Equipo 4: Ainhoa García Puebla, Keltse Santisteban Ortiz, Xabier Unzilla Higuero y Ane Aizpuru Ruiloba.

Este documento explica **qué hay hecho, cómo se hace la solución clásica ("manual") y cuáles son los próximos pasos**. El reparto de tareas está en [REPARTO_TAREAS.md](REPARTO_TAREAS.md).

---

## 1. Dónde estamos

Hay una **prueba inicial con MediaPipe**: [prueba_mediapipe_letras.py](prueba_mediapipe_letras.py).

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

**Cómo ejecutarlo** (desde la carpeta `Proyecto`):

```powershell
..\.venvVision\Scripts\python.exe prueba_mediapipe_letras.py            # todos los vídeos
..\.venvVision\Scripts\python.exe prueba_mediapipe_letras.py --ver      # viéndolo en directo
```

**Límites de esta prueba:**
- Los umbrales están ajustados con **solo 3 vídeos, de una misma palabra**. Las reglas de A, V y W están escritas pero **sin probar**.
- MediaPipe es un **modelo entrenado**, así que **no cuenta como solución clásica**. Hay que preguntar al profesor si puede usarse como referencia o como ayuda (por ejemplo, para recortar la mano).

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

La clase `Palabra` de `prueba_mediapipe_letras.py` **se reutiliza tal cual**, porque solo recibe una letra por fotograma, venga del método que venga:
- Una letra se acepta si aparece **10 fotogramas seguidos** (unos 0,33 s a 30 fps).
- No se repite mientras se mantiene el gesto.
- Para escribir una letra doble hay que **soltar el gesto** al menos 8 fotogramas y volver a hacerlo.

---

## 3. Próximos pasos

1. **Grabar un dataset etiquetado.**
   - **Un vídeo por letra** (A, B, L, O, V, W), hecho por **las cuatro personas** del grupo y con un par de iluminaciones distintas.
   - Cámara fija, fondo liso y manga larga oscura.
   - De cada vídeo se extraen fotogramas a carpetas `datos/A/`, `datos/B/`, etc.
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
