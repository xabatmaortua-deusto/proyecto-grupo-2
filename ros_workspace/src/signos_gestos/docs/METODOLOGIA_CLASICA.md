# Metodología de la vía clásica

Reconocimiento de letras del alfabeto dactilológico (LSE) **solo con OpenCV, sin modelos entrenados**.
Este documento explica qué pasos sigue la vía clásica, cuáles están hechos y por qué se ha elegido cada técnica,
para que quien retome el trabajo sepa por dónde seguir.

Estado a 24/11/2026. Actualizad las marcas de estado cada vez que se termine un paso.

| Marca | Significado |
|---|---|
| ✅ | Hecho y con tests |
| 🔲 | Pendiente |
| ⚠️ | Pendiente y con una decisión que tomar antes de empezar |

---

## 1. Objetivo y condiciones

- **Letras:** A, B, L, O, V y W. Se eligieron porque sus siluetas son claramente distintas y se pueden separar con
  reglas geométricas sencillas (hito 1 de Visión).
- **Entrada:** un fotograma en color de OpenCV (BGR).
- **Salida:** la letra, o `'?'` si no encaja claramente en ninguna. **Es preferible no contestar a adivinar**: una
  letra equivocada acaba con una ficha mal colocada por el robot.
- **Interfaz común con la vía de deep learning:** las dos vías ofrecen la misma función, para que la construcción de
  palabras, la evaluación y el nodo ROS funcionen con cualquiera de ellas:

  ```python
  def reconocer(frame_bgr) -> str:   # 'A', 'B', 'L', 'O', 'V', 'W' o '?'
  ```

- **Condiciones de captura:** cámara fija, fondo liso y claro, **manga larga oscura** y la mano de frente a la cámara.
  Estas condiciones simplifican mucho la segmentación y son habituales en soluciones clásicas; hay que mencionarlas
  como limitación en la memoria.

### Criterio sobre las técnicas

Se pueden usar técnicas no vistas en clase si se justifica su uso y son minoría: **tiene que prevalecer lo visto en
clase**. Cada técnica no vista se justifica en el código (comentario) y en la memoria. La tabla del apartado 4 lleva la
cuenta.

---

## 2. La cadena de procesado

```
fotograma ──► 1. segmentar ──► máscara ──► 2-6. medidas ──► 8. reglas ──► letra o '?'
                                               │
                                               └──► 7. análisis (elige los umbrales de las reglas)
```

| Paso | Qué hace | Fichero | Estado |
|---|---|---|---|
| 0 | Dataset de imágenes etiquetadas, separado en entrenamiento y test | Drive del grupo | 🔲 |
| 1 | Segmentación: de la foto a la máscara de la mano, sin antebrazo | `signos_gestos/clasico/segmentar.py` | ⚠️ |
| 2 | Contorno de la mano y agujero interior | `signos_gestos/clasico/medidas.py` | ✅ |
| 3 | Solidez | `signos_gestos/clasico/medidas.py` | ✅ |
| 4 | Alargamiento | `signos_gestos/clasico/medidas.py` | ✅ |
| 5 | Número de dedos | `signos_gestos/clasico/medidas.py` | 🔲 |
| 6 | Ángulo entre dedos | `signos_gestos/clasico/medidas.py` | 🔲 |
| 7 | Análisis de las medidas por letra para elegir umbrales | `herramientas/analisis_medidas.ipynb` | 🔲 |
| 8 | Reglas de decisión | `signos_gestos/clasico/reconocer.py` | 🔲 |
| 9 | Evaluación y comparación con la vía DL | (función común, por definir) | 🔲 |

La **máscara** es el punto de unión entre la segmentación y las medidas. Formato acordado: imagen de un canal, mano en
blanco (cualquier valor distinto de 0) y fondo en negro, **sin antebrazo**. `medidas.py` ya espera este formato, así
que los dos lados se pueden desarrollar por separado.

---

## 3. Los pasos, uno a uno

### Paso 0 · Dataset 🔲

- Fotos o vídeos de las 6 letras, hechos por **varias personas** y con **al menos dos iluminaciones**, en las
  condiciones de captura del apartado 1. También algunas palabras completas (LOBO, BOLA, OLA) para probar la
  construcción de palabras.
- **Separar el conjunto de test desde el principio**, por ejemplo una persona entera, y no usarlo para ajustar nada
  hasta la evaluación final. Si se ajustan los umbrales mirando el test, el resultado deja de ser fiable
  (buenas prácticas de la sesión 3 de CNN y de la Práctica 3).
- Fotogramas casi iguales (de un mismo vídeo) siempre en el mismo conjunto, para que no haya «fotos gemelas» entre
  entrenamiento y test.
- Conviene que sea **el mismo conjunto que use la vía DL**: así la comparación entre las dos vías es justa.
- Las imágenes no van a git: van al Drive del grupo, y en el repositorio se documenta dónde están.

### Paso 1 · Segmentación ⚠️

Objetivo: una máscara limpia de la mano, sin fondo, sin ruido y sin antebrazo.

**1a. Umbral de color de piel.** Se propone el espacio **YCrCb** con los rangos de la Práctica 4:
`133 ≤ Cr ≤ 173` y `77 ≤ Cb ≤ 127`, que habrá que ajustar a nuestra cámara y nuestra luz con el dataset.
- *Por qué YCrCb:* separa el brillo (Y) del color (Cr, Cb), así que el umbral aguanta mejor los cambios de luz que
  trabajar en BGR. Se vio en la Práctica 4 para filtrar falsas caras por color de piel.
- *Alternativa vista en clase:* HSV (tema 4). Con fondo claro, la piel tiene más saturación que la pared, y un umbral
  en el canal S puede bastar. Merece la pena comparar las dos con el dataset.

**1b. Reducir el ruido.** Suavizado gaussiano (`cv2.GaussianBlur`, tema 4) antes del umbral, para que no salgan
píxeles sueltos. Después, quedarse con la mancha más grande: esto **ya lo hace `medidas.py`** (paso 2), así que las
manchas pequeñas de ruido no afectan a las medidas.
- La limpieza con morfología (apertura y cierre, `cv2.morphologyEx`) **no se vio en clase**. Solo se debería añadir si
  el dataset muestra que hace falta (por ejemplo, agujeros falsos dentro de la mano), y justificándolo.

**1c. Quitar el antebrazo.** ⚠️ **Decisión pendiente.** Si el antebrazo entra en la máscara, el alargamiento y el
número de dedos salen mal. Opciones:
1. **Por las condiciones de captura** (recomendada primero): con manga larga oscura el antebrazo no tiene color de
   piel y el umbral no lo coge. No añade ninguna técnica.
2. **Recorte geométrico con técnicas de clase:** por ejemplo, con el círculo mínimo (`cv2.minEnclosingCircle`) o el
   rectángulo mínimo (`cv2.minAreaRect`, tema 4) de la mano, cortar lo que quede por debajo de la muñeca.
3. **Transformada de distancia** (`cv2.distanceTransform`, **no vista en clase**): su máximo está en el centro de la
   palma y da el radio de la palma; se corta a ~1,5 radios. Es la solución más robusta, pero habría que justificarla.

Se recomienda probar la opción 1 con el dataset y pasar a la 2 o la 3 solo si no basta.

- *Alternativa vista en clase descartada de momento:* **GrabCut** (tema 5) separa primer plano y fondo muy bien, pero
  necesita un rectángulo inicial y es lento para vídeo en directo. Podría servir para generar máscaras de referencia
  con las que comprobar la segmentación.

### Paso 2 · Contorno de la mano y agujero ✅

- `cv2.findContours` con **`RETR_TREE`** (tema 4) devuelve todos los contornos y, en la jerarquía, de qué contorno es
  hijo cada uno.
- La mano es el **contorno exterior más grande**; se descartan las manchas pequeñas de ruido.
- Los **hijos** del contorno de la mano son sus agujeros. Un agujero cuenta si ocupa al menos el **2 % del área de la
  mano** (`FRACCION_MIN_AGUJERO`); los huecos más pequeños son ruido. El 2 % es un valor inicial que se ajustará en el
  paso 7.
- **Para qué sirve:** detectar la **O**, cuyo círculo índice-pulgar deja un hueco dentro de la silueta. Es la única
  de las 6 letras con agujero.
- *Por qué `RETR_TREE` y no `RETR_CCOMP`:* las dos devuelven los agujeros, pero `RETR_TREE` es la que se vio en clase.

### Paso 3 · Solidez ✅

- **Solidez = área del contorno / área de su envolvente convexa** (`cv2.convexHull`, tema 4). Va de 0 a 1.
- La envolvente convexa es como una goma elástica alrededor de la mano: con la mano cerrada (A) o los dedos juntos
  (B) queda pegada y la solidez se acerca a 1; con los dedos separados (L, V, W) cubre los huecos entre ellos y la
  solidez baja.
- **Para qué sirve:** distinguir mano **cerrada** de mano **abierta**.
- **Limitación comprobada en los tests:** la solidez **no sirve para contar dedos**. Un solo dedo en un lado de la mano
  deja un hueco grande dentro de la envolvente y da una solidez más baja (0,69) que tres dedos (0,71). Por eso hace
  falta el paso 5.

### Paso 4 · Alargamiento ✅

- **Alargamiento = lado largo / lado corto del rectángulo de área mínima** (`cv2.minAreaRect`, tema 4). Vale 1 o más.
- **Para qué sirve:** distinguir la **B** (dedos juntos y estirados, silueta larga) de la **A** (puño, silueta
  compacta). Las dos tienen 0 dedos separados y solidez alta, así que las otras medidas no las separan.
- *Por qué `minAreaRect` y no `boundingRect`* (las dos del tema 4): `minAreaRect` gira el rectángulo para ajustarse a
  la forma, así que el resultado no cambia si la mano está inclinada, y la B se hace de lado. Comprobado en los tests:
  un rectángulo 3:1 girado 35° da **3,0** con `minAreaRect` y **1,19** con `boundingRect`, que lo confundiría con un
  puño.
- **Depende de que la máscara no incluya el antebrazo** (paso 1c): con el antebrazo, todo parece alargado.

### Paso 5 · Número de dedos 🔲

Propuesta:
- A partir de la envolvente convexa (`cv2.convexHull` con `returnPoints=False`), **`cv2.convexityDefects`** da los
  «defectos»: los tramos donde el contorno se separa de la envolvente. Cada uno trae su punto más profundo y su
  profundidad.
- Un defecto es un **hueco entre dos dedos** si:
  - es **profundo**: su profundidad supera una fracción del tamaño de la mano. Para medir el tamaño se propone el
    radio del círculo mínimo (`cv2.minEnclosingCircle`, tema 4), que no requiere técnicas nuevas;
  - y es **estrecho**: el ángulo en el punto profundo, entre los dos extremos del defecto, es menor de unos 90°.
    Los huecos anchos son la curva natural entre la palma y el pulgar o la muñeca, no huecos entre dedos.
- **Dedos = huecos + 1** si hay al menos un hueco; **0** si no hay ninguno (A, B).
- **Para qué sirve:** A/B (0 dedos separados), L/V (2), W (3).
- **Técnica no vista en clase: `cv2.convexityDefects`.** Justificación:
  - Es la continuación directa de `cv2.convexHull`, que sí se vio: devuelve justamente los puntos donde el contorno
    se aleja de la envolvente, es decir, los huecos entre los dedos.
  - Ninguna medida vista en clase cuenta dedos: la solidez no sirve (comprobado en el paso 3).
  - Es la forma habitual de contar dedos sobre una silueta.
- Los umbrales (fracción de profundidad y ángulo) quedan como constantes al principio de `medidas.py`, con su
  justificación, y se ajustan en el paso 7.
- Tests a añadir: formas dibujadas con 0, 2 y 3 dedos, y el caso de un solo dedo.

### Paso 6 · Ángulo entre dedos 🔲

Propuesta:
- Cuando hay **2 dedos**, el hueco entre ellos (paso 5) da tres puntos: los dos extremos del defecto (cerca de las
  puntas de los dedos) y su punto más profundo (la base, entre los dedos).
- El ángulo en la base se calcula con el mismo producto escalar que la función `angulo(a, b, c)` de la **sesión 5 y la
  Práctica 5** de MediaPipe, aplicado a puntos del contorno en lugar de puntos de MediaPipe.
- **Para qué sirve:** distinguir la **L** (~90°, índice y pulgar) de la **V** (~30–50°, índice y corazón).
- Tests a añadir: dos «dedos» dibujados a 90° y a 40°.

### Paso 7 · Análisis de las medidas 🔲

- En `herramientas/analisis_medidas.ipynb`: segmentar y medir todas las imágenes **de entrenamiento**, y dibujar con
  matplotlib (usado en todas las prácticas) **una gráfica por medida, con un color por letra**.
- En cada gráfica se ve qué letras separa esa medida y dónde poner el corte. Esos cortes son los umbrales del paso 8.
- También sirve para revisar los valores provisionales ya escritos (`FRACCION_MIN_AGUJERO` y los del paso 5).
- Las gráficas que vayan a la memoria se guardan en `docs/figuras/`.

### Paso 8 · Reglas de decisión 🔲

Árbol de decisión escrito a mano, en el orden en que se separan las letras con más seguridad:

```
¿hay agujero?                  → O
0 dedos:  alargada             → B
          compacta             → A
2 dedos:  ángulo ≈ 90°         → L
          ángulo < 50°         → V
3 dedos                        → W
cualquier otro caso            → '?'
```

- La solidez se puede usar como comprobación extra: por ejemplo, devolver `'?'` si dice «mano abierta» y no se han
  encontrado dedos.
- Los umbrales salen del paso 7 y se ajustan **solo con datos de entrenamiento**. Quedan como constantes al principio de
  `reconocer.py`, cada una con un comentario que diga de dónde sale.
- Tests: medidas inventadas que deben dar cada letra y casos que deben dar `'?'`.

### Paso 9 · Evaluación y comparación 🔲

- Pasar `reconocer` por el **conjunto de test** (una sola vez, al final) y medir, como en la Práctica 3:
  - acierto por letra y global (`classification_report` de sklearn);
  - **matriz de confusión**, para ver qué letras se confunden entre sí;
  - **porcentaje de `'?'`**: una vía que contesta poco pero acierta lo que contesta puede ser mejor para el robot.
- Robustez: resultados por persona y por iluminación.
- Palabras completas: pasar las letras por la clase `Palabra` con los vídeos de palabras.
- La misma evaluación se hace con la vía DL para compararlas, por eso conviene una función de evaluación común.

---

## 4. Resumen de técnicas

| Técnica | Paso | ¿Vista en clase? | Para qué |
|---|---|---|---|
| Conversión a YCrCb + umbral de piel | 1 | Sí (Práctica 4) | Separar la mano del fondo |
| Conversión a HSV + umbral | 1 (alternativa) | Sí (tema 4) | Separar la mano del fondo |
| `cv2.GaussianBlur` | 1 | Sí (tema 4) | Reducir ruido antes del umbral |
| `cv2.findContours` con `RETR_TREE` | 2 | Sí (tema 4) | Contorno de la mano y sus agujeros |
| `cv2.contourArea` | 2, 3 | No aparece en las transparencias; función básica de medida de contornos, de la misma familia que las del tema 4 | Área de la mano y de los agujeros |
| `cv2.convexHull` | 3, 5 | Sí (tema 4) | Envolvente convexa: solidez y base para los dedos |
| `cv2.minAreaRect` | 4 | Sí (tema 4) | Alargamiento, sin depender de la inclinación |
| `cv2.minEnclosingCircle` | 5 (y quizá 1c) | Sí (tema 4) | Tamaño de la mano para umbrales relativos |
| Ángulo entre tres puntos | 6 | Sí (sesión 5 y Práctica 5) | Distinguir L de V |
| matplotlib | 7 | Sí (todas las prácticas) | Gráficas para elegir umbrales |
| `classification_report`, matriz de confusión | 9 | Sí (Práctica 3) | Evaluación |
| **`cv2.convexityDefects`** | 5 | **No** | Contar dedos. Justificado en el paso 5 |
| *`cv2.morphologyEx`* | *1, solo si hace falta* | *No* | *Limpiar la máscara. Pendiente de decidir* |
| *`cv2.distanceTransform`* | *1c, solo si hace falta* | *No* | *Quitar el antebrazo. Pendiente de decidir* |
| *GrabCut* | *1, como referencia* | *Sí (tema 5)* | *Máscaras de referencia. Descartado para tiempo real* |

Ahora mismo hay **una sola técnica no vista en clase** en uso (`convexityDefects`), frente a más de diez vistas en
clase. Si se añaden morfología o transformada de distancia en el paso 1, hay que justificarlas aquí y en la memoria y
comprobar que siguen siendo minoría.

---

## 5. Cómo retomar el trabajo

- **Código:** `signos_gestos/clasico/`. Cada fichero empieza con una descripción de lo que hace o hará.
- **Tests:** `test/test_medidas.py` y `test/test_reconocer_clasico.py`. Desde la carpeta `signos_gestos/`:

  ```bash
  python -m pytest test/ -v
  ```

  Con `python -m` la carpeta actual entra en el path y se puede importar el paquete. Ejecutadlos antes de cada commit.
- **Cada medida nueva** va con su función en `medidas.py`, su entrada en el diccionario que devuelve `medidas()` y al
  menos un test con formas dibujadas con código.
- **Umbrales y valores provisionales:** siempre como constantes al principio del fichero, con un comentario que diga de
  dónde salen. Hoy: `FRACCION_MIN_AGUJERO = 0.02` en `medidas.py`.
- **Estilo:** el paquete ROS revisa el estilo con flake8 y pep257 (líneas de 99 caracteres como máximo y docstrings en
  todas las funciones). Código, comentarios y mensajes en español.
- **Siguiente paso sugerido:** el paso 5 (número de dedos) se puede hacer ya con formas dibujadas; el paso 1 necesita
  las primeras fotos del dataset.
- Organización general de la carpeta: `docs/ESTRUCTURA.md`.
