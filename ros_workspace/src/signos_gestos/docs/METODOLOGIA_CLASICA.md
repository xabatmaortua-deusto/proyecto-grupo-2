# Metodología de la vía clásica

Reconocimiento de letras del alfabeto dactilológico (LSE) **solo con OpenCV, sin modelos entrenados**.
Este documento explica qué pasos sigue la vía clásica, cuáles están hechos y por qué se ha elegido cada técnica,
para que quien retome el trabajo sepa por dónde seguir.

Estado a 26/11/2026. Actualizad las marcas de estado cada vez que se termine un paso.

| Marca | Significado |
|---|---|
| ✅ | Hecho y con tests |
| 🟡 | Hecho y con tests, pero solo probado con imágenes sintéticas: falta validarlo con el dataset real |
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
| 1 | Segmentación: de la foto a la máscara de la mano, sin antebrazo | `signos_gestos/clasico/segmentar.py` | 🟡 |
| 2 | Contorno de la mano y agujero interior | `signos_gestos/clasico/medidas.py` | ✅ |
| 3 | Solidez | `signos_gestos/clasico/medidas.py` | ✅ |
| 4 | Alargamiento | `signos_gestos/clasico/medidas.py` | ✅ |
| 5 | Número de dedos | `signos_gestos/clasico/medidas.py` | ✅ |
| 6 | Ángulo entre dedos | `signos_gestos/clasico/medidas.py` | ✅ |
| 7 | Análisis de las medidas por letra para elegir umbrales | `herramientas/analisis_medidas.ipynb` | 🟡 |
| 8 | Reglas de decisión | `signos_gestos/clasico/reconocer.py` | 🟡 |
| 9 | Evaluación y comparación con la vía DL | `signos_gestos/comun/evaluar.py` (común a todas las vías) | 🔲 |

**Toda la cadena funciona de principio a fin con imágenes dibujadas** (`test_reconocer_clasico.py`: reconoce las 6
letras y devuelve `'?'` sin mano o con un solo dedo). Lo que falta es el **dataset real** (paso 0) para ajustar los
umbrales de los pasos 1, 7 y 8 y evaluar (paso 9).

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

### Paso 1 · Segmentación 🟡

Objetivo: una máscara limpia de la mano, sin fondo, sin ruido y sin antebrazo.

**Implementado** en `segmentar.py` con la función `mascara_mano(frame_bgr) -> mascara`:

1. Suavizado gaussiano con un filtro de 5 × 5 (`TAM_KERNEL`).
2. Conversión a YCrCb.
3. Umbral de piel en los canales Cr y Cb con los rangos de la Práctica 4 (`CR_MIN`, `CR_MAX`, `CB_MIN`, `CB_MAX`).
4. Máscara de un canal con valores 0 y 255, el formato que espera `medidas.py`.

No usa morfología ni recorta el antebrazo: confía en las condiciones de captura (opción 1 del apartado 1c).
Tiene 7 tests en `test_segmentar.py` (máscara de un canal y binaria, fondo vacío, detecta el color de piel, la manga
oscura no se segmenta, conserva el agujero de la O y es compatible con `medidas.py`). Para ver la máscara de una foto
real: `herramientas/visualizar_segmentacion.py`.

**Pendiente:** ajustar los rangos de color con fotos reales de nuestra cámara y nuestra luz, y comprobar con ellas que
la manga oscura basta para eliminar el antebrazo.

**1a. Umbral de color de piel.** Espacio **YCrCb** con los rangos de la Práctica 4:
`133 ≤ Cr ≤ 173` y `77 ≤ Cb ≤ 127`, que habrá que ajustar a nuestra cámara y nuestra luz con el dataset.
- *Por qué YCrCb:* separa el brillo (Y) del color (Cr, Cb), así que el umbral aguanta mejor los cambios de luz que
  trabajar en BGR. Se vio en la Práctica 4 para filtrar falsas caras por color de piel.
- *Alternativa vista en clase:* HSV (tema 4). Con fondo claro, la piel tiene más saturación que la pared, y un umbral
  en el canal S puede bastar. Merece la pena comparar las dos con el dataset.

**1b. Reducir el ruido.** Suavizado gaussiano (`cv2.GaussianBlur`, tema 4) antes del umbral, para que no salgan
píxeles sueltos. Después, quedarse con la mancha más grande: esto **lo hace `medidas.py`** (paso 2), así que las
manchas pequeñas de ruido no afectan a las medidas.
- La limpieza con morfología (apertura y cierre, `cv2.morphologyEx`) **no se vio en clase**. Solo se debería añadir si
  el dataset muestra que hace falta (por ejemplo, agujeros falsos dentro de la mano), y justificándolo.

**1c. Quitar el antebrazo.** Si el antebrazo entra en la máscara, el alargamiento y el número de dedos salen mal.
**Decisión actual: opción 1 (manga larga oscura).** Si con las fotos reales no basta, se pasa a la 2 o la 3. Opciones:
1. **Por las condiciones de captura** (recomendada primero): con manga larga oscura el antebrazo no tiene color de
   piel y el umbral no lo coge. No añade ninguna técnica.
2. **Recorte geométrico con técnicas de clase:** por ejemplo, con el círculo mínimo (`cv2.minEnclosingCircle`) o el
   rectángulo mínimo (`cv2.minAreaRect`, tema 4) de la mano, cortar lo que quede por debajo de la muñeca.
3. **Transformada de distancia** (`cv2.distanceTransform`, **no vista en clase**): su máximo está en el centro de la
   palma y da el radio de la palma; se corta a ~1,5 radios. Es la solución más robusta, pero habría que justificarla.

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

### Paso 5 · Número de dedos ✅

- A partir de la envolvente convexa (`cv2.convexHull` con `returnPoints=False`), **`cv2.convexityDefects`** da los
  «defectos»: los tramos donde el contorno se separa de la envolvente. Cada uno trae sus dos extremos (cerca de las
  puntas de los dedos), su punto más profundo (la base del hueco) y su profundidad.
- Un defecto es un **hueco entre dos dedos** si:
  - es **profundo**: su profundidad es al menos el 25 % del radio del círculo mínimo que encierra la mano
    (`FRACCION_MIN_PROFUNDIDAD`; el radio se calcula con `cv2.minEnclosingCircle`, tema 4, sin técnicas nuevas);
  - y es **estrecho**: el ángulo en la base, entre los dos extremos, es menor de 120° (`ANGULO_MAX_HUECO`). Los
    huecos anchos son los lados de la palma, no huecos entre dedos.
- **Dedos = huecos + 1** si hay al menos un hueco; **0** si no hay ninguno (A, B).
- **Para qué sirve:** A/B (0 dedos separados), L/V (2), W (3).
- **Resultados con manos dibujadas** (palma redonda y dedos gruesos; ver `test_medidas.py`):

  | Mano | Dedos | Ángulo |
  |---|---|---|
  | A (puño) | 0 | — |
  | B (tres dedos juntos) | 0 | — |
  | V (±20°) | 2 | 52° |
  | V estrecha (±12°) | 2 | 35° |
  | L recta / inclinada | 2 | 110° / 107° |
  | W | 3 | — |

- **Por qué el límite es 120° y no 90°:** el hueco de la L mide unos 105–110°, porque se mide desde la base hasta los
  extremos de dos dedos gruesos, no entre los ejes de los dedos. Con 90° se perdería la L; con 110° quedaba justo en
  el límite.
- **Depende de que la máscara no incluya el antebrazo:** con un trozo de muñeca dibujado bajo la palma, la L contaba
  3 dedos, porque el hueco entre el pulgar y la muñeca (~100°) parece un hueco entre dedos. El ángulo no lo puede
  filtrar; la solución es recortar el antebrazo en el paso 1c.
- **Caso a vigilar en las reglas (paso 8):** una mano con **un solo dedo** levantado da 0 dedos (no deja huecos
  estrechos) y es alargada, así que se parece a la B. No es una de nuestras letras, pero si alguien la hace, las
  reglas deberían devolver `'?'`. Resuelto en el paso 8 con la solidez (`UMBRAL_SOLIDEZ`), aunque con poco margen.
- **Compatibilidad:** OpenCV 5 (el de `requirements.txt`) devuelve los defectos con forma `(N, 4)` y OpenCV 4 (el de
  los ejemplos de clase) con `(N, 1, 4)`. El código usa `reshape` para funcionar con los dos.
- **Técnica no vista en clase: `cv2.convexityDefects`.** Justificación:
  - Es la continuación directa de `cv2.convexHull`, que sí se vio: devuelve justamente los puntos donde el contorno
    se aleja de la envolvente, es decir, los huecos entre los dedos.
  - Ninguna medida vista en clase cuenta dedos: la solidez no sirve (comprobado en el paso 3).
  - Es la forma habitual de contar dedos sobre una silueta.
- Los umbrales (fracción de profundidad y ángulo) quedan como constantes al principio de `medidas.py`, con su
  justificación, y se ajustan en el paso 7.

### Paso 6 · Ángulo entre dedos ✅

- Cuando hay **2 dedos**, el único hueco entre ellos (paso 5) da tres puntos: los dos extremos y la base.
- El ángulo en la base se calcula con el mismo producto escalar que la función `angulo(a, b, c)` de la **sesión 5 y la
  Práctica 5** de MediaPipe, aplicado a puntos del contorno en lugar de puntos de MediaPipe.
- Con otro número de dedos devuelve `None`: solo hace falta para separar la L de la V.
- **Para qué sirve:** distinguir la **L** (índice y pulgar, ~105–110° medido así) de la **V** (índice y corazón,
  ~35–55°). La separación entre las dos es muy grande, así que el umbral de la regla (paso 8) tiene margen.

### Paso 7 · Análisis de las medidas 🟡

- En `herramientas/analisis_medidas.ipynb`: segmentar y medir todas las imágenes **de entrenamiento**, y dibujar con
  matplotlib (usado en todas las prácticas) **una gráfica por medida, con un color por letra**.
- En cada gráfica se ve qué letras separa esa medida y dónde poner el corte. Esos cortes son los umbrales del paso 8.
- También sirve para revisar los valores provisionales ya escritos (`FRACCION_MIN_AGUJERO` y los del paso 5).
- Las gráficas que vayan a la memoria se guardan en `docs/figuras/`.

**Implementado:** el cuaderno carga las imágenes (la letra sale del nombre del archivo), las segmenta con
`mascara_mano()`, las mide con `medidas()` y muestra una tabla de medidas, una gráfica por medida (agujero, dedos,
alargamiento, solidez y ángulo) y los mínimos, máximos y medias de cada medida por letra. Con esas gráficas se eligen
los umbrales de `reconocer.py` (apartados 7, 8 y 9 del cuaderno).

**Ajuste de los umbrales de `medidas.py`** (apartado 10 del cuaderno). Las gráficas anteriores muestran medidas ya
decididas («hay agujero», «2 dedos»), que no dicen por cuánto falla un umbral. Este apartado muestra los **valores en
bruto**, con el umbral actual como línea discontinua:
- **10.1** El tamaño del agujero más grande de cada mano respecto a su área (`mayor_fraccion_agujero()`), para
  elegir `FRACCION_MIN_AGUJERO`. Solo la O debe quedar por encima de la línea.
- **10.2** La profundidad y el ángulo de **todos** los defectos de convexidad (`defectos_convexidad()`), también los
  que se descartan, para elegir `FRACCION_MIN_PROFUNDIDAD` y `ANGULO_MAX_HUECO`. Solo los huecos reales entre dedos
  de la L, la V y la W deben quedar en la zona de arriba a la izquierda.

Probado con imágenes dibujadas: las O dan un agujero del 9–18 % y el resto de letras 0 %; los huecos entre dedos
quedan en la zona correcta. Hay un grupo de defectos de la L, la V y la W **justo por debajo** de la línea de
profundidad (0,21–0,23, con 130–140°): son los lados entre el dedo de fuera y la palma. Con manos reales hay que
vigilar que no suban por encima de la línea, porque se contarían como dedos.

**Pendiente:** se ha ejecutado con **imágenes sintéticas** guardadas en una carpeta temporal
(`/tmp/pruebas_gestos_sinteticas/`) que **no está en el repositorio**, así que nadie más puede repetir ese análisis.
Hay que cambiar `CARPETA_DATOS` a las imágenes de entrenamiento del dataset real (paso 0) y volver a ejecutarlo.

### Paso 8 · Reglas de decisión 🟡

Árbol de decisión escrito a mano, en el orden en que se separan las letras con más seguridad:

```
¿hay agujero?                  → O
0 dedos:  alargada             → B
          compacta             → A
2 dedos:  ángulo grande        → L   (~105-110° con manos dibujadas)
          ángulo pequeño       → V   (~35-55°)
3 dedos                        → W
cualquier otro caso            → '?'
```

- Los umbrales salen del paso 7 y se ajustan **solo con datos de entrenamiento**. Quedan como constantes al principio de
  `reconocer.py`, cada una con un comentario que diga de dónde sale.

**Implementado** en `reconocer.py`: `clasificar_medidas(medidas) -> letra` aplica el árbol, y `reconocer(frame_bgr)`
encadena segmentación, medidas y reglas. Umbrales iniciales, elegidos con las imágenes sintéticas del paso 7:

| Constante | Valor | De dónde sale (imágenes sintéticas) | Para qué |
|---|---|---|---|
| `UMBRAL_ALARGAMIENTO` | 1,8 | A = 1,06; B = 2,63 | Más alargada que el umbral → B; si no → A |
| `UMBRAL_ANGULO` | 90° | L = 115°; V = 67° | Más abierta que el umbral → L; si no → V |
| `UMBRAL_SOLIDEZ` | 0,9 | A y B cerca de 1; un solo dedo = 0,82 | Con 0 dedos, por debajo del umbral → `'?'` |

La solidez se usa como comprobación extra: con 0 dedos, si la forma no es compacta, devuelve `'?'` en lugar de A o B.
Así no confunde con la B una mano con un solo dedo levantado (el caso que se avisaba en el paso 5).

Tests en `test_reconocer_clasico.py` (20):
- **Reglas, con medidas inventadas:** cada letra, que el agujero manda sobre el resto y los casos que deben dar `'?'`
  (sin mano, 0 dedos con poca solidez, 2 dedos sin ángulo, 1, 4 o 5 dedos). Los valores se calculan a partir de las
  constantes `UMBRAL_*`, para que los tests sigan valiendo cuando se ajusten los umbrales.
- **Cadena completa, con imágenes en color dibujadas** (fondo claro y mano del color de piel de `test_segmentar.py`):
  reconoce A, B, O, L, V y W, y devuelve `'?'` sin imagen, con el fondo vacío o con un solo dedo.

**Pendiente: ajustar los umbrales con el dataset real.** Hay que vigilar sobre todo la solidez:
- Con una B dibujada como tres dedos juntos (con pequeños huecos entre ellos), la solidez es **0,81** y el reconocedor
  devuelve `'?'` en lugar de B. Un solo dedo levantado da 0,76–0,82: hay muy poco margen entre los dos casos.
- Una B real, con los dedos juntos pero no pegados del todo, podría quedar por debajo de 0,9. Si pasa, habrá que bajar
  el umbral o buscar otra forma de separar la B del dedo suelto (por ejemplo, la anchura relativa de la forma).
- Los ángulos de la L y la V dependen de cómo se dibuje la mano: con las manos de `test_medidas.py` salen L ≈ 107–110°
  y V ≈ 35–52°; con las imágenes del paso 7, L = 115° y V = 67°. En los dos casos el umbral de 90° las separa, pero el
  valor bueno saldrá de las fotos reales.

### Paso 9 · Evaluación y comparación 🔲

- Pasar `reconocer` por el **conjunto de test** (una sola vez, al final) y medir, como en la Práctica 3:
  - acierto por letra y global (`classification_report` de sklearn);
  - **matriz de confusión**, para ver qué letras se confunden entre sí;
  - **porcentaje de `'?'`**: una vía que contesta poco pero acierta lo que contesta puede ser mejor para el robot.
- Robustez: resultados por persona y por iluminación.
- Palabras completas: pasar las letras por la clase `Palabra` con los vídeos de palabras.
- La misma evaluación se hace con la vía DL para compararlas, por eso conviene una función de evaluación común.

**Herramienta disponible:** `signos_gestos/comun/evaluar.py` (común a todas las vías, en la rama `grupo-4-gestos`)
ya calcula el acierto por letra, el porcentaje de `'?'`, el acierto sin contar los `'?'`, la matriz de confusión y la
evaluación de palabras completas, y tiene una opción para la vía clásica. Desde la carpeta `signos_gestos/`:

```bash
python -m signos_gestos.comun.evaluar --via clasico --personas <persona_de_test>
python -m signos_gestos.comun.evaluar --via clasico --palabras --personas <persona_de_test>
```

Lee los vídeos de `datos/videos/<LETRA>/` y `datos/palabras/<PALABRA>/`. `--personas` elige el conjunto de test.

**Pendiente:** el dataset real (paso 0) y unir `grupo-4-gestos` a la rama de la vía clásica para tener `evaluar.py`.

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
| `cv2.imshow`, `cv2.imread` | Herramienta de visualización | Sí (tema 3) | Ver la máscara de una foto |
| `cv2.bitwise_and` | Herramienta de visualización | No aparece en las transparencias; solo se usa para mostrar la mano recortada, no en el reconocimiento | Ver la mano sin el fondo |
| `classification_report`, matriz de confusión | 9 | Sí (Práctica 3) | Evaluación |
| **`cv2.convexityDefects`** | 5 | **No** | Contar dedos. Justificado en el paso 5 |
| *`cv2.morphologyEx`* | *1, solo si hace falta* | *No* | *Limpiar la máscara. Pendiente de decidir* |
| *`cv2.distanceTransform`* | *1c, solo si hace falta* | *No* | *Quitar el antebrazo. Pendiente de decidir* |
| *GrabCut* | *1, como referencia* | *Sí (tema 5)* | *Máscaras de referencia. Descartado para tiempo real* |

Ahora mismo hay **una sola técnica no vista en clase** en el reconocimiento (`convexityDefects`), frente a más de diez
vistas en clase. Si se añaden morfología o transformada de distancia en el paso 1, hay que justificarlas aquí y en la memoria y
comprobar que siguen siendo minoría.

---

## 5. Cómo retomar el trabajo

- **Código:** `signos_gestos/clasico/`. Cada fichero empieza con una descripción de lo que hace o hará.
- **Tests:** `test/test_segmentar.py`, `test/test_medidas.py` y `test/test_reconocer_clasico.py` (45 en total).
  Desde la carpeta `signos_gestos/`:

  ```bash
  python -m pytest test/test_segmentar.py test/test_medidas.py test/test_reconocer_clasico.py -v
  ```

  Con `python -m` la carpeta actual entra en el path y se puede importar el paquete. Ejecutadlos antes de cada commit.
  No uséis `pytest test/` fuera del contenedor: en esa carpeta también están los tests de estilo de ROS
  (`test_flake8.py`, `test_pep257.py`, `test_copyright.py`), que necesitan las herramientas de ROS. Esos se ejecutan
  con `colcon test` dentro del contenedor.
- **Cada medida nueva** va con su función en `medidas.py`, su entrada en el diccionario que devuelve `medidas()` y al
  menos un test con formas dibujadas con código.
- **Umbrales y valores provisionales:** siempre como constantes al principio del fichero, con un comentario que diga de
  dónde salen. Todos se revisan en el paso 7 con fotos reales. Hoy:
  - `segmentar.py`: `TAM_KERNEL = (5, 5)`, `CR_MIN = 133`, `CR_MAX = 173`, `CB_MIN = 77`, `CB_MAX = 127`.
  - `medidas.py`: `FRACCION_MIN_AGUJERO = 0.02`, `FRACCION_MIN_PROFUNDIDAD = 0.25`, `ANGULO_MAX_HUECO = 120`.
  - `reconocer.py`: `UMBRAL_ALARGAMIENTO = 1.8`, `UMBRAL_ANGULO = 90`, `UMBRAL_SOLIDEZ = 0.9`.
- **Estilo:** el paquete ROS revisa el estilo con flake8 y pep257 (líneas de 99 caracteres como máximo, salto de línea
  al final de cada fichero y docstrings en todas las funciones). Código, comentarios y mensajes en español.
- **Siguiente paso sugerido:** toda la cadena está hecha y probada con imágenes dibujadas. Lo que falta depende del
  **dataset real** (paso 0), en este orden:
  1. Comprobar la segmentación con fotos reales (`visualizar_segmentacion.py`) y ajustar los rangos de color.
  2. Volver a ejecutar el análisis (paso 7) con las imágenes de entrenamiento y ajustar los umbrales de `medidas.py`
     y `reconocer.py`, vigilando sobre todo la solidez de la B.
  3. Evaluar con el conjunto de test usando `evaluar.py --via clasico` (paso 9) y comparar con la vía DL.
- Organización general de la carpeta: `docs/ESTRUCTURA.md`.
