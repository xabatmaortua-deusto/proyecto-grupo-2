# Reparto de tareas · Vía clásica y vía deep learning

Equipo 4: **Ainhoa, Ane, Keltse y Xabi**. El contexto y el detalle técnico están en [PLAN.md](PLAN.md).

> **Esto es una propuesta.** Los nombres se pueden cambiar entre nosotros. Lo importante es que cada tarea tenga una persona responsable y un entregable claro.

---

## Resumen

|  | Vía clásica (OpenCV) | Vía deep learning | Tarea transversal |
|---|---|---|---|
| **Ainhoa** | Segmentación de la mano | — | Organizar el dataset propio |
| **Ane** | Medidas geométricas y reglas | — | Memoria y documentación |
| **Xabi** | — | Datos para la red (públicos + propios) | Integración con Robótica |
| **Keltse** | — | Entrenamiento del modelo | Evaluación común y comparación |

Hay **dos personas por vía** para que cada una pueda avanzar por su cuenta. Además, cada persona tiene una **tarea transversal** que une las dos vías.

---

## 0. Lo que hacemos todos al principio

### 0.1 Acordar una interfaz común

Las dos vías tienen que ofrecer **la misma función**, para que la construcción de palabras y la evaluación sirvan para las dos sin cambiar nada:

```python
def reconocer(frame_bgr) -> str:
    """Recibe un fotograma de OpenCV (BGR) y devuelve 'A', 'B', 'L', 'O', 'V', 'W' o '?'."""
```

- Vía clásica: `clasico/reconocer.py`
- Vía deep learning: `deep/reconocer.py`
- Construcción de la palabra: la clase `Palabra` del módulo común `signos_gestos/comun/palabra.py` (antes en `prueba_mediapipe_letras.py`).

### 0.2 Grabar el dataset propio (todos, lo coordina Ainhoa)

- **Cada persona** graba **un vídeo de 5 a 10 s por letra** (A, B, L, O, V, W) con la cámara fija, fondo liso y manga larga oscura.
- Se graba **con dos iluminaciones**, por ejemplo luz de día y luz artificial.
- Cada persona graba también **3 palabras**: LOBO, BOLA y OLA.
- En total salen 4 personas × 6 letras × 2 luces = **48 vídeos de letras** y 12 de palabras.
- Antes de grabar, **confirmamos juntos cómo se hace cada letra** en LSE, para que todos la hagamos igual.

---

## 1. Vía clásica (OpenCV, sin modelos entrenados) · Ainhoa y Ane

### Ainhoa · Segmentación de la mano

| # | Tarea | Entregable |
|---|---|---|
| C1 | Segmentar la piel en **YCrCb** y en **HSV**, ajustando los rangos a nuestras grabaciones | `clasico/segmentar.py` con la función `mascara_mano(frame) -> mascara` |
| C2 | Limpiar la máscara: apertura y cierre morfológicos y el componente conexo más grande | Igual que C1 |
| C3 | **Quitar el antebrazo**: centro de la palma con `distanceTransform` y corte a ~1,5·*r* | Función `recortar_brazo(mascara) -> mascara, centro, radio` |
| C4 | Revisar las máscaras **a ojo** en todos los vídeos y anotar en cuáles falla (luz, ropa...) | Carpeta `resultados_clasico/mascaras/` y una lista de fallos |

### Ane · Medidas geométricas y reglas

Puede empezar con máscaras hechas a mano, o con las primeras de Ainhoa, sin esperar a que C1-C4 estén terminadas.

| # | Tarea | Entregable |
|---|---|---|
| C5 | Contorno con `RETR_CCOMP` y cálculo de las medidas: **nº de dedos** (defectos de convexidad), **agujero** (O), **solidez**, **alargamiento** y **ángulo entre dedos** | `clasico/medidas.py` con `medidas(mascara) -> dict` |
| C6 | Calcular las medidas en todo el dataset y hacer **gráficas por letra** para ver qué medida separa qué letras | Notebook `clasico/analisis_medidas.ipynb` |
| C7 | Escribir las **reglas** (árbol de decisión) eligiendo los umbrales a partir de las gráficas. Si no hay una letra clara, devolver `?`. | `clasico/reconocer.py` con `reconocer(frame)` |
| C8 | Ajustar con los vídeos de **entrenamiento**, nunca con los de test | Umbrales finales comentados en el código |

---

## 2. Vía deep learning (transfer learning) · Xabi y Keltse

### Xabi · Datos para la red

| # | Tarea | Entregable |
|---|---|---|
| D1 | Revisar **ASL Alphabet** y **Sign Language MNIST** y comparar **letra a letra con la LSE**. Por ejemplo, nuestra O es la F de la ASL. Decidir qué letras se pueden aprovechar. | Tabla de correspondencia ASL ↔ LSE en este repositorio |
| D2 | Extraer fotogramas de nuestros vídeos a `datos/<letra>/` (unos 5 por segundo, para que no salgan casi iguales) | Script `deep/extraer_fotogramas.py` |
| D3 | **Recortar la mano** en cada imagen, para que la red no aprenda el fondo ni la ropa. Puede usarse la segmentación de Ainhoa (C1-C3) o MediaPipe, si el profesor lo permite. | Carpeta `datos_recortados/<letra>/` |
| D4 | Dividir en train, validación y test **por persona** (por ejemplo, una persona solo para test), para que la evaluación sea honesta | Fichero con la división, compartido con Keltse |

### Keltse · Entrenamiento del modelo

| # | Tarea | Entregable |
|---|---|---|
| D5 | Adaptar el notebook de la **práctica 3** (`ImageFolder` + red preentrenada, por ejemplo ResNet18 o MobileNet) a nuestras 6 letras | `deep/entrenar.ipynb` |
| D6 | **Aumento de datos**: rotaciones pequeñas, cambios de brillo y contraste y recortes. **No usar volteo horizontal**, porque cambia la mano. | Transformaciones en el notebook |
| D7 | Entrenar primero solo la última capa y luego hacer *fine-tuning*. Guardar el mejor modelo según validación. | `deep/modelo.pt` y gráficas de pérdida y acierto |
| D8 | Rechazar las predicciones inseguras: si la probabilidad máxima es menor que un umbral, devolver `?` | `deep/reconocer.py` con `reconocer(frame)` |
| D9 | Si da tiempo, **ampliar a más letras** (hasta las 24 estáticas), como dice el hito | Nueva versión del modelo |

**Nota:** el portátil no tiene una GPU compatible, así que se entrena en CPU, con pocas épocas e imágenes de 224 px, o en **Google Colab** con GPU.

---

## 3. Tareas transversales

### Keltse · Evaluación común y comparación

| # | Tarea | Entregable |
|---|---|---|
| T1 | Script que, dada una función `reconocer`, la pasa por el **conjunto de test** y calcula el acierto por letra, la **matriz de confusión** y el porcentaje de `?` | `comun/evaluar.py` |
| T2 | Evaluar **palabras completas** con los vídeos de palabras, usando `Palabra` | Tabla: palabra real, palabra reconocida y si es correcta |
| T3 | **Robustez**: comparar resultados por persona y por iluminación | Tabla y gráficas para la memoria |
| T4 | Pasar también MediaPipe por la misma evaluación, como **referencia** | Fila extra en la comparación |

### Xabi · Integración con Robótica

| # | Tarea | Entregable |
|---|---|---|
| T5 | Acordar con el equipo de Robótica **cómo y cuándo se envía la palabra**: formato (texto), medio (fichero, socket, ROS...) y cómo se detecta el final de la palabra (por ejemplo, mano fuera de cuadro 1 s) | Documento corto con el protocolo |
| T6 | Programa de demostración: cámara en directo → `reconocer` → `Palabra` → envío al robot | `demo.py` |
| T7 | Pruebas integradas con palabras cortas | Vídeo de la demo |

### Ane · Memoria y documentación

| # | Tarea | Entregable |
|---|---|---|
| T8 | Ir redactando la memoria a medida que cada parte avanza: metodología, decisiones y problemas encontrados | Borrador compartido |
| T9 | Recoger figuras: máscaras, gráficas de medidas, curvas de entrenamiento y matrices de confusión | Carpeta `figuras/` |
| T10 | Explicar las diferencias **LSE ↔ ASL** y las limitaciones (fondo uniforme, gestos estáticos...) | Apartado de la memoria |

### Ainhoa · Organización del dataset

| # | Tarea | Entregable |
|---|---|---|
| T11 | Guion de grabación (qué grabar, cómo nombrar los ficheros, condiciones) y comprobar que están todos los vídeos | `datos/LEEME.md` y tabla de vídeos |
| T12 | Estructura de carpetas común para las dos vías | Carpetas creadas en el repositorio |

---

## 4. Orden y dependencias

```
Fase 1  ─ Todos: interfaz común (0.1) + guion y grabación del dataset (0.2, T11-T12)
            │
Fase 2  ─┬─ Ainhoa: segmentación (C1-C4)      ─┐
         ├─ Ane: medidas y gráficas (C5-C6)    ─┤  en paralelo
         ├─ Xabi: datos y recorte (D1-D4)      ─┤
         └─ Keltse: evaluar.py (T1) y          ─┘
            notebook de entrenamiento (D5-D6)
            │
Fase 3  ─┬─ Ane: reglas (C7-C8)
         ├─ Keltse: entrenamiento (D7-D8)
         └─ Xabi: protocolo con Robótica (T5)
            │
Fase 4  ─── Keltse: evaluación y comparación (T2-T4)
            Xabi: demo e integración (T6-T7)
            Ane: memoria (T8-T10), con aportaciones de todos
```

**Dependencias clave:**
- Sin el **dataset (0.2)** no se puede ajustar ni entrenar nada. Es lo primero.
- **D3** (recortar la mano) puede reutilizar la segmentación de Ainhoa: Xabi y Ainhoa deben coordinarse.
- **T1** (evaluar) solo necesita la interfaz `reconocer`, así que puede hacerse pronto con el reconocedor de MediaPipe que ya existe.

---

## 5. Normas del equipo

- **Un único entorno:** `.venvVision`. Si alguien instala un paquete, lo apunta en `requirements.txt`.
- **El conjunto de test no se toca** hasta la evaluación final, ni para ajustar umbrales ni para elegir el modelo.
- Cada script se ejecuta desde la carpeta `Proyecto/` y guarda sus resultados en su propia carpeta `resultados_*`.
- Lo de cada uno, en una **rama de git propia**, y se junta en `main` cuando funciona.
