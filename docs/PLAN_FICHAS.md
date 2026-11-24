# Plan de visión: fichas (Equipo 9)

Equipo 9: Mikel, Garai, Xabat y Markel. Este documento es el plan del **bloque B** de [`TAREAS.md`](TAREAS.md)
(B1-B8) organizado para nosotros cuatro. No cambia nada del diseño común de [`CLAUDE.md`](../CLAUDE.md): el
contrato (`Ficha`, `FichaArray`, `/signos/fichas`) y las reglas de desarrollo siguen siendo las de allí.

## Cadena de procesamiento

Cada imagen de la cámara cenital pasa por cinco etapas. La 2 y la 3 tienen versión clásica y DL; se eligen con
`metodo:=clasico|dl` y las dos devuelven exactamente lo mismo.

| Etapa | Qué hace | Clásico | DL |
|---|---|---|---|
| 1. Bandeja | ArUco → imagen enderezada, píxel → mm | común (`bandeja.py`, hecho) | común |
| 2. Detección | Dónde hay fichas: centro (mm), ángulo de sus bordes y recorte alineado con la ficha | umbral/Otsu + `findContours` + `minAreaRect` (tema 4) | YOLO (sesión 5 y scripts de SAM de clase) + `minAreaRect` dentro de la caja |
| 3. Letra | Qué letra es y cuál de los 4 giros del recorte la deja derecha (raya abajo) | la raya da el giro; la letra, por agujeros (jerarquía de contornos) y cortes con líneas (Hito 1) | clasificador CNN (ResNet18, práctica 3) sobre los 4 giros del recorte; gana el de más confianza |
| 4. Pose final | `theta` = ángulo de la etapa 2 + giro de la etapa 3; paso de bandeja a `world` | común | común |
| 5. ROS | Nodo `signos_fichas` que publica `FichaArray` | común | común |

### Contrato interno del Equipo 9

Es lo que permite trabajar en paralelo. Todo se puede probar ya con las imágenes sintéticas (`sinteticas.py`),
que traen la verdad de cada ficha (letra, x, y, theta).

```python
deteccion.detectar_clasico(enderezada) -> [{'x': mm, 'y': mm, 'angulo': rad, 'recorte': imagen}, ...]
deteccion.detectar_dl(enderezada)      -> igual
letras.reconocer_clasico(recorte)      -> (letra, giro, confianza)   # giro: 0, 1, 2 o 3 cuartos de vuelta
letras.reconocer_dl(recorte)           -> igual
```

Como la ficha mide 18 × 20 mm y es casi cuadrada, la detección no sabe hacia dónde mira la letra: por eso la
etapa 3 decide entre 4 giros.

Convenio de `theta` (igual que en `CLAUDE.md`): dirección raya → parte de arriba de la letra, en radianes.
En la bandeja se mide desde su +X (del ArUco 0 al 1); el nodo lo pasa a `world`.

## Reparto de tareas (simbólico e inicial, sujeto a cambios)

Cuatro bloques de tamaño parecido. Garai y Xabat hacen **su etapa con los dos métodos**, así cada comparación
clásico vs DL la hace una misma persona. Mikel y Markel llevan lo común (cámara, ROS, datos y evaluación).
Los nombres son una propuesta: se pueden intercambiar los bloques.

| | Mikel · Bandeja, calibración y ROS | Garai · Detección de fichas | Xabat · Reconocimiento de letra | Markel · Datos y evaluación |
|---|---|---|---|---|
| **Común** | `bandeja.py` y `sinteticas.py` (hechos) · calibración ligera de la ELP con `calibrar.py` de clase y probar si `edge_refine` mejora · bandeja → `world` (B3, con robótica) | | | Material (B1): letra y raya en las fichas (≥ 2 por letra), ArUco pegados, medir `ANCHO_MM` y `FONDO_MM` |
| **Clásico** | | `detectar_clasico`: umbral/Otsu en la enderezada, contornos, filtrar por tamaño (18 × 20 mm), `minAreaRect`, recorte alineado | `reconocer_clasico`: raya → giro; agujeros (B; A/O) y cortes con líneas (L, V, W) | |
| **DL** | | `detectar_dl`: YOLO entrenado con las fotos etiquetadas (clase "ficha") | `reconocer_dl`: ResNet18 con una carpeta por letra + truco de los 4 giros · extra: más letras | Dataset (B4): vídeos con la ELP, `sacar_fotogramas.py`, `etiquetar_sam.py`, `preparar_dataset.py`, Drive |
| **ROS** | Nodo `signos_fichas` con `metodo` · ventana de depuración con las detecciones dibujadas | | | |
| **Evaluación** | Error de posición y ángulo con el robot real (en la integración) | Tiempo por imagen, fichas perdidas y falsas (su etapa) | Matriz de confusión de letras y acierto del giro (su etapa) | `evaluacion.py`: fotos con verdad → los dos métodos → % letra correcta, error en mm y en grados, falsos positivos, tiempo |
| **Documentos** | Coordina el README de fichas y la arquitectura | Apartado de detección (clásico vs DL) | Apartado de reconocimiento (clásico vs DL) | Coordina y une la memoria de Visión y la comparativa final |

Cada uno escribe en la memoria el objetivo, el método, el código y los resultados de **su** parte.

### Lo que toca a todo el grupo (con PR revisado por las dos partes, según `CLAUDE.md`)

- `signos_interfaces`: `Ficha.msg` y `FichaArray.msg` exactamente como están en `CLAUDE.md`. Si cuando haga falta
  aún no están (Fase 0), los crea Mikel tal cual, sin cambiar nada del contrato.
- `signos_bringup/config/mesa.yaml`: solo la posición de la bandeja (ArUco) respecto a `world` (B3).

## Dependencias y orden

1. **Hecho** (Mikel): `bandeja.py` + `sinteticas.py` con tests. Se integra primero en `grupo-9-fichas` para
   que todos partan de ahí.
2. **Sesión de laboratorio** (todos): preparar fichas y bandeja (B1) y grabar los primeros vídeos con la cámara
   cenital (B4). Mikel captura también la hoja A4 de calibración.
3. **En paralelo:** detección (Garai), letras (Xabat), dataset y etiquetado (Markel), calibración y nodo (Mikel).
   Los métodos clásicos se empiezan con sintéticas y se pasan a fotos reales en cuanto las haya.
4. **DL:** depende de las fotos etiquetadas de Markel. Para no esperar, Markel entrega pronto un primer lote
   pequeño. La validación se separa por vídeo (como en `preparar_dataset.py`).
5. **Evaluación** (Markel): necesita las dos cadenas completas. Se reserva un vídeo de test que no se usa ni
   para entrenar ni para ajustar umbrales.
6. **Integración con el robot** (B3, Fase 2), memoria y ensayo de la defensa.

## Para la defensa: revisión cruzada

Todos tenemos que poder explicar todo. Cada bloque lo revisa y lo prueba otra persona antes de darlo por cerrado:

| Autor | Revisa |
|---|---|
| Mikel (bandeja y ROS) | Markel |
| Garai (detección) | Xabat |
| Xabat (letras) | Garai |
| Markel (datos y evaluación) | Mikel |

Antes de la defensa, cada uno debe saber hacer solo: generar sintéticas, pasar una foto por la cadena completa
con `clasico` y con `dl`, explicar ArUco → homografía → mm, explicar una regla del método clásico y cómo se
entrenaron YOLO y la CNN, y lanzar el nodo y ver `/signos/fichas` con `ros2 topic echo`.

## Convenciones

- Lógica en módulos Python puros en `ros_workspace/src/signos_fichas/signos_fichas/` (`bandeja.py`,
  `deteccion.py`, `letras.py`, `evaluacion.py`) con tests de pytest en `test/`. El nodo ROS es una capa fina.
- Una rama por persona a partir de `grupo-9-fichas` (`mikel-hernandez`, …) y PR a `grupo-9-fichas` revisado por
  quien indica la tabla de revisión cruzada. De `grupo-9-fichas` a `main`, con PR como dice `CLAUDE.md`.
- Fotos, vídeos y pesos de modelos en Drive, nunca en git.
- Código, comentarios y commits en español.

## Pendiente

- Medir la distancia real entre los centros de los ArUco (`ANCHO_MM`, `FONDO_MM` en `bandeja.py`; ahora 300 × 250 mm provisionales).
- Primeras fotos reales de la bandeja con la ELP.
- Probar el código en el contenedor (`requirements.txt` con OpenCV ≥ 4.7, la versión del material de ArUco de clase).
