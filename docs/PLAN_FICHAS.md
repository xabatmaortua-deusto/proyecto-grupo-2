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
deteccion.detectar_clasico(enderezada) -> [{'x': mm, 'y': mm, 'angulo': rad, 'recorte': imagen, 'confianza': 0..1}, ...]
deteccion.detectar_dl(enderezada)      -> igual
letras.reconocer_clasico(recorte)      -> (letra, giro, confianza)   # giro: 0, 1, 2 o 3 cuartos de vuelta
letras.reconocer_dl(recorte)           -> igual
theta = deteccion.theta_en_bandeja(angulo, giro)                     # = angulo + giro * 90°
```

Como la ficha mide 18 × 20 mm y es casi cuadrada, la detección no sabe hacia dónde mira la letra: por eso la
etapa 3 decide entre 4 giros. `giro` es el `k` tal que `np.rot90(recorte, k)` deja la letra derecha (raya abajo);
el convenio completo está en [`DETECCION_FICHAS.md`](DETECCION_FICHAS.md).

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

## Sesión de fotos (B1 y B4): lista de comprobación

Las fotos del 23-11 no bastan: no tienen O, V ni W, solo se ve un ArUco y la exposición automática quema la madera
(se pierde el número pequeño de Scrabble, que sirve para la orientación). Las fichas reales son de Scrabble y **no
tienen raya**: el giro sale de la forma de la letra. Con una sesión bien planificada salen a la vez los datos de
letras (clásico y DL), el test de detección y la evaluación final.

**Antes de grabar**
- [ ] Montaje definitivo, el de la demo: bandeja con los 4 ArUco visibles y la ELP cenital en el trípode a la altura final.
- [ ] `ANCHO_MM` y `FONDO_MM` medidos (si no, las fotos se enderezan con un solo ArUco, como las del 23-11).
- [ ] Cámara en manual: exposición, balance de blancos y enfoque fijos. Apuntar la resolución y los ajustes.
- [ ] Fichas: O, V, W, A, B y L, al menos 2 de cada una, y que salgan todas (la impresión cambia de una a otra).
- [ ] Sin manos en la imagen: mover las fichas, sacar la mano y esperar 2-3 s quietos (como un stop-motion).

**Tipo 1: una letra por vídeo, todas derechas (entrenar)**
- [ ] Un vídeo por letra, solo con las fichas de esa letra.
- [ ] Todas con la parte de arriba de la letra hacia el lado de los ArUco 0 y 1, torcidas ±30° como mucho. Así el
      giro es siempre 0 y la letra la da el nombre del vídeo: no hay que etiquetar a mano. Los otros giros se
      generan con `np.rot90`.
- [ ] Por toda la bandeja: centro, bordes, esquinas y cerca de los ArUco. 30-60 s y 10-15 posiciones por vídeo.

**Tipo 2: escenas mezcladas, como en la demo (validar y test)**
- [ ] Las 6 letras juntas, en cualquier orientación, separadas lo mínimo que garantiza el montaje (sin tocarse).
- [ ] Algunas fichas giradas exactamente 90° y 180°.
- [ ] Unas fotos de la bandeja vacía (detecciones falsas).
- [ ] Opcional: alguna ficha de otra letra (C, D, E…) para ver qué hace el sistema con una letra que no conoce.

**Luz**: cada tipo, con tres luces.
- [ ] La del laboratorio tal cual (la de la demo).
- [ ] Con sombra: alguien o el brazo del robot tapando parte de la bandeja.
- [ ] Lateral o más baja: lámpara de lado o parte de las luces apagadas.

**Test**
- [ ] Un bloque de vídeos del tipo 2 reservado para el test, mejor de otro día u otra hora. No se usa para
      entrenar ni para ajustar umbrales.
- [ ] De 20-30 de sus fotos, marcar a mano la letra y el giro de cada ficha.

**Organización**
- [ ] Una carpeta por sesión: `ros_workspace/datasets/fichas/crudo/<fecha>/`, y lo mismo en Drive.
- [ ] Nombres que digan qué hay: `t1_O_luzlab.mp4`, `t1_V_sombra.mp4`, `t2_mezcla_lateral.mp4`, `test_mezcla_luzlab.mp4`…
- [ ] Un `notas.txt` con los ajustes de la cámara, la luz de cada vídeo y qué lado son los ArUco 0 y 1.

**Mínimo si no hay tiempo**: tipo 1 de las 6 letras con la luz del laboratorio, tipo 2 con dos luces y un vídeo de test.

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
- ~~Primeras fotos reales de la bandeja con la ELP~~: hechas el 23-11. Falta una segunda sesión con O, V y W
  (ver la lista de comprobación de la sesión de fotos).
- ~~Probar el código en el contenedor~~: hecho. El OpenCV de apt es 4.6 y no tiene `ArucoDetector`; con
  `pip install --break-system-packages -r requirements.txt` (OpenCV 4.11, numpy 1.26) pasan `colcon build` y
  `colcon test` de `signos_fichas`.
