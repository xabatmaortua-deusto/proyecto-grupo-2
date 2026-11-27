# Medición de distancias con marcadores ArUco

Sistema de visión para medir **distancias reales (cm) entre marcadores ArUco** con una cámara USB sin calibrar, independientemente de la perspectiva y de la distancia focal. También sirve para calibrar la perspectiva (homografía) y la relación píxel/metro de una cámara sobre un plano.

La escala métrica sale exclusivamente del **tamaño conocido de los marcadores** (80 mm de lado), no de los parámetros intrínsecos de la cámara.

---

## Contenido

| Archivo | Qué es |
|---|---|
| `generar_arucos.py` | Genera la hoja de calibración (`arucos_calibracion_A4.pdf`) y su layout (`arucos_calibracion_A4.json`). |
| `arucos_calibracion_A4.pdf` | Hoja A4 para imprimir: 4 marcadores `DICT_4X4_50`, IDs 0-3, lado 80 mm. |
| `arucos_calibracion_A4.json` | Coordenadas reales (m) de centros y esquinas de cada marcador en la hoja. |
| `capturar.py` | Guarda frames crudos del streaming de la cámara en `capturas/`. |
| `calibrar.py` | Calibración ligera (distorsión de lente + offset de borde) a partir de `capturas/`. Escribe `calibracion_camara.json`. |
| `medir_arucos.py` | Script principal: detección y medición en tiempo real (o sobre una imagen). |
| `calibracion_camara.json` | Resultado de `calibrar.py` (se carga automáticamente si existe). *No versionar entre cámaras distintas.* |

Dependencias: Python 3, `opencv-python` ≥ 4.7 (API `cv2.aruco.ArucoDetector`; probado con 4.13), `numpy`, `scipy`, `matplotlib` (sólo para generar el PDF). La ventana de vídeo requiere OpenCV con soporte GUI (Qt/GTK).

---

## Flujo de trabajo rápido

```bash
# 1. Generar e imprimir la hoja (imprimir a TAMAÑO REAL / 100 %, sin "ajustar a página")
python3 generar_arucos.py

# 2. Capturar la hoja desde varios ángulos/distancias (ESPACIO guarda, a = auto, q sale)
python3 capturar.py                       # o: --no-gui --interval 1 --count 40

# 3. Calibrar (distorsión + offset de borde) -> calibracion_camara.json
python3 calibrar.py capturas/

# 4. Medir en tiempo real
python3 medir_arucos.py                   # ventana con distancias superpuestas
python3 medir_arucos.py --no-gui          # sólo consola (1 línea/segundo)
python3 medir_arucos.py --image foto.png  # una imagen
```

La cámara USB de este equipo es `/dev/video2` (índice `2`, valor por defecto de `--cam`). `/dev/video0` es la webcam integrada. Los nodos impares (`video1`, `video3`) son nodos de metadatos y no dan imagen. Para listar cámaras: `for d in /sys/class/video4linux/video*; do echo "$d $(cat $d/name)"; done`.

**Windows y macOS.** Los scripts eligen solos el backend de vídeo (V4L2 en Linux, DirectShow/Media Foundation en Windows, AVFoundation en macOS). Fuera de Linux, la cámara por defecto es la `0`; si no es la buena, `python medir_arucos.py --list-cams` lista las que dan imagen y luego se elige con `--cam 1`, `--cam 2`… En Windows el comando suele ser `python` (o `py`) en vez de `python3`. Para que se abran las ventanas hace falta `opencv-python`, no la versión `-headless`.

---

## 1. La hoja de calibración (`generar_arucos.py`)

- Diccionario `DICT_4X4_50`, IDs **0 (sup-izq), 1 (sup-der), 2 (inf-izq), 3 (inf-der)**.
- Lado del marcador: **80 mm**, incluido el borde negro (6×6 módulos de 13,33 mm: 4×4 bits + 1 módulo de borde por lado).
- Centros (mm, origen en la esquina superior izquierda de la hoja, X a la derecha, Y hacia abajo): 0 → (55, 75), 1 → (155, 75), 2 → (55, 225), 3 → (155, 225).
- Distancias reales entre centros: **0-1 y 2-3 = 10 cm**, **0-2 y 1-3 = 15 cm**, **diagonales = 18,03 cm**.
- Se dibujan como rectángulos vectoriales → el tamaño impreso es exacto. Incluye una regla de 100 mm para comprobar la escala de impresión.
- El JSON contiene las esquinas de cada marcador en el orden que devuelve OpenCV: TL, TR, BR, BL.

> **Escala de impresión.** Un escalado *uniforme* de la impresora **no afecta** a las medidas: el marcador y la separación cambian en la misma proporción y la distancia medida en "unidades de marcador" no varía. Lo que sí importa es que el valor de `--size` sea el lado real impreso si usas marcadores sueltos, y que el escalado no sea anisótropo (X ≠ Y).

Para cambiar tamaño, IDs o posiciones, edita las constantes al principio de `generar_arucos.py`. Si los marcadores van a estar más separados o lejos de la cámara, conviene imprimirlos más grandes (uno por hoja).

---

## 2. Captura de imágenes (`capturar.py`)

Guarda los frames **sin anotaciones** (PNG sin pérdida por defecto) en `capturas/img_YYYYMMDD_HHMMSS_mmm.png`. La previsualización muestra cuántos marcadores ve la cámara, pero esto no se graba en la imagen.

| Opción | Por defecto | Descripción |
|---|---|---|
| `--cam` | `2` en Linux, `0` en Windows/macOS | Índice o ruta del dispositivo |
| `--out` | `capturas` | Carpeta de salida |
| `--width/--height` | 1280×720 | Resolución solicitada (la cámara puede elegir otra; la USB actual da **1024×768**) |
| `--interval` | 0 | Guardado automático cada N s |
| `--count` | 0 | Parar tras N imágenes |
| `--fmt` | `png` | `png` o `jpg` |
| `--no-gui` | — | Sin ventana (requiere `--interval`) |

Para calibrar, las mejores capturas son las de la hoja completa con los 4 marcadores nítidos, desde **ángulos variados**, inclinaciones distintas, varias distancias y con los marcadores repartidos por **todo el campo de visión** (incluidas las esquinas de la imagen, donde la distorsión es mayor).

---

## 3. Medición (`medir_arucos.py`): pipeline técnico

Para cada frame:

### 3.1 Detección
`cv2.aruco.ArucoDetector` con `CORNER_REFINE_NONE`. El refinado de OpenCV se desactiva porque el propio (siguiente paso) es mejor con imágenes borrosas o sobreexpuestas (ver §5). Sólo se usan los IDs de `--ids`.

### 3.2 Refinado sub-píxel por bordes (`edge_refine`)
Para cada uno de los 4 lados exteriores del marcador:
1. Se toman 40 puntos a lo largo del lado (del 12 % al 88 %, evitando las esquinas).
2. En cada punto se muestrea un perfil de intensidad perpendicular al borde, de ±0,6 módulos (imagen suavizada con un gaussiano σ=1, interpolación bilineal con `cv2.remap`). Hacia dentro sólo hay borde negro (1 módulo), así que el perfil no se contamina con los bits.
3. Los niveles locales de negro y blanco son las medianas de los extremos del perfil. El borde es el cruce con el **nivel medio (50 %)**, interpolado sub-píxel. Se descartan perfiles con contraste < 20 niveles.
4. Se ajusta una recta robusta (`cv2.fitLine`, `DIST_HUBER`).
5. En la última iteración, la recta se desplaza `edge_offset_px` hacia fuera (compensación de la sobreexposición, ver §4).

Las esquinas son las intersecciones de rectas consecutivas. Se hacen 2 iteraciones. Si un lado tiene menos de 10 puntos válidos o una esquina salta más del 25 % del lado, el marcador se descarta en ese frame.

### 3.3 Corrección de la lente (`LensModel`)
Modelo radial polinómico en coordenadas normalizadas por la semidiagonal de la imagen, con el centro óptico supuesto en el centro de la imagen:

```
v = (p - c) / R          r² = |v|²
p_corr = c + R · v · (1 + k1·r² + k2·r⁴)
```

Se aplica a las esquinas *después* del refinado, que trabaja sobre la imagen original. Al no depender de la resolución en píxeles, el modelo sirve aunque esta cambie (con la misma relación de aspecto). `edge_offset_px` se reescala proporcionalmente al ancho.

### 3.4 Solver en modo plano (por defecto, sin intrínsecos)
**Hipótesis: todos los marcadores están en el mismo plano** (pueden estar en cualquier posición y orientación dentro de él).

Incógnitas:
- `G`: homografía plano → imagen (8 parámetros, `G33 = 1`).
- Para cada marcador salvo el de referencia: `(θ, tx, ty)`, su giro y posición en el plano (en metros).

El marcador de referencia (el de mayor área en la imagen) fija el origen y la orientación del plano. Cada esquina del marcador *i* se modela como `G · (R(θi)·esquina_local + ti)`, donde `esquina_local` es la esquina de un cuadrado de lado `--size`. Se minimiza el error de reproyección de **todas las esquinas a la vez** con Levenberg-Marquardt (`scipy.optimize.least_squares`).

Inicialización: homografía del marcador de referencia; los demás marcadores se proyectan al plano con ella y su pose se obtiene por ajuste rígido de Procrustes (SVD).

¿Por qué funciona sin focal? Una homografía plano-imagen absorbe perspectiva, focal, posición y orientación de la cámara. Que cada marcador sea un **cuadrado rígido de tamaño conocido** fija la estructura métrica del plano (escala, ángulos y razón de aspecto). Con 2 marcadores ya hay 16 ecuaciones para 11 incógnitas. Usar todos los marcadores juntos evita extrapolar la homografía de un solo marcador pequeño a todo el plano, que era la principal fuente de error.

El resultado son los centros de los marcadores en metros sobre el plano. Las distancias se calculan entre centros para todos los pares, y el **error de ajuste** (RMS de reproyección en px) es un indicador de calidad.

### 3.5 Solver en modo 3D (`--calib`, opcional)
Si se dispone de una calibración intrínseca completa de OpenCV (`.npz` con `K`/`dist` o `camera_matrix`/`dist_coeffs`, o un `.yaml` de `FileStorage` con `camera_matrix`/`distortion_coefficients`), se usa `solvePnP(SOLVEPNP_IPPE_SQUARE)` por marcador y se mide la distancia 3D entre los `tvec`. Este modo **no requiere coplanaridad**. En él, la distorsión se gestiona con `dist` de OpenCV, así que conviene usar también `--no-cal`.

### 3.6 Control de calidad y suavizado
Un frame se considera **bueno** si su error de ajuste es ≤ `--max-rms` (1,5 px) y su nitidez (varianza del Laplaciano) es ≥ `--min-sharp` (6). Sólo los frames buenos actualizan la media móvil exponencial por par (`--smooth`, α = 0,3). Un par que no se ve durante 15 frames se olvida. En pantalla, las líneas amarillas son medidas fiables y las naranjas indican baja calidad.

### Opciones de `medir_arucos.py`

| Opción | Por defecto | Descripción |
|---|---|---|
| `--cam` | `2` en Linux, `0` en Windows/macOS | Dispositivo de vídeo |
| `--image` | — | Procesar una imagen y salir |
| `--size` | `0.08` | Lado del marcador (m), incluido el borde negro |
| `--ids` | `0 1 2 3` | IDs a usar |
| `--dict` | `DICT_4X4_50` | Diccionario ArUco |
| `--cal` | `calibracion_camara.json` | Calibración ligera |
| `--no-cal` | — | No aplicar la calibración ligera |
| `--no-refine` | — | Usar las esquinas crudas de OpenCV |
| `--calib` | — | Intrínsecos de OpenCV → modo 3D |
| `--smooth` | `0.3` | α de la EMA (1 = sin suavizado) |
| `--max-rms` | `1.5` | Umbral de error de ajuste (px) |
| `--min-sharp` | `6` | Umbral de nitidez |
| `--no-gui` | — | Sin ventana |

Teclas en la ventana: `q`/`ESC` sale, `s` guarda una captura **anotada** (para capturas crudas usar `capturar.py`).

---

## 4. Calibración ligera (`calibrar.py`)

```bash
python3 calibrar.py capturas/ [--sheet arucos_calibracion_A4.json] [--out calibracion_camara.json]
```

Usa las imágenes con ≥ 3 marcadores (`--min-markers`) y nitidez ≥ 6. Estima en dos pasos:

1. **Distorsión radial `k1, k2`, no supervisada.** Busca (Nelder-Mead) los coeficientes que minimizan el error de ajuste al plano de todas las imágenes. **No usa las distancias reales**: una lente distorsionada impide que las esquinas encajen en un plano, y corregirla reduce ese error.
2. **Offset de borde `edge_offset_px`, supervisado.** Con el layout conocido de la hoja (`--sheet`), busca el desplazamiento de borde que anula el sesgo de escala medio de las distancias. Compensa que la sobreexposición o el desenfoque "encogen" el negro del marcador, lo que hace que el marcador parezca más pequeño y las distancias salgan más largas. Se puede omitir con `--no-offset`.

El script imprime una tabla del error antes y después de cada corrección y guarda:

```json
{ "k1": 0.0638, "k2": 0.0184, "edge_offset_px": 0.91, "width": 1024, "height": 768,
  "n_images": 24, "sheet": "arucos_calibracion_A4.json" }
```

(Valores obtenidos el 2026-11-12 con la cámara "HD USB Camera" a 1024×768.)

**Cuándo recalibrar:** al cambiar de cámara, de zoom o de enfoque, al cambiar mucho la iluminación o la exposición (el offset depende de ella) y al cambiar la relación de aspecto. El offset es un parámetro empírico y es lo primero que conviene revisar si aparece un sesgo de escala constante.

---

## 5. Resultados de validación

Se usaron 49 capturas reales de la hoja impresa (2026-11-12, 1024×768, imágenes sobreexpuestas y algo movidas). Se descartaron 11 inútiles (menos de 2 marcadores o nitidez < 5). Errores medidos frente a las distancias reales de la hoja (10 / 15 / 18,03 cm):

**Comparativa de refinado de esquinas** (sin corrección de lente, imágenes con ≥ 3 marcadores):

| Método | Sesgo | Desv. | Máx. | Error de ajuste |
|---|---|---|---|---|
| OpenCV `NONE` | +3,3 mm | 1,6 mm | 10,2 mm | 1,44 px |
| OpenCV `SUBPIX` (versión inicial) | +3,5 mm | 1,5 mm | 9,7 mm | 1,47 px |
| OpenCV `CONTOUR` | +4,2 mm | 1,7 mm | 11,3 mm | 1,35 px |
| OpenCV `APRILTAG` | +2,6 mm | 0,9 mm | 4,6 mm | 1,36 px |
| **Por bordes (propio)** | +2,3 mm | 0,8 mm | 5,8 mm | 1,34 px |

**Efecto de la calibración ligera** (24 imágenes útiles, error relativo):

| Corrección | Error de ajuste | Sesgo | Desv. | Máx. |
|---|---|---|---|---|
| Ninguna | 1,34 px | +1,56 % | 0,37 % | 2,91 % |
| + distorsión | 0,53 px | +0,93 % | 0,48 % | 2,84 % |
| + offset de borde | 0,54 px | 0,00 % | 0,44 % | 1,69 % |

**Validación cruzada** (calibración con la mitad de las imágenes y evaluación con la otra mitad): el sesgo pasa de +2,2 mm a −0,2 mm, la desviación de 0,87 a 0,68 mm y el máximo de 5,5 a 3,0 mm. La calibración generaliza y no está sobreajustada.

**Imagen sintética** (hoja renderizada con perspectiva fuerte, sin distorsión): errores ≤ 0,6 mm.

Rendimiento: ~16 fps en directo a 1024×768, limitado por la cámara.

---

## 6. Limitaciones y buenas prácticas

- **Coplanaridad**: el modo plano supone que todos los marcadores están en el mismo plano. Si no lo están, el error de ajuste sube y las distancias se falsean; en ese caso usa el modo 3D con `--calib`.
- **Las distancias son entre centros** de marcadores, medidas sobre el plano.
- **Precisión ≈ 0,5 % típico, < 2 % en el peor caso** con las capturas actuales. El error crece con el desenfoque, el movimiento, los marcadores pequeños en la imagen (< ~100 px de lado) y los ángulos muy rasantes.
- **Lo que más mejora la precisión**: enfocar bien, **reducir la exposición** (evitar que el blanco se sature) y una iluminación difusa sin reflejos (el papel mate es mejor que el brillante). Además, la hoja o los marcadores deben estar planos y rígidos.
- El centro de distorsión se supone en el centro de la imagen y sólo se modela la distorsión radial. Para lentes gran angular u ojo de pez es preferible una calibración completa de OpenCV (tablero ChArUco) y el modo 3D.
- `edge_offset_px` es empírico y depende de la exposición: si cambias la iluminación, recalibra.
- Con `--size` distinto de 0,08 (marcadores sueltos de otro tamaño), mide el lado impreso real con una regla.

---

## 7. Uso como librería (para otros scripts/agentes)

```python
import cv2, medir_arucos as m

det = m.make_detector("DICT_4X4_50")
lens = m.LensModel.load("calibracion_camara.json")   # o None
gray = cv2.cvtColor(cv2.imread("foto.png"), cv2.COLOR_BGR2GRAY)

det_pix, det_und, _, _ = m.detect(gray, det, ids_wanted=[0, 1, 2, 3], lens=lens)
# det_pix: {id: (4,2)} esquinas refinadas en la imagen (para dibujar)
# det_und: {id: (4,2)} esquinas corregidas de distorsión (para medir)
centers, rms = m.solve_plane(det_und, size_m=0.08)   # {id: (x,y) en m}, error de ajuste en px
```

Funciones principales: `edge_refine(gray_float, corners, offset_px)`, `LensModel.undistort(pts, shape)`, `solve_plane(det, size_m)`, `solve_3d(det, size_m, K, dist)` y `sharpness(gray)`.

Para obtener una homografía imagen → plano métrico (por ejemplo, para enderezar la imagen con `cv2.warpPerspective`) con la hoja de calibración, usa las esquinas de `det_und` y las `corners_m` de `arucos_calibracion_A4.json` con `cv2.findHomography`. Recuerda que esa homografía actúa sobre coordenadas ya corregidas de distorsión.

### Notas para agentes
- Para probar sin cámara: `python3 medir_arucos.py --image <png> --no-gui`. La salida son líneas `a-b: X.XX cm`.
- Al ejecutar en directo con `timeout`, usa `python3 -u` y `timeout -s INT` para que la salida no se pierda en el buffer.
- La cámara USB tarda ~0,5 s en dar el primer frame; el primer segundo reporta menos fps.
- Las capturas de `capturar.py` son la fuente de verdad para depurar y recalibrar. Las de la tecla `s` de `medir_arucos.py` están anotadas y **no** sirven para calibrar.

---

## Estado actual

- `capturas/` y `calibracion_camara.json` **no están presentes** en el directorio (se eliminaron después de la validación). Hasta recalibrar, `medir_arucos.py` funciona sin corrección de lente (sesgo esperado ≈ +1,5 %). Para restaurarlos, captura de nuevo la hoja con `python3 capturar.py` y ejecuta `python3 calibrar.py capturas/`.
