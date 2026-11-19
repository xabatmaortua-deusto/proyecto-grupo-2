# Transcriptor robótico de lengua de signos — Grupo 2

Una persona deletrea una palabra con el alfabeto dactilológico delante de una cámara. Cada vez que se valida una letra,
el robot UR3e (con pinza RG2) coge la ficha de esa letra de una bandeja y la deja, derecha, en el siguiente hueco
de la zona de la palabra. La palabra termina cuando la persona baja los brazos.

Proyecto conjunto de Robótica y Visión por Computador. **No hay enunciado oficial**: este fichero y
`docs/` son la referencia. La entrega es el propio repositorio y la defensa es el día del examen, con una demo en
vivo con el robot real.

## Equipo

- **Equipo 4 — gestos** (Visión): Ainhoa, Keltse, Xabier, Ane.
- **Equipo 9 — fichas** (Visión): Mikel, Garai, Xabat, Markel.
- **Robótica**: los 8. Cada persona trabaja en parte de visión y en parte de robótica.

## Alcance

**Mínimo (aprobado):**
- 6 letras: A, B, L, O, V, W. Palabras de hasta 6 letras, sin comprobar contra un diccionario (vale cualquier secuencia).
- Los dos módulos de visión implementados **dos veces**, una clásica (solo OpenCV) y otra con deep learning,
  y se elige cuál usar al lanzar.
- El robot real forma la palabra letra a letra, con las fichas derechas y en una zona fija.
- Una palabra por ejecución. Las personas vuelven a colocar las fichas en la bandeja a mano.

**Extras (solo cuando el mínimo funcione):** más letras (aunque algunas sean solo para DL), modo "palabra completa",
varias palabras seguidas, corregir letras. En la demo se usan las letras que soporten **los dos** métodos.

**Descartado por ahora:** fichas apiladas y comprobación de que la palabra existe.

## Montaje físico (medidas aproximadas, pendientes de confirmar)

- Robot **205**, uno solo. Su base está en el centro del lado norte de la mesa.
- Mesa de ~65 cm de ancho por ~50 cm de fondo. Cogiendo desde arriba, el UR3e llega cómodo hasta unos **40 cm desde su base**:
  las esquinas del sur (~60 cm) quedan **fuera de alcance**, y no pasa nada. La bandeja y la zona de la palabra tienen que estar dentro de ese radio.
- **Bandeja**: en una esquina del lado norte, con bordes bajos (altura por medir) y ArUco en las esquinas. Fondo oscuro mate.
- **Fichas**: de madera, color natural, de **18 mm de ancho × 20 mm de largo × 5 mm de alto**, con una letra negra mayúscula
  de tipografía sencilla y **una raya bajo la letra** que marca la orientación. Hay al menos 2 fichas de cada letra y
  se garantiza una separación mínima entre ellas.
  La pinza coge por la cara de 18 mm: con solo 5 mm de alto, los dedos tienen que bajar casi hasta la mesa, así que
  la altura de agarre y los bordes de la bandeja se ajustan en `mesa.yaml`, no en el código.
- **Zona de la palabra**: 6 huecos fijos en fila delante del robot. La orientación "derecha" (con la letra legible)
  la marca una cara principal de la mesa.
- **Cámaras**: ELP USB, propias. Una cenital fija en trípode para la bandeja, otra para las manos y,
  opcionalmente, una tercera para la vista general. Se lanzan con `usb_cam` en espacios de nombres distintos (ver `robot_y_camara.launch.py`).

## Arquitectura

```
ros_workspace/src/
├── launcher_robots_lab_robotica/   # BASE de la asignatura: no se modifica
├── rg2_driver/                     # BASE de la asignatura: no se modifica
├── signos_interfaces/              # ament_cmake. Solo .msg: el contrato entre módulos
├── signos_gestos/                  # Equipo 4. ament_python
├── signos_fichas/                  # Equipo 9. ament_python
├── signos_robot/                   # Robótica. ament_python
└── signos_bringup/                 # Launch del sistema, config/mesa.yaml, nodos falsos para simulación
docs/                               # Hitos de visión, decisiones, tareas
herramientas/                       # Fuera de ROS: entrenar, etiquetar, generar ArUco y fichas
requirements.txt                    # Dependencias pip extra (DL, MediaPipe…)
```

```
cámara manos ──► signos_gestos ──/signos/palabra──►┐
                                                   ├─► signos_robot ──► MoveIt 2 + rg2_driver ──► UR3e 205
cámara bandeja ─► signos_fichas ──/signos/fichas──►┘
```

### Contrato (`signos_interfaces`)

Cambiar el contrato afecta a los tres módulos: **solo se cambia con un PR revisado por alguien de cada parte**.

```
# PalabraEnCurso.msg  →  topic /signos/palabra  (lo publica signos_gestos)
string texto        # palabra validada hasta ahora, en mayúsculas: "B", "BO", "BOL"...
bool terminada      # true cuando la persona baja los brazos
```
- Se vuelve a publicar la palabra **entera** cada vez que se valida una letra. El robot coloca `texto[n_colocadas:]`,
  así que si se pierde un mensaje no pasa nada.
- Una letra solo es válida si el gesto se mantiene estable durante un tiempo configurable. La misma letra puede repetirse
  (LOBO) si entre medias hay una pausa o un cambio de gesto.

```
# Ficha.msg
string letra        # mayúscula
float64 x           # m, en el marco world del robot
float64 y           # m, en el marco world del robot
float64 theta       # rad, en el plano XY de world: dirección de la raya → parte de arriba de la letra, medida desde +X
float32 confianza   # 0..1 (en el método clásico, 1.0 si la regla es clara)

# FichaArray.msg  →  topic /signos/fichas  (lo publica signos_fichas, de forma continua)
std_msgs/Header header   # stamp = instante de la imagen; frame_id = "world"
Ficha[] fichas           # TODAS las fichas detectadas dentro de la bandeja
```
- La posición de los ArUco (el tablero) respecto a `world` está en `signos_bringup/config/mesa.yaml` y la calibra el Equipo 9.
  Los nodos de visión publican ya en `world`: el robot no hace conversiones de coordenadas.
- Los dos nodos de visión aceptan el parámetro `metodo:=clasico|dl`, y **los dos métodos publican exactamente el mismo mensaje**.

### Robot (`signos_robot`)

Es una máquina de estados con un único nodo orquestador:
1. Espera a que `/signos/palabra` traiga una letra nueva.
2. Va a `home` (fuera del campo de la cámara de la bandeja) y espera una `FichaArray` con `stamp` posterior a su llegada.
   Se vuelve a detectar **antes de cada ficha**.
3. Elige una ficha con esa letra. Si no hay ninguna, avisa por log, se queda en `home` y lo reintenta cada X s.
4. Coge la ficha: se acerca por encima, baja con velocidad reducida, cierra la pinza, comprueba el agarre (el driver devuelve
   "agarre detectado") y sube. Si falla, lo reintenta hasta N veces.
5. Gira según `theta`, la deja en el hueco `n` con velocidad reducida y vuelve a `home`.

- Hay dos factores de velocidad como parámetros: uno general y otro de aproximación (coger/dejar), más bajo. Se empieza con valores bajos (≤ 0,1).
- La mesa, la bandeja (con bordes) y la zona de la palabra son objetos de colisión en MoveIt.
- Las medidas, los huecos, las alturas, N, X y las velocidades están en `mesa.yaml`, no escritos en el código.
- Se parte del patrón de `ejemplos/control_robot.py` (rclpy + acciones de `move_group`).

## Reglas de desarrollo

- **Idioma**: español en el código (variables, funciones, comentarios), en los mensajes de log y en la documentación, como en el repo base.
- **No se modifican** `launcher_robots_lab_robotica` ni `rg2_driver`. Si hace falta otro launch o config, va en `signos_bringup`.
- **La visión no depende de ROS por dentro**: la lógica (segmentar, clasificar, estabilizar letras, ArUco…) va en módulos
  Python puros que se prueban con fotos o vídeos grabados. El nodo ROS es una capa fina que lee la imagen, llama a la lógica y publica.
- **Simulación primero**: el robot se desarrolla en `sim` con los nodos falsos de `signos_bringup`. En el laboratorio
  solo se cambian los nodos falsos por los reales.
- **Pruebas**: toda lógica no trivial (reglas de clasificación, estabilizador de letras, cálculo de pose,
  elegir ficha, máquina de estados) tiene al menos un test pequeño (`pytest` dentro del paquete) con datos reales o grabados.
- **Fuera de git**: datasets, vídeos y pesos de modelos (`*.pt`, `*.onnx`…). Van en la carpeta compartida del grupo
  (Drive), y en el repo solo se documenta dónde están y cómo se generan.
- **Dependencias**: las extra van en `requirements.txt` (`pip install -r requirements.txt` dentro del contenedor). No se toca el Dockerfile
  mientras las librerías no estén decididas.
- **Hacerlo simple**: no se añaden abstracciones, parámetros ni paquetes "por si acaso". Lo más corto que funcione y se entienda.

### Git

- `main` siempre compila (`colcon build`) y funciona en simulación.
- Una rama por tarea (`gestos/...`, `fichas/...`, `robot/...`, `bringup/...`), y se integra en `main` con un **pull request revisado por otra persona**.
- Mensajes de commit en español, en presente y cortos.

## Entorno

- Se trabaja en Linux o WSL con Docker Engine y los perfiles **`local`** / **`local_gpu`** (con GPU para el DL). El contenedor `desktop` no se usa.
  - `./lab.sh iniciar --lab` (o `--lab-gpu`), `./lab.sh compilar`, `./lab.sh terminal`.
  - Simulación: `./lab.sh sim 205` · Robot real: `./lab.sh robot 205` (desde la red del laboratorio).
  - Conexión desde portátil Windows: `docs/ROBOT_DESDE_WINDOWS.md`.
- Workspace en el contenedor: `/home/laboratorio/ros_workspace`. Hay que hacer `source install/setup.bash` en cada terminal.
- Solo se conserva `ros_workspace/` si se borra el contenedor: lo instalado con pip hay que volver a instalarlo con `requirements.txt`.
- Pendiente de verificar con el robot real: la pinza RG2 y las cámaras USB dentro del contenedor.

## Seguridad en el laboratorio

- Cada movimiento nuevo se prueba antes en simulación. Con el robot real, la primera ejecución se hace a velocidad mínima.
- Siempre hay alguien con la seta de emergencia o el teach pendant a mano mientras el robot se mueve.
- Nadie mete las manos en la zona de trabajo del robot durante la ejecución. La cámara de gestos está fuera de su alcance.
