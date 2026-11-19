# Preguntas abiertas — Transcriptor robótico de lengua de signos (Grupo 2\)

Responded debajo de cada pregunta (`R:`). Si una no aplica o aún no se sabe, poned "no se sabe" y quién lo averigua. Las marcadas con ⭐ bloquean el arranque: son las que más prisa corren.

---

## 0\. Propuesta de estructura del repositorio (para validar)

El repo base es un workspace ROS 2 Jazzy dentro de Docker. Solo `ros_workspace/` se monta en el contenedor, así que **todo el código ROS del proyecto vive en `ros_workspace/src/`**. Los paquetes de la asignatura (`launcher_robots_lab_robotica`, `rg2_driver`) no se tocan, para poder traer actualizaciones del profesor sin conflictos.

```
ros_workspace/src/
├── launcher_robots_lab_robotica/   # BASE (no tocar): URDF, MoveIt, lanzadores sim/real, cámara
├── rg2_driver/                     # BASE (no tocar): pinza
├── signos_interfaces/              # Mensajes/servicios propios (ament_cmake). Contrato entre equipos
├── signos_gestos/                  # Equipo 4: cámara de la mano -> palabra (clásico + DL)
├── signos_fichas/                  # Equipo 9: cámara de la bandeja -> fichas con pose en mm (clásico + DL)
├── signos_robot/                   # Robótica (los 8): pick & place, orquestador
└── signos_bringup/                 # Launch files del sistema completo + config (zonas, calibración)
docs/                               # Hitos, especificaciones, diagramas
herramientas/                       # Scripts fuera de ROS: entrenamiento, etiquetado, generar fichas/ArUco
```

Ideas clave:

- **`signos_interfaces` es el contrato**: se define el primer día y a partir de ahí cada equipo trabaja en paralelo.  
- Cada equipo de visión puede probar sin robot (vídeo/fotos grabadas) y robótica puede probar sin visión (un nodo falso que publica fichas en posiciones inventadas en la simulación).  
- Datasets y pesos de modelos **fuera de git** (o Git LFS / Drive), solo se versiona el código.

**P0.1 ⭐** ¿Os parece bien esta estructura? ¿Algún prefijo distinto a `signos_`? **R:** Si, asi bien.

---

## 1\. Alcance y objetivo final

**P1.1 ⭐** ¿Qué tiene que hacer la demo final, exactamente? Ejemplo: "una persona deletrea BOLA, el robot coloca B-O-L-A en una fila en una zona marcada". ¿Es en vivo con el robot real o vale un vídeo / simulación? **R:** Si, la persona deletrea la palabra, se interpreta y se va escribiendo. En vivo. Aunque podemos hacer pruebas con videos.

**P1.2** ¿Cuál es el conjunto de letras objetivo para la demo integrada? ¿Las 6 comunes (A, B, L, O, V, W) o el abecedario que soporte el DL? (Equipo 4 llega a 24 letras estáticas; Equipo 9 dice "todo el abecedario"; las que no coincidan no sirven en el sistema conjunto.) **R:** La idea es empezar con algunas letras estáticas (6 aproximadamente) y detectarlas de forma manual y a través de DL. A partir de ahí, se irán añadiendo letras adicionales según el tiempo del que dispongamos (depende de su complejidad, quizás se integren algunas solo para el modo de detección de DL). Para la demo, imagino que habrá que probar el programa tanto con la detección manual como la detección de DL, por lo que habrá que utilizar aquellas letras comunes en ambos métodos.

**P1.3** ¿Las palabras deben existir (diccionario) o vale cualquier secuencia? **R:** Palabras que existan.

**P1.4** ¿Se forma una sola palabra por ejecución o varias seguidas? Si son varias, ¿quién recoge/recoloca las fichas entre palabras (persona o robot)? **R:** En principio una sola palabra. Luego, vemos si podemos llegar a hacer. Colocamos nosotros las fichas.

**P1.5** ¿Qué pasa con letras repetidas (LOBO \= dos O)? ¿Cuántas fichas hay de cada letra en la bandeja? **R:** Habrá fichas que tengan la misma letra, para casos como este. Mínimo 2 fichas por letra

**P1.6** ¿Qué es "lo mínimo aprobable" y qué es "extra"? (Ej.: mínimo \= 6 letras, palabras ≤4 letras, robot real; extra \= DL, dos robots, más letras…) **R:** Lo mínimo aprobable es que funcione con las 6 letras, que se creen las palabras mediante el robot y que los dos subproyectos de visión se hagan mediante deep learning y de forma manual (se podrá seleccionar si usar el método manual o el de deep learning).

---

## 2\. Evaluación, plazos y organización

**P2.1 ⭐** Fechas de entregas/hitos de Robótica y de Visión. ¿Qué se entrega en cada uno (código, memoria, vídeo, presentación)? **R:** Se entrega todo el mismo día Todo. El repositorio es la entrega. La defensa es el mismo día del examen.

**P2.2** ¿Qué criterios de evaluación hay para la parte de Robótica? (Para priorizar: ¿se valora más planificación de trayectorias, escena de colisiones, robustez, integración…?) **R:** Que se identifiquen las fichas, que el robot vaya cogiendo las fichas y que se forme la palabra en un lugar específico (ficha a ficha). Aun así, eso sería lo básico, cuando funcione se buscará realizar algo más ambicioso con alguna idea que surja. Se programará todo movimiento mediante ros2. Planificación de trayectorias y lo robusto que sea al integrarlo.

**P2.3** ¿Los equipos de visión tienen que entregar su parte por separado en Visión por Computador? ¿Pueden vivir en este mismo repo o cada uno necesita su repo? **R:** Todo puede vivir en el mismo repositorio, y también es lo ideal.

**P2.4** ¿Cómo os vais a repartir los 8 la parte de Robótica? ¿Subgrupos fijos o rotación? **R:** Habrá integrantes de los subgrupos de visión que trabajen en el robot más que los demás, no está fijado quién exactamente todavía. Aunque luego cada uno trabaje un poco en todo. Lo ideal es repartir las tareas de manera equitativa y que todos hagan parte de lo de visión y de lo de robótica.

**P2.5** ¿Hay un repositorio remoto (GitHub del grupo)? ¿Flujo de trabajo: ramas por tarea \+ pull request con revisión de otra persona, o push directo a `main`? **R:** Si, [https\://github.com/xabatmaortua-deusto/proyecto-grupo-2](https://github.com/xabatmaortua-deusto/proyecto-grupo-2) . Flujo de trabajo: ramas por tarea \+ pull request/merge al main

**P2.6** Idioma del código y documentación: el repo base usa español en nombres de variables y comentarios. ¿Seguimos así? **R:** Si, seguimos así.

---

## 3\. Hardware y laboratorio

**P3.1 ⭐** ¿Qué robot(s) tenemos asignados (203, 204, 205, 206)? ¿Uno o una pareja? (El repo soporta escenas de dos robots: ¿lo queremos usar, p. ej. uno coge y otro coloca?) **R:** Tenemos un solo robot (de un brazo), que es el 205\.

**P3.2 ⭐** ¿Cuándo tenemos acceso al laboratorio / robot real? ¿Horas por semana? (El README dice que el robot real, la cámara USB y la pinza real **no están probados** en este port a ROS 2.) **R:** Tenemos acceso al robot varias veces a la semana. La cámara es nuestra. Hemos conseguido conectar el ordenador al robot (mira la conversación del fork). Queda verificar la pinza

**P3.3** ¿Cuántas cámaras hay y cuáles? ¿Una para la mano y otra cenital para la bandeja? ¿Modelo/resolución? ¿Las da el laboratorio o son vuestras (webcam del portátil)? **R:** Tenemos 5 cámaras en total. En principio, la idea es utilizar una para visualizar la bandeja y otra para observar los gestos del lenguaje de signos y detectar las palabras. Se puede usar una tercera para una visión general de la mesa. Las cámaras son ELP 120fps USB Kamera, con 12 Megapixeles, USB.

**P3.4** La cámara de la bandeja: ¿fija sobre la mesa (cenital) o montada en el robot (eye-in-hand)? ¿Ya hay soporte físico? **R:** La cámara que observa la bandeja se sujeta fija mediante un trípode sobre la mesa. No estará montada en el robot.

**P3.5** ¿Dimensiones de la mesa y dónde va la bandeja respecto a la base del robot? El UR3e alcanza \~500 mm: la bandeja y la zona de colocación tienen que caber dentro. **R:** La mesa mide 50cm x 60cm. si el robot no llega a alcanzar todo, ¿pasa algo?

**P3.6** ¿Hay GPU disponible para el DL (YOLO)? El contenedor `desktop` no tiene GPU. ¿El DL se ejecuta en el PC del laboratorio (perfil `local_gpu`), fuera de Docker, o en CPU? **R:** Si, hay GPU disponible. No se usa el contenedor desktop, se usará local o local\_gpu. Iremos viendo cual de las opciones es mejor. Tenlo en cuenta.

---

## 4\. Las fichas (física y agarre)

**P4.1 ⭐** Tamaño y **grosor** de las fichas, y material. La pinza RG2 tiene que agarrarlas por los lados: una ficha de 2 mm de grosor apoyada en la mesa es casi imposible de coger con una pinza paralela. ¿Hay margen para hacerlas gruesas (≥ 8–10 mm, tipo bloque/cubo)? **R:** Si, las fichas tienen el grosor suficiente para agarrarlas fácilmente. Forma circular, tamaño/grosor (?) y son de madera

**P4.2** ¿Quién las fabrica y cómo (impresión 3D, madera, cartón pluma \+ pegatina)? ¿Para cuándo? **R:** Ya existen. Las fichas son de colores y tenemos que añadir la letra sobre ellas.

**P4.3** ¿Las fichas pueden estar tocándose o apiladas en la bandeja, o se garantiza separación mínima? (Afecta a la visión clásica y a si la pinza cabe entre fichas.) **R:** Se garantiza una separación mínima inicial. Puede que haya que verificar esa separación. También podemos apilar las que tengan la misma letra en un montón. Veámos cuál es la mejor opción.

**P4.4** ¿La bandeja tiene bordes? Altura de los bordes (colisión de la pinza). **R:** Si tiene bordes. Altura por confirmar. Algo no muy alto.

---

## 5\. Colocación de la palabra

**P5.1 ⭐** ¿Dónde se forma la palabra? ¿Zona fija marcada en la mesa, otra bandeja, un soporte/atril? ¿En fila horizontal de izquierda a derecha desde el punto de vista de quién? **R:** En una zona de la mesa, que esté vacía/haya espacio. Alineada con una cara “principal”, que será la orientación base.

**P5.2** ¿Las fichas se dejan con la letra derecha (alineada para leerla)? Equipo 9 usa la raya bajo la letra para saber la orientación: ¿hay que girar la ficha antes de soltarla? **R:** Si, hay que dejar la ficha alineada.

**P5.3** ¿Longitud máxima de palabra? (Define el tamaño de la zona de colocación.) **R:** 6 letras

---

## 6\. Interfaces entre módulos (el contrato)

**P6.1 ⭐** Gestos → Robot: propuesta: Equipo 4 publica la palabra completa como texto en un topic (`/palabra`, `std_msgs/String`) cuando el usuario termina. ¿De acuerdo? ¿O preferís letra a letra (el robot empieza a coger mientras se deletrea)? **R:** En principio, letra a letra. Vemos según avance el proyecto si podemos hacerlo de la otra manera también.

**P6.2 ⭐** Fichas → Robot: Equipo 9 dice "encontrar la ficha con la letra indicada" (pregunta-respuesta). Alternativa: publica **todas** las fichas detectadas (letra, x, y, ángulo, confianza) y el robot elige. Propuesta: la segunda, es más simple de probar y depurar. ¿De acuerdo? **R:** La segunda opción, publicando todas las fichas disponibles

**P6.3** ¿En qué sistema de referencia publica Equipo 9 las posiciones? Propuesta: en milímetros/metros respecto al marco `world` del robot, para que robótica no tenga que hacer conversiones. Esto requiere calibrar la posición de los ArUco respecto al robot: ¿quién lo hace y cómo (medir a mano, tocar los marcadores con la herramienta del robot…)? **R:** En milimetros/metros con ayuda de ArUco. Lo hacen las personas encargadas de la parte de Visión del Equipo 9 (a confirmar quienes).

**P6.4** ¿Cómo indica el usuario que ha terminado la palabra (y que empieza una nueva)? ¿Gesto especial, quitar la mano X segundos, tecla? **R:** cuando la persona baja los brazos

**P6.5** ¿Hay feedback al usuario (pantalla mostrando la letra reconocida y la palabra en curso)? ¿Puede corregir/borrar una letra? **R:** No se ha pensado, pero se puede llegar a integrar algo simple.

**P6.6** ¿El método (clásico vs DL) se elige al lanzar (parámetro) y ambos publican exactamente el mismo mensaje? (Ambos PDF lo dan a entender: confirmadlo.) **R:** Sí, así es.

**P6.7** ¿Los equipos de visión entregan nodos ROS 2 ya hechos, o código Python puro que robótica envuelve en un nodo? (Propuesta: nodos ROS dentro de su paquete, con el código de visión en módulos que no dependan de ROS para poder probarlos con fotos.) **R:** Acorde a la propuesta

---

## 7\. Comportamiento del robot y errores

**P7.1** ¿Una vez llega la palabra, el robot hace una sola foto de la bandeja al principio o re-detecta antes de cada ficha? (Re-detectar es más robusto si alguna ficha se mueve.) **R:** Mejor redetectar, por si acaso hay alguna ficha que se mueve al coger otra.

**P7.2** ¿Qué hacer si falta una letra en la bandeja? (Saltarla, abortar, avisar.) **R:** Avisar y continuar. Volver a intentar al de X tiempo (?). Volver a home (?)

**P7.3** ¿Qué hacer si el agarre falla (la pinza se cierra del todo \= no ha cogido nada)? ¿Reintentar N veces? **R:** Reintentar N veces.

**P7.4** ¿Hay que tener en cuenta seguridad/zonas prohibidas (la persona que deletrea está cerca del robot)? ¿Velocidad máxima? (El ejemplo usa 10 %.) **R:** Sí, hay que tener en cuenta la seguridad. La velocidad máxima podría ser un parámetro al iniciar el programa, y que en movimientos más críticos (ej. recoger/dejar ficha) haya otra velocidad máxima establecida.

**P7.5** ¿Queremos modelar en MoveIt la mesa, la bandeja y la zona de colocación como objetos de colisión? (Recomendado.) **R:** Si

---

## 8\. Simulación y pruebas

**P8.1** ¿Se exige que todo funcione en simulación antes del robot real? Propuesta: robótica desarrolla 100% en simulación con un "nodo de visión falso" que publica fichas inventadas; el día de laboratorio solo se cambia el nodo falso por el real. **R:** No se exige, pero al no tener acceso al robot siempre, la parte de la simulación es necesaria. Así que hacemos lo que dices.

**P8.2** ¿Tenéis ya fotos/vídeos de la bandeja y de las manos con las cámaras reales? Si no, ¿quién y cuándo se graban? (Sirven a los dos equipos de visión para desarrollar sin el laboratorio.) **R:** Todavía no. Cada grupo de visión se encargará de su parte.

**P8.3** ¿Cómo mediremos el éxito del sistema completo? (Ej.: % de palabras completadas sin intervención sobre N intentos.) **R:** Si, por ejemplo.&nbsp;

---

## 9\. Entorno de desarrollo

**P9.1** ¿Todos tenéis el entorno Docker funcionando (`lab.bat iniciar` \+ `lab.bat sim 203`)? ¿Alguien con Mac ARM o Linux? (El build en ARM es más lento y está sin probar.) **R:** Se ejecuta en WSL con docker engine, en Linux (mira el fork, que hemos preparado para la conexión con el robot). De momento no todos lo tenemos funcionando, buscaremos la manera más fiable y se extenderá a mas ordenadores

**P9.2** El DL necesitará librerías (ultralytics/YOLO, PyTorch, MediaPipe…) que no están en la imagen. ¿Modificamos el Dockerfile del proyecto o se instalan con `pip` en el contenedor? (Lo instalado fuera de `ros_workspace` se pierde si se borra el contenedor.) **R:** Creo que mejor si los instalamos con pip, porque todavía no sabemos qué librerías queremos usar.

**P9.3** ¿Equipo 4 usará MediaPipe (detección de landmarks de la mano) en alguna de las dos versiones? (Cambia mucho la dificultad y las dependencias.) **R:** Sí, se utiliza MediaPipe para la detección a través de DL.

---

## 10\. Lo que necesito además de estas respuestas

- Las **bases/enunciado oficial** del proyecto de Robótica (y de Visión si hay requisitos de entrega en código), si existen como documento. **R:** No hay. El proyecto es libre y no hay enunciado.&nbsp;  
- Los **nombres de las 8 personas** y quién está en qué equipo (ya los tengo de los PDF de visión: Equipo 4 \= Ainhoa, Keltse, Xabier, Ane; Equipo 9 \= Mikel, Garai, Xabat, Markel). Confirmad. **R:** Si, son esos  
- Si hay una foto o croquis del **montaje físico** (mesa, robot, bandeja, cámaras), me ayuda mucho. **R:** De momento no tenemos. Imagina una mesa rectangular (con las medidas especificadas anteriormente) que tiene el brazo robótico en el ala norte, en la mitad. Las cámaras están ubicadas según lo definido anteriormente también. La bandeja estará en una esquina, tal vez una de las de arriba (o mas cerca del robot) y la zona de colocación será el resto de la mesa.