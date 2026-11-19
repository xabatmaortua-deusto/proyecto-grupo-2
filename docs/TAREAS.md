# Reparto de tareas

Referencia del diseño: `CLAUDE.md`. Las tareas se agrupan por bloque y se indica cuántas personas se recomiendan para cada una.
El grupo decide quién hace cada cosa. Recordad que **todos** tienen que tocar visión y robótica: los bloques de robótica
se cubren con gente de los dos equipos de visión, que puede ir rotando.

```
Fase 0 (bloqueante, corta)  ──►  Fase 1: bloques A, B, C, D en paralelo  ──►  Fase 2: integración en el laboratorio  ──►  Fase 3: extras + demo
```

---

## Fase 0 — Base común (2 personas, una de cada equipo de visión)

Hay que terminarla antes que todo lo demás, porque desbloquea al resto.

1. **Paquetes vacíos que compilan**: `signos_interfaces` (con `PalabraEnCurso.msg`, `Ficha.msg` y `FichaArray.msg` tal como están en CLAUDE.md),
   `signos_gestos`, `signos_fichas`, `signos_robot` y `signos_bringup`. `colcon build` sin errores.
2. **Nodos falsos** en `signos_bringup`:
   - Gestos falso: lee una palabra por teclado y publica `/signos/palabra` letra a letra (con un pequeño retardo).
   - Fichas falso: lee las fichas de un yaml y publica `/signos/fichas` de forma continua.
3. **`config/mesa.yaml` inicial** con las medidas aproximadas (mesa, bandeja, huecos de la palabra, alturas, velocidades).
4. **Launch de simulación** que levanta `sim_205` y los dos nodos falsos.
5. **`requirements.txt`** vacío o con lo mínimo, y que todo el grupo tenga el entorno `local`/`local_gpu` funcionando.

**Hecho cuando:** en `main`, `ros2 topic echo` muestra los dos topics con los nodos falsos en simulación.

---

## Fase 1 — Bloques en paralelo

### Bloque A — Gestos (Equipo 4, 4 personas)

| Tarea | Personas | Depende de |
|---|---|---|
| A1. Grabar un dataset propio de manos con la cámara real (las 6 letras, varias personas, varias luces) y dejarlo en Drive | todo el equipo, una sesión | — |
| A2. Método **clásico**: segmentar la mano, sacar el contorno y aplicar reglas geométricas para A, B, L, O, V, W (y "ninguna" si no está claro) | 2 | A1 |
| A3. Método **DL**: landmarks con MediaPipe y clasificador entrenado (ASL público + dataset propio) | 2 | A1 |
| A4. **Constructor de palabra** (común a los dos métodos): estabilidad temporal, letras repetidas, fin al bajar los brazos | 1 (de A2 o A3) | — (se prueba con secuencias inventadas) |
| A5. **Nodo ROS** `signos_gestos` con el parámetro `metodo`, publica `PalabraEnCurso`, y ventana de feedback de OpenCV (letra actual y palabra) | 1 | A4, Fase 0 |
| A6. Evaluación comparativa clásico vs DL (precisión, robustez, palabras) para la memoria de Visión | todo el equipo | A2, A3 |

### Bloque B — Fichas (Equipo 9, 4 personas)

| Tarea | Personas | Depende de |
|---|---|---|
| B1. **Preparar el material**: añadir la letra y la raya a las fichas (al menos 2 por letra), poner los ArUco en la bandeja, medir los bordes | 1–2 | — |
| B2. **ArUco y medida**: detectar los marcadores, enderezar la imagen y pasar de píxel a mm en el plano de la bandeja (común a los dos métodos) | 1 | B1 |
| B3. **Calibración tablero → `world`**: medir dónde están los ArUco respecto a la base del robot y guardarlo en `mesa.yaml` (comprobarlo llevando la pinza a una esquina) | 1 (con alguien de robótica) | B2, acceso al robot |
| B4. Fotos de la bandeja con la cámara cenital real (fichas en distintas posiciones y giros, distintas luces) para Drive | todo el equipo, una sesión | B1 |
| B5. Método **clásico**: umbral, centro, ángulo y raya → `theta`, y letra por agujeros y cortes | 2 | B2, B4 |
| B6. Método **DL**: etiquetar el dataset y entrenar YOLO (cajas orientadas, o detector + clasificador del recorte) | 2 | B4 |
| B7. **Nodo ROS** `signos_fichas` con el parámetro `metodo`, publica `FichaArray` en `world` | 1 | B2, Fase 0 |
| B8. Evaluación comparativa clásico vs DL (precisión de la letra, error de posición en mm y de ángulo) para la memoria de Visión | todo el equipo | B5, B6 |

### Bloque C — Robótica en simulación (3 personas, mezcla de los dos equipos)

| Tarea | Personas | Depende de |
|---|---|---|
| C1. **Escena de colisión**: mesa, bandeja con bordes y zona de la palabra en MoveIt, a partir de `mesa.yaml` | 1 | Fase 0 |
| C2. **Coger y dejar una ficha**: aproximación desde arriba, bajada lenta, giro según `theta`, apertura y cierre de la pinza y comprobación del agarre | 1–2 | Fase 0 |
| C3. **Orquestador**: máquina de estados (esperar letra → home → detección nueva → elegir ficha → coger → dejar en hueco n), reintentos y letra que falta | 1 | C2 (puede empezar con C2 sin terminar) |
| C4. Tests de la lógica sin robot: elegir ficha, calcular hueco y orientación, transiciones de la máquina de estados | quien haga C3 | C3 |

### Bloque D — Robot real (2 personas, las que más vayan al laboratorio)

| Tarea | Personas | Depende de |
|---|---|---|
| D1. Conexión estable del portátil o PC con el robot 205 y **verificar la pinza RG2 real** (abrir, cerrar, leer el agarre) | 1–2 | — |
| D2. Comprobar que las cámaras USB funcionan dentro del contenedor (`usb_cam`, resolución, que cada dispositivo se identifique bien) | 1 | — |
| D3. Coger y dejar a mano una ficha con el robot real: altura real de agarre, fuerza y anchura de la pinza, y ajustar `mesa.yaml` | 1–2 | C2, D1 |

---

## Fase 2 — Integración en el laboratorio (todos, por turnos)

1. Launch del sistema completo (`signos_bringup`): robot 205 real, cámaras, `signos_gestos` y `signos_fichas` con el parámetro `metodo`, y orquestador.
2. Primero los nodos reales de uno en uno (fichas real + gestos falso, después gestos real + fichas falso), y luego todo junto.
3. Medir el éxito: % de palabras completadas sin intervención sobre N intentos, con cada combinación de métodos. Apuntarlo para la memoria.

## Fase 3 — Extras y entrega

- Extras según el tiempo: más letras (empezando por DL), modo "palabra completa", varias palabras seguidas, corregir letras.
- README del proyecto (cómo lanzar la demo), memoria y ensayo de la defensa con la demo en vivo.
