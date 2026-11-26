# Plan de captura del dataset

Equipo 4 (gestos): Ainhoa, Ane, Keltse y Xabier.

Qué tenemos que grabar, cómo y dónde se guarda, para que **una sola tanda de grabaciones sirva a las dos vías**
(clásica y deep learning) y a la evaluación común (`comun/evaluar.py`).

> Es una propuesta de Keltse a partir de las tareas 0.2, T11 y T12 de [REPARTO_TAREAS.md](../REPARTO_TAREAS.md),
> que coordina Ainhoa. Antes de grabar, la revisamos juntos y se cambia aquí lo que haga falta.

---

## 1. Resumen

| Qué | Por persona | Total (4 personas) |
|---|---|---|
| Grabaciones de **letras** (A, B, L, O, V, W y `nada`), 5 s cada una | 7 clases × 3 grabaciones × 2 luces = **42** | 168 |
| Vídeos de **palabras** | 5 palabras × 2 luces = **10** | 40 |
| Tiempo | 2 sesiones de ~25 min (una por luz) | 8 sesiones |

Con esto salen, por letra, unas 24 grabaciones, unos **1 200 recortes** para la red (1 de cada 3 fotogramas) y unas
1 600 ventanas para el Random Forest. Los vídeos a 720 px ocupan en total unos 500 MB.

---

## 2. Antes de la primera sesión (una vez, todos juntos)

1. **Acordar cómo se hace cada letra en LSE** (A, B, L, O, V, W). Se hace una **foto de referencia** de cada una y se guarda
   en `docs/figuras/letras/`, para que los cuatro la hagamos igual. Nuestra O es la de la LSE (tres dedos arriba y círculo
   índice-pulgar), no la de la ASL.
2. **Decidir qué persona es el conjunto de test** (tarea D4, de Xabier). Propuesta: **una persona entera** (sus dos luces),
   y las otras tres para entrenar y validar. Se decide **antes** de grabar y no se cambia: sus vídeos no se miran para
   ajustar umbrales ni para elegir el modelo.
3. **La cámara.** Lo ideal es grabar con la **cámara ELP de la demo**. Si todavía no la tenemos, la webcam del portátil,
   pero al menos una sesión por persona tiene que ser con la ELP **en el laboratorio**, que es donde se hará la demo.

---

## 3. Condiciones de grabación

| | Cómo | Por qué |
|---|---|---|
| Mano | **Derecha**, siempre | Los vídeos de prueba son todos con la derecha. La vía clásica distingue letras por la orientación |
| Ropa | **Manga larga oscura**, sin anillos, pulsera ni reloj | La segmentación por color de piel confunde la ropa clara con la mano |
| Fondo | Liso y claro (pared blanca), sin objetos de color piel detrás | Igual que en el punto anterior |
| Encuadre | Cámara fija (trípode o apoyada). Mano a la altura del pecho y **siempre dentro del cuadro**, con algo de antebrazo | En IMG_2486 la mano sale del cuadro y la L se repite (LLOBO) |
| Distancia | Como en la demo: unos 50–80 cm de la cámara | — |
| Luz | Dos sesiones: **`dia`** (luz natural de ventana) y **`artificial`** (fluorescentes del aula o del laboratorio) | Robustez por iluminación (T3) |
| En el cuadro | Solo una persona. Puede salir la cara | Los vídeos solo van al Drive del grupo, nunca a git ni a sitios públicos |

---

## 4. Qué grabar en cada sesión

Cada sesión es **una persona con una luz**. Se apunta en el registro (apartado 7).

### 4.1 Letras (con `grabar_letras.py`)

Para cada clase (A, B, L, O, V, W y `nada`) se hacen **3 grabaciones de 5 s**, cada una distinta:

| Grabación | Qué hacer durante los 5 s |
|---|---|
| 1 | La letra de frente, a la distancia normal, moviendo la mano un poco |
| 2 | Más lejos y más cerca de la cámara |
| 3 | Girando la muñeca un poco (±20°) y cambiando la altura |

La clase **`nada`** es la mano que **no** hace ninguna letra: en reposo, bajando, subiendo o cambiando de una letra a
otra. Sirve para que el reconocedor diga `?` en vez de inventarse una letra.

### 4.2 Palabras (cámara de Windows o móvil, y después `importar_palabra.py`)

| Palabra | Para qué |
|---|---|
| LOBO, BOLA, OLA | Las acordadas en REPARTO_TAREAS.md |
| VALLA | **Letra doble** (LL): hay que soltar el gesto entre las dos L |
| BLOWAV | Las **6 letras** y la **longitud máxima** (6). No es una palabra, pero el robot acepta cualquier secuencia |

En cada vídeo:
1. Se empieza con los **brazos bajados** 1–2 s.
2. Se hace cada letra **durante 1 s** más o menos y se pasa a la siguiente de forma natural.
3. En una letra doble, se suelta el gesto (se abre la mano o se baja un poco) medio segundo y se repite.
4. Al acabar, se **bajan los brazos** y se esperan 2 s: así termina la palabra en el sistema (`terminada`).

---

## 5. Paso a paso de una sesión

Todo desde la carpeta `signos_gestos/` (la del `package.xml`):

```powershell
# 1. Letras: ESPACIO graba (cuenta atrás de 1 s), d/a cambia de letra, q sale. 3 grabaciones por letra.
python grabar_letras.py --persona ane --letras A B L O V W nada --segundos 5

# 2. Palabras: se graban con la cámara de Windows (o el móvil) y se meten en el dataset, una palabra cada vez
python herramientas/importar_palabra.py --palabra LOBO --persona ane WIN_20261127_10_00_00_Pro.mp4

# 3. Recortes para la red y revisión a ojo de resultados_recortes/muestra.jpg
python extraer_recortes.py --letras A B L O V W nada
```

4. **Revisar** que las letras están bien hechas (sobre todo la O y la V) y que la mano no sale del cuadro. Lo que esté
   mal se borra (el `.npz` y el `.mp4` con el mismo nombre) y se repite.
5. **Subir al Drive** (apartado 6) y **rellenar el registro** (apartado 7).

`--persona` lleva solo letras y números, en minúsculas y siempre igual (`ainhoa`, `ane`, `keltse`, `xabier`): de ahí se
saca la persona para separar train y test.

---

## 6. Dónde se guarda

En local, dentro de `signos_gestos/`. En el Drive del grupo, en una carpeta `dataset_gestos/` con **las mismas
subcarpetas**, así que basta con copiarlas:

```
dataset_letras/<LETRA>/<persona>_<fecha>.npz    puntos de MediaPipe            → Drive (¿o git? decisión pendiente)
datos/videos/<LETRA>/<persona>_<fecha>.mp4      vídeo limpio de cada letra     → Drive
datos/palabras/<PALABRA>/<persona>_<fecha>.mp4  vídeo de cada palabra          → Drive
datos/recortes/<LETRA>/...jpg                   recortes de la mano (red)      → no se sube: se regenera
```

| Vía | Usa |
|---|---|
| Clásica | `datos/videos/` (fotograma entero) |
| MediaPipe + Random Forest | `dataset_letras/` |
| Deep learning (ResNet) | `datos/recortes/`, que sale de `datos/videos/` + `dataset_letras/` |
| Evaluación de letras (T1) | `datos/videos/` |
| Evaluación de palabras (T2) | `datos/palabras/` |

---

## 7. Registro de sesiones

Una fila por sesión. Con esto se sabe qué luz y qué cámara tiene cada grabación (por fecha) para la robustez (T3).

| Persona | Fecha | Hora (inicio–fin) | Luz | Cámara | Lugar | Letras (de 21) | Palabras (de 5) | En el Drive | Notas |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

---

## 8. Vídeos de prueba que ya teníamos

Son los vídeos con los que se hicieron las primeras pruebas. Ya están en el formato del dataset, en `datos/palabras/`
(importados con `importar_palabra.py`). Como **no sabemos con seguridad quién sale**, la persona es `desconocido`:
así no entran nunca en el conjunto de test. Si alguien se reconoce, se renombran.

Las reglas de MediaPipe se ajustaron con los IMG, así que estos vídeos son **solo de desarrollo**: sirven para probar
que todo funciona, no para dar resultados en la memoria.

| Original | En el dataset | Palabra | Cámara | Reglas de MediaPipe (`evaluar.py`) |
|---|---|---|---|---|
| IMG_2485.MOV | `LOBO/desconocido_20261116-161409.mp4` | LOBO | móvil, vertical | LOBO ✔ |
| IMG_2486.MOV | `LOBO/desconocido_20261116-161415.mp4` | LOBO | móvil, vertical | LLOBO ✘ (la mano sale del cuadro) |
| IMG_2487.MOV | `LOBO/desconocido_20261116-161418.mp4` | LOBO | móvil, vertical | LOBO ✔ |
| WIN_20261123_18_37_31_Pro.mp4 | `LOBO/desconocido_20261123-183753.mp4` | LOBO | webcam | OLBO ✘ (O falsa al entrar la mano) |
| WIN_20261123_18_47_26_Pro.mp4 | `BOLO/desconocido_20261123-184741.mp4` | BOLO | webcam | BOLOWO ✘ |
| WIN_20261123_18_50_33_Pro.mp4 | `LOBO/desconocido_20261123-185046.mp4` | LOBO | webcam | LOBO ✔ |
| WIN_20261123_18_52_03_Pro.mp4 | `BOLA/desconocido_20261123-185220.mp4` | BOLA | webcam | BOLA ✔ |

Resultado: **4 de 7**. Ojo: el vídeo de BOLO, a su resolución original (1080p), daba BOLO; reducido a 720 px da
BOLOWO. La resolución cambia el resultado, así que conviene grabar el dataset con la misma cámara y resolución que la demo.

Los originales están en el portátil de Keltse. Hay que subirlos al Drive, a `dataset_gestos/datos/palabras/`.

---

## 9. Decisiones pendientes

- [ ] Cómo se hace cada letra en LSE: fotos de referencia en `docs/figuras/letras/` (todos).
- [ ] Qué persona es el conjunto de test (Xabier, D4).
- [ ] Si `dataset_letras/*.npz` va en git o en el Drive (todo el grupo, ver `ESTRUCTURA.md`).
- [ ] Si tenemos ya la cámara ELP y cuándo se puede grabar en el laboratorio.
- [ ] Quién sale en los vídeos de prueba del apartado 8.
