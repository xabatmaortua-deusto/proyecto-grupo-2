"""Genera un PDF A4 con 4 marcadores ArUco de tamaño conocido para calibrar
perspectiva (homografía) y relación píxel/metro de una cámara.

Los marcadores se dibujan como rectángulos vectoriales -> tamaño exacto al imprimir
(imprimir a "Tamaño real" / 100 %, sin "Ajustar a página").
"""
import json
import cv2
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------------- Parámetros ----------------
DICT_NAME = "DICT_4X4_50"
IDS = [0, 1, 2, 3]            # sup-izq, sup-der, inf-izq, inf-der
MARKER_MM = 80.0              # lado del marcador (borde negro incluido)
PAGE_W, PAGE_H = 210.0, 297.0  # A4 en mm
# Centros de los marcadores en mm (origen = esquina sup-izq de la hoja, y hacia abajo)
CENTERS = {
    0: (55.0, 75.0),
    1: (155.0, 75.0),
    2: (55.0, 225.0),
    3: (155.0, 225.0),
}
OUT_PDF = "arucos_calibracion_A4.pdf"
OUT_JSON = "arucos_calibracion_A4.json"
# --------------------------------------------

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(getattr(aruco, DICT_NAME))
n_bits = dictionary.markerSize          # 4
n_mod = n_bits + 2                      # + borde negro de 1 módulo a cada lado
mod_mm = MARKER_MM / n_mod

fig = plt.figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, PAGE_W)
ax.set_ylim(PAGE_H, 0)  # y hacia abajo, en mm
ax.set_aspect("equal")
ax.axis("off")

def draw_marker(mid, cx, cy):
    img = aruco.generateImageMarker(dictionary, mid, n_mod)  # 1 px = 1 módulo
    x0, y0 = cx - MARKER_MM / 2, cy - MARKER_MM / 2
    for r in range(n_mod):
        for c in range(n_mod):
            if img[r, c] < 128:
                ax.add_patch(Rectangle((x0 + c * mod_mm, y0 + r * mod_mm),
                                       mod_mm, mod_mm, facecolor="black",
                                       edgecolor="black", linewidth=0.05))
    # marcas de esquina fuera del marcador (zona blanca) y del centro
    t = 3.0
    for sx in (-1, 1):
        for sy in (-1, 1):
            px, py = cx + sx * MARKER_MM / 2, cy + sy * MARKER_MM / 2
            ax.plot([px + sx * 2, px + sx * (2 + t)], [py, py], color="0.5", lw=0.4)
            ax.plot([px, px], [py + sy * 2, py + sy * (2 + t)], color="0.5", lw=0.4)
    ax.text(cx, cy + MARKER_MM / 2 + 6, f"ID {mid}  ·  centro ({cx:.0f}, {cy:.0f}) mm",
            ha="center", va="center", fontsize=7, color="0.35")

for mid in IDS:
    draw_marker(mid, *CENTERS[mid])

# Título
ax.text(PAGE_W / 2, 12, f"Calibración ArUco · {DICT_NAME} · lado {MARKER_MM:.0f} mm",
        ha="center", va="center", fontsize=11, weight="bold")
ax.text(PAGE_W / 2, 19, "Imprimir a TAMAÑO REAL (100 %) — verificar con la regla de 100 mm",
        ha="center", va="center", fontsize=8, color="0.3")

# Regla de verificación de 100 mm (centro de la hoja)
ry, rx0 = 150.0, 55.0
ax.plot([rx0, rx0 + 100], [ry, ry], color="black", lw=0.6)
for i in range(0, 101, 10):
    h = 4 if i % 50 == 0 else 2.5
    ax.plot([rx0 + i, rx0 + i], [ry - h, ry], color="black", lw=0.5)
    if i % 50 == 0:
        ax.text(rx0 + i, ry - 6, f"{i}", ha="center", va="center", fontsize=6)
for i in range(0, 101):
    ax.plot([rx0 + i, rx0 + i], [ry - 1.2, ry], color="black", lw=0.25)
ax.text(rx0 + 50, ry + 5, "100 mm", ha="center", va="center", fontsize=7)

dx = CENTERS[1][0] - CENTERS[0][0]
dy = CENTERS[2][1] - CENTERS[0][1]
ax.text(PAGE_W / 2, ry + 12,
        f"Distancia entre centros: horizontal {dx:.0f} mm · vertical {dy:.0f} mm",
        ha="center", va="center", fontsize=7, color="0.3")

ax.text(PAGE_W / 2, PAGE_H - 10,
        "Origen: esquina superior izquierda de la hoja, X → derecha, Y ↓ abajo. "
        "Esquinas de cada marcador en orden ArUco: TL, TR, BR, BL.",
        ha="center", va="center", fontsize=6, color="0.4")

fig.savefig(OUT_PDF)

# Coordenadas reales (m) de las 4 esquinas de cada marcador, en el orden que devuelve OpenCV
h = MARKER_MM / 2
data = {"dictionary": DICT_NAME, "marker_size_m": MARKER_MM / 1000,
        "origin": "esquina sup-izq de la hoja, x derecha, y abajo",
        "markers": {}}
for mid in IDS:
    cx, cy = CENTERS[mid]
    corners = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
    data["markers"][str(mid)] = {
        "center_m": [cx / 1000, cy / 1000],
        "corners_m": [[x / 1000, y / 1000] for x, y in corners],
    }
with open(OUT_JSON, "w") as f:
    json.dump(data, f, indent=2)
print("OK:", OUT_PDF, OUT_JSON)
