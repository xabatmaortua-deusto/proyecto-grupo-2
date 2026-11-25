"""Construcción de la palabra a partir de la letra de cada fotograma · Responsable: Keltse.

La usan las dos vías: solo recibe la letra que devuelve reconocer(frame_bgr) en cada fotograma,
venga del método que venga. Antes estaba dentro de prueba_mediapipe_letras.py.
"""

ESTABLE = 10   # fotogramas seguidos con la misma letra para aceptarla (≈ 0,33 s a 30 fps)
HUECO = 8      # fotogramas sin letra que separan dos letras iguales seguidas (p. ej. «LL»)


class Palabra:
    """Acepta una letra cuando se repite `estable` fotogramas seguidos.

    - No vuelve a añadir la misma letra mientras se mantiene el gesto.
    - Para escribir dos letras iguales seguidas hay que «soltar» el gesto al menos `hueco`
      fotogramas (mano fuera de cuadro o gesto no reconocido) y volver a hacerlo.
    - Entre dos letras distintas no hace falta pausa: el cambio de gesto basta (LOBO).
    """

    def __init__(self, estable=ESTABLE, hueco=HUECO):
        self.estable, self.hueco = estable, hueco
        self.texto = ''
        self.candidata, self.cuenta = None, 0     # letra que se está repitiendo y cuántas veces
        self.ultima = None                        # última letra añadida
        self.sin_letra = 0                        # fotogramas seguidos sin letra válida

    def actualizar(self, letra):
        """Recibe la letra de un fotograma ('?' o None si no hay) y devuelve la letra
        aceptada en este fotograma, o None."""
        if letra in (None, '?'):
            self.sin_letra += 1
            if self.sin_letra >= self.hueco:
                self.ultima = None                # se ha soltado: se permite repetir letra
            self.candidata, self.cuenta = None, 0
            return None
        self.sin_letra = 0
        if letra == self.candidata:
            self.cuenta += 1
        else:
            self.candidata, self.cuenta = letra, 1
        if self.cuenta == self.estable and letra != self.ultima:
            self.texto += letra
            self.ultima = letra
            return letra                          # letra nueva aceptada en este fotograma
        return None
