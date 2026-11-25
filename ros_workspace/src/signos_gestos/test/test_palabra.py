"""Tests de signos_gestos/comun/palabra.py · Responsable: Keltse.

Cada test simula la letra que devuelve reconocer() fotograma a fotograma.

Se ejecuta desde la carpeta signos_gestos/ con:  python -m pytest test/test_palabra.py
"""
from signos_gestos.comun.palabra import Palabra


def pasar(palabra, secuencia):
    """secuencia: lista de (letra, n_fotogramas). Devuelve las letras aceptadas en orden."""
    aceptadas = []
    for letra, n in secuencia:
        for _ in range(n):
            nueva = palabra.actualizar(letra)
            if nueva:
                aceptadas.append(nueva)
    return aceptadas


def test_letra_estable_se_acepta_una_sola_vez():
    p = Palabra(estable=10)
    assert pasar(p, [('B', 9)]) == []
    assert pasar(p, [('B', 1)]) == ['B']
    assert pasar(p, [('B', 50)]) == []          # mantener el gesto no la repite
    assert p.texto == 'B'


def test_parpadeo_reinicia_la_cuenta():
    p = Palabra(estable=10)
    pasar(p, [('A', 6), ('?', 1), ('A', 6)])
    assert p.texto == ''


def test_cambio_de_gesto_sin_pausa_forma_lobo():
    p = Palabra(estable=10, hueco=8)
    pasar(p, [('L', 12), ('O', 12), ('B', 12), ('O', 12)])
    assert p.texto == 'LOBO'


def test_letra_doble_necesita_soltar_el_gesto():
    p = Palabra(estable=10, hueco=8)
    pasar(p, [('L', 12), ('?', 3), ('L', 12)])    # pausa demasiado corta
    assert p.texto == 'L'
    pasar(p, [('?', 8), ('L', 12)])                # pausa suficiente
    assert p.texto == 'LL'


def test_none_cuenta_como_sin_letra():
    p = Palabra(estable=3, hueco=2)
    pasar(p, [('V', 3), (None, 2), ('V', 3)])
    assert p.texto == 'VV'


def test_valores_por_defecto():
    p = Palabra()
    assert (p.estable, p.hueco) == (10, 8)
