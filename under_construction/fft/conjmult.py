import numpy as np
import pytest
from pyha import Hardware, simulate, sims_close, Complex, resize, Sfix


def conjugate(x):
    imag = resize(-x.imag, x.imag.left, x.imag.right)
    return Complex(x.real, imag)

class ConjMult(Hardware):
    def __init__(self):
        self.out = Sfix(0, 0, -17)
        self.DELAY = 1

    def main(self, complex_in):
        self.out = (conjugate(complex_in) * complex_in).real
        return self.out


def test_abs():
    M = 128
    dut = ConjMult()
    inp = np.random.uniform(-1, 1, M) + np.random.uniform(-1, 1, M) * 1j
    inp *= 0.5

    sims = simulate(dut, inp, simulations=['MODEL_PYHA', 'PYHA'])
    assert sims_close(sims)