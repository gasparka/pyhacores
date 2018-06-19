import numpy as np
import pytest
from pyha import Hardware, simulate, sims_close, Complex, resize, Sfix, right_index, left_index
from under_construction.fft.packager import DataWithIndex, unpackage, package


class ConjMult(Hardware):
    def __init__(self):
        self.out = DataWithIndex(Sfix(0.0, 0, -35), 0)
        self.DELAY = 1

    def conjugate(self, x):
        imag = resize(-x.imag, left_index(x.imag), right_index(x.imag))
        return Complex(x.real, imag)

    def main(self, inp):
        conj = self.conjugate(inp.data)
        self.out.data = (conj.real * inp.data.real) - (conj.imag * inp.data.imag)
        self.out.index = inp.index
        # self.out = (self.conjugate(complex_in) * complex_in).real
        return self.out

    def model_main(self, data):
        return (np.conjugate(data) * data).real


def test_abs():
    M = 128
    dut = ConjMult()
    inp = np.random.uniform(-1, 1, size=(2, M)) + np.random.uniform(-1, 1, size=(2, M)) * 1j
    inp *= 0.5 * 0.0001

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA',
                                           # 'RTL'
                                           ], output_callback=unpackage, input_callback=package)
    assert sims_close(sims)
