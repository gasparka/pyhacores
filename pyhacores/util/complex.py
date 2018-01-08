import numpy as np
import pytest
from pyha import Hardware, Complex, simulate, sims_close


class ComplexConjugate(Hardware):
    def __init__(self):
        self.y = Complex(0, 0, -17,
                             overflow_style='saturate')  # protect against overflow when negating -1
        self.DELAY = 1

    def main(self, x):
        self.y.real = x.real
        self.y.imag = -x.imag
        return self.y

    def model_main(self, x):
        return np.conjugate(x)


class ComplexMultiply(Hardware):
    """ (x + yj)(u + vj) = (xu - yv) + (xv + yu)j """

    def __init__(self):
        self.y = Complex(0 + 0j, 0, -17)
        self.DELAY = 1

    def main(self, a, b):
        self.y.real = (a.real * b.real) - (a.imag * b.imag)
        self.y.imag = (a.real * b.imag) + (a.imag * b.real)
        return self.y

    def model_main(self, a, b):
        return np.array(a) * np.array(b)


def test_conjugate():
    inp = [0.5 + 0.1j, -0.234 + 0.1j, 0.5 - 0.1j, 1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
    expect = [0.5 - 0.1j, -0.234 - 0.1j, 0.5 + 0.1j, 1 - 1j, 1 + 1j, -1 - 1j, -1 + 1j]

    dut = ComplexConjugate()

    sims = simulate(dut, inp)
    assert sims_close(sims, expect)


def test_conjugate_low_magnitude():
    inp = (np.random.rand(1024) + np.random.rand(1024) * 1j) * 0.01

    dut = ComplexConjugate()
    sims = simulate(dut, inp)
    assert sims_close(sims)


def test_multiply():
    a = [0.123 + .492j, 0.444 - 0.001j, -0.5 + 0.432j, -0.123 - 0.334j]
    b = [0.425 + .445j, -0.234 - 0.1j, -0.05 + 0.32j, 0.453 + 0.5j]

    y = [-0.166665 + 0.263835j, -0.103996 - 0.044166j, -0.113240 - 0.1816j,
         0.111281 - 0.212802j]

    dut = ComplexMultiply()
    sims = simulate(dut, a, b)
    assert sims_close(sims, y)


def test_multiply_low_magnitude():
    """ Test that Pyha and RTL stuff matches up """
    a = (np.random.rand(1024) + np.random.rand(1024) * 1j) * 0.01
    b = (np.random.rand(1024) + np.random.rand(1024) * 1j) * 0.01

    dut = ComplexMultiply()
    sims = simulate(dut, a, b)
    assert sims_close(sims)


def test_multiply_harmonic():
    # 2hz signal
    t = np.linspace(0, 2, 1024)
    a = np.exp(1j * 2 * np.pi * 2 * t) * 0.9

    # 4hz signal
    b = np.exp(1j * 2 * np.pi * 4 * t) * 0.9

    expect = a * b

    dut = ComplexMultiply()
    sims = simulate(dut, a, b)
    assert sims_close(sims, expect)


def test_multiply_harmonic_overflow():
    pytest.xfail('Fails because ComplexMultiply has no protection of overflow!')
    # 2hz signal
    t = np.linspace(0, 2, 1024)
    a = np.exp(1j * 2 * np.pi * 2 * t)

    # 4hz signal
    b = np.exp(1j * 2 * np.pi * 4 * t)

    expect = a * b

    dut = ComplexMultiply()
    sims = simulate(dut, a, b)
    assert sims_close(sims, expect)
