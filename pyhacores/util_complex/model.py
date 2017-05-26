import numpy as np

from pyha.common.hwsim import HW
from pyha.common.sfix import resize, ComplexSfix, Sfix, fixed_truncate


class Conjugate(HW):
    def __init__(self):
        self.y = ComplexSfix(0, 0, -17, round_style=fixed_truncate)

        self._delay = 1

    def main(self, x):
        self.y.real = x.real
        self.y.imag = -x.imag
        return self.y

    def model_main(self, x):
        return np.conjugate(x)


class ComplexMultiply(HW):
    """ (x + yj)(u + vj) = (xu - yv) + (xv + yu)j """

    def __init__(self):
        self.real_xu = Sfix(0, 0, -17, round_style=fixed_truncate)
        self.real_yv = Sfix(0, 0, -17, round_style=fixed_truncate)

        self.imag_xv = Sfix(0, 0, -17, round_style=fixed_truncate)
        self.imag_yu = Sfix(0, 0, -17, round_style=fixed_truncate)

        self.outreg = ComplexSfix(0+0j, 0, -17, round_style=fixed_truncate)

        self._delay = 2

    def main(self, a, b):
        self.real_xu = a.real * b.real
        self.real_yv = a.imag * b.imag
        self.outreg.real = self.real_xu - self.real_yv

        self.imag_xv = a.real * b.imag
        self.imag_yu = a.imag * b.real
        self.outreg.imag = self.imag_xv + self.imag_yu

        return self.outreg

    def model_main(self, a, b):
        return np.array(a) * np.array(b)
