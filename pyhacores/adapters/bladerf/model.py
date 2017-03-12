from pyha.common.hwsim import HW
from pyha.common.sfix import resize, scalb, ComplexSfix, Sfix


class ComplexSource(HW):
    """ Convert BladeRF style I/Q into Pyha Complex type
    (4 downto -11) -> (0 downto -17)"""

    def __init__(self):
        self.out = ComplexSfix(0, 0, -17)

        self._delay = 1

    def main(self, i, q):
        self.next.out.real = scalb(i, 4)
        self.next.out.imag = scalb(q, 4)
        return self.out

    def model_main(self, i, q):
        return i * (2 ** 4) + q * (2 ** 4) * 1j


class FloatSink(HW):
    """ Pyha 18 bit signal to BladeRF style """
    def __init__(self):
        self.out = Sfix(0, 0, -15)

        self._delay = 1

    def main(self, x):
        self.next.out = scalb(x, -4)
        return self.out

    def model_main(self, x):
        return x * (2 ** -4)
