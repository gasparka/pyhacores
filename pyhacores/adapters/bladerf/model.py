from pyha.common.hwsim import HW
from pyha.common.sfix import scalb, ComplexSfix, Sfix


class Source(HW):
    """ Convert BladeRF style I/Q into Pyha Complex type
    (4 downto -11) -> (0 downto -17)"""

    def __init__(self):
        self.out = ComplexSfix(0, 0, -17)

        self._delay = 1

    def main(self, real, imag):
        self.next.out.real = scalb(real, 4)
        self.next.out.imag = scalb(imag, 4)
        return self.out

    def model_main(self, i, q):
        return i * (2 ** 4) + q * (2 ** 4) * 1j


class Sink(HW):
    """ Default ComplexSfix to BladeRF style """
    def __init__(self):
        self.out_real = Sfix(0, 0, -15)
        self.out_imag = Sfix(0, 0, -15)

        self._delay = 1

    def main(self, c):
        self.next.out_real = scalb(c.real, -4)
        self.next.out_imag = scalb(c.imag, -4)
        return self.out_real, self.out_imag

    def model_main(self, c):
        real = [x.real * (2 ** -4) for x in c]
        imag = [x.imag * (2 ** -4) for x in c]
        return real, imag
