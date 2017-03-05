from pyha.common.hwsim import HW
from pyha.common.sfix import Sfix
from scipy import signal

from pyhacores.filter.fir.model import normalize_taps


class FIR(HW):
    """ FIR filter, taps will be normalized to sum 1 """
    def __init__(self, taps, type):
        self.taps = normalize_taps(taps)

        # registers
        self.mul = [Sfix(0.0, size_res=type.mult)] * len(self.taps)
        self.acc = [Sfix(0.0, left=1, right=self.mul[0].right)] * len(self.taps)
        self.out = type

        # constants
        self.taps_fix_reversed = Const([Sfix(x, 0, -17) for x in reversed(self.taps)])
        self._delay = 3

    def main(self, x):
        """
        Transposed form FIR implementation, uses full precision
        """
        for i in range(len(self.taps)):
            self.next.mul[i] = x * self.taps[i]
            if i == 0:
                self.next.acc[0] = self.mul[i]
            else:
                self.next.acc[i] = self.acc[i - 1] + self.mul[i]

        self.next.out = self.acc[-1]
        return self.out

    def model_main(self, x):
        return signal.lfilter(self.taps, [1.0], x)
