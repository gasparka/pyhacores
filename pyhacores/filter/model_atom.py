from pyha.common.const import Const
from pyha.common.hwsim import HW
from pyha.common.sfix import Sfix, fixed_truncate, fixed_wrap

from scipy import signal
import numpy as np


def rescale_taps(taps):
    """
    Rescale taps in that way that their sum equals 1
    """
    taps = np.array(taps)
    cs = sum(taps)
    # fixme: not sure here, abs seems right as it avoids overflows in core,
    # then again it reduces the fir gain
    # cs = sum(abs(taps))
    for (i, x) in enumerate(taps):
        taps[i] = x / cs

    return taps.tolist()


class MAC(HW):
    def __init__(self, coef):
        self.coef = Sfix(coef, 0, -17)
        self.acc = Sfix(left=1, round_style=fixed_truncate, overflow_style=fixed_wrap)
        self.mul = Sfix(round_style=fixed_truncate, overflow_style=fixed_wrap)

        self._delay = 2

    def main(self, a, sum_in):
        self.next.mul = self.coef * a
        self.next.acc = self.mul + sum_in
        return self.acc

    def model_main(self, a):
        import numpy as np

        muls = np.array(a) * self.coef
        return np.cumsum(muls)


class FIR_atom(HW):
    def __init__(self, taps):
        self.taps = rescale_taps(taps)

        # registers
        self.mac = [MAC(x) for x in reversed(self.taps)]
        self.out = Sfix(0, 0, -17, round_style=fixed_truncate)

        # constants
        self._delay = 3

    def main(self, x):
        sum_in = Sfix(0.0, 1, -34)
        for mav in self.mac:
            sum_in = mav.main(x, sum_in)

        self.next.out = sum_in
        return self.out

    def model_main(self, x):
        return signal.lfilter(self.taps, [1.0], x)
