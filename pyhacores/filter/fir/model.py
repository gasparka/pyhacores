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


class FIR(HW):
    """ FIR filter, taps will be normalized to sum 1 """
    def __init__(self, taps):
        self.taps = rescale_taps(taps)

        # registers
        self.acc = [Sfix(left=1, round_style=fixed_truncate, overflow_style=fixed_wrap)] * len(self.taps)
        self.mul = [Sfix(round_style=fixed_truncate, overflow_style=fixed_wrap)] * len(self.taps)
        self.out = Sfix(0, 0, -17, round_style=fixed_truncate)

        # constants
        self.taps_fix_reversed = Const([Sfix(x, 0, -17) for x in reversed(self.taps)])
        # self.taps_fix_reversed = [Sfix(x, 0, -17) for x in reversed(self.taps)]
        self._delay = 3

    def main(self, x):
        """
        Transposed form FIR implementation, uses full precision
        """
        for i in range(len(self.taps_fix_reversed)):
            self.next.mul[i] = x * self.taps_fix_reversed[i]
            if i == 0:
                self.next.acc[0] = self.mul[i]
            else:
                b = self.acc[i - 1] + self.mul[i]
                self.next.acc[i] = self.acc[i - 1] + self.mul[i]

        self.next.out = self.acc[-1]
        return self.out

    def model_main(self, x):
        return signal.lfilter(self.taps, [1.0], x)