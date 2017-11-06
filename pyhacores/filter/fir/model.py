from pyha.common.hwsim import Hardware
from pyha.common.sfix import Sfix, fixed_truncate, fixed_wrap

from scipy import signal
import numpy as np


class FIR(Hardware):
    """ FIR filter, taps will be normalized to sum 1 """
    def __init__(self, taps):
        self.taps = np.array(taps).tolist()

        # registers
        self.acc = [Sfix(left=0, round_style=fixed_truncate, overflow_style=fixed_wrap)] * (len(self.taps) + 1)

        # constants
        self.TAPS_REVERSED = [Sfix(x, 0, -17) for x in reversed(self.taps)]
        self.DELAY = 1

    def main(self, x):
        for i in range(1, len(self.acc)):
            self.acc[i] = self.acc[i - 1] + x * self.TAPS_REVERSED[i - 1]

        return self.acc[-1]

    def model_main(self, x):
        return signal.lfilter(self.taps, [1.0], x)