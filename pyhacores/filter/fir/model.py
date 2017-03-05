from pyha.common.const import Const
from pyha.common.hwsim import HW
from pyha.common.sfix import Sfix, resize, left_index, right_index, fixed_truncate, fixed_wrap
from pyha.simulation.simulation_interface import assert_sim_match, SIM_HW_MODEL, SIM_MODEL, SIM_RTL, SIM_GATE
from scipy import signal
import numpy as np


def normalize_taps(taps):
    taps = np.array(taps)
    cs = sum(abs(taps))
    for (i, x) in enumerate(taps):
        taps[i] = x / cs

    return taps.tolist()


class FIR(HW):
    """ FIR filter, taps will be normalized to sum 1 """
    def __init__(self, taps):
        self.taps = normalize_taps(taps)

        # registers
        self.acc = [Sfix()] * len(self.taps)
        self.mul = [Sfix()] * len(self.taps)
        self.out = Sfix()

        # constants
        self.add_growth = Const(int(np.log2(len(taps))))
        self.taps_fix_reversed = Const([Sfix(x, 0, -17) for x in reversed(self.taps)])
        self._delay = 3

    def main(self, x):
        """
        Transposed form FIR implementation, uses full precision
        """
        for i in range(len(self.taps_fix_reversed)):
            self.next.mul[i] = x * self.taps_fix_reversed[i]
            if i == 0:
                self.next.acc[0] = resize(self.mul[i], 1, right_index(self.mul[i]), round_style=fixed_truncate,
                                          overflow_style=fixed_wrap)
            else:
                self.next.acc[i] = resize(self.acc[i - 1] + self.mul[i], 1, right_index(self.mul[i]),
                                          round_style=fixed_truncate, overflow_style=fixed_wrap)

        self.next.out = resize(self.acc[-1], size_res=x, round_style=fixed_truncate)
        return self.out

    def model_main(self, x):
        return signal.lfilter(self.taps, [1.0], x)


def test_symmetric():
    taps = signal.remez(128, [0, 0.1, 0.2, 0.5], [1, 0])
    dut = FIR(taps)
    inp = np.random.uniform(-1, 1, 128)

    assert_sim_match(dut, [Sfix(left=0, right=-17)], None, inp,
                     simulations=[SIM_MODEL, SIM_HW_MODEL, SIM_RTL, SIM_GATE],
                     dir_path='/home/gaspar/git/pyhacores/conversion',
                     rtol=1e-4, atol=1e-4)


def test_non_symmetric():
    taps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    dut = FIR(taps)
    inp = np.random.uniform(-1, 1, 128)

    assert_sim_match(dut, [Sfix(left=0, right=-17)], None, inp,
                     simulations=[SIM_MODEL, SIM_HW_MODEL, SIM_RTL, SIM_GATE],
                     dir_path='/home/gaspar/git/pyhacores/conversion',
                     rtol=1e-5,
                     atol=1e-5)
