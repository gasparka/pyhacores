from pyha.common.const import Const
from pyha.common.hwsim import HW
from pyha.common.sfix import Sfix, resize, left_index, right_index, fixed_truncate, fixed_wrap
from pyha.simulation.simulation_interface import assert_sim_match, SIM_HW_MODEL, SIM_MODEL, SIM_RTL, SIM_GATE
from scipy import signal
import numpy as np


class FIR(HW):
    def __init__(self, taps):
        self.taps = taps
        if isinstance(self.taps, np.ndarray):
            self.taps = self.taps.tolist()

        # registers
        self.acc = [Sfix()] * len(self.taps)
        self.mul = [Sfix()] * len(self.taps)

        # constants
        self.taps_fix_reversed = Const([Sfix(x, 0, -17) for x in reversed(self.taps)])
        self.add_growth = Const(int(np.log2(len(taps))))
        self._delay = 2

    def main(self, x):
        left = left_index(x) + self.add_growth
        for i in range(len(self.taps_fix_reversed)):
            self.next.mul[i] = resize(x * self.taps_fix_reversed[i], size_res=x, round_style=fixed_truncate, overflow_style=fixed_wrap)
            if i == 0:
                self.next.acc[0] = resize(self.mul[i], left, right_index(x), round_style=fixed_truncate, overflow_style=fixed_wrap)
            else:
                self.next.acc[i] = resize(self.acc[i - 1] + self.mul[i], left, right_index(x), round_style=fixed_truncate, overflow_style=fixed_wrap)

        return self.acc[-1]

    def model_main(self, x):
        return signal.lfilter(self.taps, [1.0], x)


def test():
    taps = signal.remez(8, [0, 0.1, 0.2, 0.5], [1, 0])
    dut = FIR(taps)
    inp = np.random.uniform(-1, 1, 128)

    assert_sim_match(dut, [Sfix(left=0, right=-17)], None, inp,
                     simulations=[SIM_MODEL, SIM_HW_MODEL, SIM_RTL, SIM_GATE],
                     dir_path='/home/gaspar/git/pyhacores/conversion',
                     rtol=1e-4, atol=1e-4)


def test_non_symmetric():
    taps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    dut = FIR(taps)
    inp = np.random.uniform(-1, 1, 128)

    assert_sim_match(dut, [Sfix(left=0, right=-17)], None, inp,
                     simulations=[SIM_MODEL, SIM_HW_MODEL, SIM_RTL, SIM_GATE],
                     dir_path='/home/gaspar/git/pyhacores/conversion',
                     rtol=1e-4, atol=1e-4)