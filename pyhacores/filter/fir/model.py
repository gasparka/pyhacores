from pyha.common.const import Const
from pyha.common.hwsim import HW
from pyha.common.sfix import Sfix, resize, left_index, right_index, fixed_truncate, fixed_wrap
from pyha.simulation.simulation_interface import assert_sim_match, SIM_HW_MODEL, SIM_MODEL, SIM_RTL, SIM_GATE, \
    plot_assert_sim_match
from scipy import signal
import numpy as np


def rescale_taps(taps):
    """
    Rescale taps in that way that their abs sum equals 1, this assures no overflows in filter core
    """
    taps = np.array(taps)
    cs = sum(abs(taps))
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
                self.next.acc[i] = self.acc[i - 1] + self.mul[i]

        self.next.out = self.acc[-1]
        return self.out

    def model_main(self, x):
        return signal.lfilter(self.taps, [1.0], x)


def test_symmetric():
    taps = [0.01, 0.02, 0.03, 0.04, 0.03, 0.02, 0.01]
    dut = FIR(taps)
    inp = np.random.uniform(-1, 1, 64)

    assert_sim_match(dut, None, inp,
                     simulations=[SIM_MODEL, SIM_HW_MODEL, SIM_RTL, SIM_GATE],
                     dir_path='/home/gaspar/git/pyhacores/playground')


def test_sfix_bug():
    """ There was Sfix None bound based bug that made only 5. output different """
    np.random.seed(4)
    taps = [0.01, 0.02, 0.03, 0.04, 0.03, 0.02, 0.01]
    dut = FIR(taps)
    inp = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    assert_sim_match(dut, None, inp,
                     simulations=[SIM_MODEL, SIM_HW_MODEL])


def test_non_symmetric():
    taps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    dut = FIR(taps)
    inp = np.random.uniform(-1, 1, 128)

    assert_sim_match(dut, None, inp,
                     simulations=[SIM_MODEL, SIM_HW_MODEL, SIM_RTL, SIM_GATE],
                     rtol=1e-4,
                     atol=1e-4)


def test_remez32():
    np.random.seed(12345)
    taps = signal.remez(32, [0, 0.1, 0.2, 0.5], [1, 0])
    dut = FIR(taps)
    inp = np.random.uniform(-1, 1, 512) * 0.9

    assert_sim_match(dut, None, inp,
                     simulations=[SIM_MODEL, SIM_HW_MODEL, SIM_RTL, SIM_GATE],
                     dir_path='/home/gaspar/git/pyhacores/playground')


def test_remez128():
    taps = signal.remez(128, [0, 0.1, 0.2, 0.5], [1, 0])
    dut = FIR(taps)
    inp = np.random.uniform(-1, 1, 128)

    assert_sim_match(dut, None, inp,
                     simulations=[SIM_MODEL, SIM_HW_MODEL, SIM_RTL, SIM_GATE])