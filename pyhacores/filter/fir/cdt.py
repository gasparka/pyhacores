from enum import Enum

from pyha.common.const import Const
from pyha.common.hwsim import HW
from pyha.common.sfix import Sfix, resize, left_index, right_index, fixed_truncate, fixed_wrap
from pyha.simulation.simulation_interface import assert_sim_match, SIM_HW_MODEL, SIM_MODEL, SIM_RTL, SIM_GATE, \
    plot_assert_sim_match, debug_assert_sim_match
from scipy import signal
import numpy as np

class State3(Enum):
    ENUM0, ENUM1 = range(2)

class Sub3(HW):
    def __init__(self, coef):
        self.coef = coef
        self.coef2 = coef+10
        # registers
        self.mul1 = 0
        self.mul2 = 0
        self.sum = 0

        # constants
        self._delay = 1

    def main(self, a):
        self.next.mul1 = a * self.coef
        self.next.mul2 = self.mul1 * self.coef2
        self.next.sum = self.mul1 + self.mul2
        return self.sum

    def model_main(self, a):
        return np.array(a) * np.array(a)


class Experiment3(HW):
    def __init__(self):
        # registers
        self.dut = [Sub3(0), Sub3(1)]
        # self.state = 0
        self.state = State3.ENUM0

        # constants
        self._delay = self.dut[0]._delay

    def main(self, a):

        # for i in range(len(self.dut)):
        #     if self.state == i:
        #         tmp = self.dut[i].main(a)
        #         self.next.state = i + 1
        #
        #         if self.state == 1:
        #             self.next.state = 0


        if self.state == State3.ENUM0:
            tmp = self.dut[0].main(a)
            self.next.state = State3.ENUM1
            # self.dut[1].next.mul = self.dut[1].mul
        else:
        # elif self.state == State3.ENUM1:
            tmp = self.dut[1].main(a)
            self.next.state = State3.ENUM0
            # self.dut[0].next.mul = self.dut[0].mul

        # tmp = self.dut[0].main(a, b)
        # tmp = self.dut[1].main(a, b)
        # tmp = self.dut[2].main(a, b)
        # tmp = self.dut[3].main(a, b)

        return self.dut[0].sum, self.dut[1].sum

    def model_main(self, a):
        return [fir.model_main(a) for fir in self.dut]


def test_experiment3():
    np.random.seed(12345)
    dut = Experiment3()
    a = list(range(32))

    debug_assert_sim_match(dut, None, a,
                     simulations=[SIM_MODEL, SIM_HW_MODEL, SIM_RTL, SIM_GATE],
                     dir_path='/home/gaspar/git/pyhacores/playground')