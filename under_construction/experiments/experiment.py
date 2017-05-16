from enum import Enum

from pyha.common.const import Const
from pyha.common.hwsim import HW
from pyha.common.sfix import Sfix, resize, left_index, right_index, fixed_truncate, fixed_wrap
from pyha.simulation.simulation_interface import assert_sim_match, SIM_HW_MODEL, SIM_MODEL, SIM_RTL, SIM_GATE, \
    plot_assert_sim_match, debug_assert_sim_match
from scipy import signal
import numpy as np
from scipy.signal import firwin

from pyhacores.filter.fir.model import FIR


class Experiment(HW):
    """ FIR filter, taps will be normalized to sum 1 """
    def __init__(self):
        taps1 = signal.remez(16, [0, 0.1, 0.2, 0.5], [1, 0])
        taps2 = signal.remez(16, [0, 0.2, 0.3, 0.5], [1, 0])

        # registers
        self.fir = [FIR(taps1), FIR(taps2)]

        # constants
        self._delay = self.fir[0]._delay

    def main(self, x):

        for i in range(len(self.fir)):
            tmp = self.fir[i].main(x)

        return self.fir[0].out, self.fir[1].out

    def model_main(self, x):
        return [fir.model_main(x) for fir in self.fir]


class State(Enum):
    ENUM0, ENUM1, ENUM2, ENUM3 = range(4)

class Experiment2(HW):
    def __init__(self):
        taps1 = firwin(128, 0.1)
        taps2 = firwin(128, 0.2)
        taps3 = firwin(128, 0.3)
        taps4 = firwin(128, 0.4)
        print(taps1)

        print(taps2)

        # registers
        self.fir = [FIR(taps1), FIR(taps2), FIR(taps3), FIR(taps4)]
        self.state = State.ENUM0

        # constants
        self._delay = self.fir[0]._delay

    def main(self, x):

        # if self.state == State.ENUM0:
        #     tmp = self.fir[0].main(x)
        #     self.next.state = State.ENUM1
        # elif self.state == State.ENUM1:
        #     tmp = self.fir[1].main(x)
        #     self.next.state = State.ENUM2
        # elif self.state == State.ENUM2:
        #     tmp = self.fir[2].main(x)
        #     self.next.state = State.ENUM3
        # else:
        #     tmp = self.fir[3].main(x)
        #     self.next.state = State.ENUM0

        tmp = self.fir[0].main(x)
        tmp = self.fir[1].main(x)
        tmp = self.fir[2].main(x)
        tmp = self.fir[3].main(x)

        return self.fir[0].out, self.fir[1].out, self.fir[2].out, self.fir[3].out

    def model_main(self, x):
        return [fir.model_main(x) for fir in self.fir]



class State3(Enum):
    ENUM0, ENUM1, ENUM2, ENUM3 = range(4)

class Sub3(HW):
    def __init__(self, coef):
        self.coef = coef

        # registers
        self.mul = 0
        # self.add = 0

        # constants
        self._delay = 1

    def main(self, a, b):
        self.next.mul = a * self.coef
        # self.next.add = self.mul + self.coe
        return self.mul

    def model_main(self, a, b):
        return np.array(a) * np.array(b)

class Experiment3(HW):
    def __init__(self):
        # registers
        self.dut = [Sub3(0), Sub3(1), Sub3(2), Sub3(3)]
        self.state = State3.ENUM0

        # constants
        self._delay = self.dut[0]._delay

    def main(self, a, b):

        if self.state == State3.ENUM0:
            tmp = self.dut[0].main(a, b)
            self.next.state = State3.ENUM1
        elif self.state == State3.ENUM1:
            tmp = self.dut[1].main(a, b)
            self.next.state = State3.ENUM2
        elif self.state == State3.ENUM2:
            tmp = self.dut[2].main(a, b)
            self.next.state = State3.ENUM3
        else:
            # elif self.state == State3.ENUM3:
            tmp = self.dut[3].main(a, b)
            self.next.state = State3.ENUM0

        # tmp = self.dut[0].main(a, b)
        # tmp = self.dut[1].main(a, b)
        # tmp = self.dut[2].main(a, b)
        # tmp = self.dut[3].main(a, b)

        return self.dut[0].mul, self.dut[1].mul, self.dut[2].mul, self.dut[3].mul
        # return self.fir[0].out, self.fir[1].out

    def model_main(self, a, b):
        return [fir.model_main(a, b) for fir in self.dut]


def test_experiment1():
    np.random.seed(12345)
    dut = Experiment()
    inp = np.random.uniform(-1, 1, 32)

    assert_sim_match(dut, None, inp,
                     simulations=[SIM_MODEL, SIM_HW_MODEL, SIM_RTL, SIM_GATE],
                     dir_path='/home/gaspar/git/pyhacores/playground')


def test_experiment2():
    np.random.seed(12345)
    dut = Experiment2()
    inp = np.random.uniform(-1, 1, 32)

    debug_assert_sim_match(dut, None, inp,
                     simulations=[SIM_MODEL, SIM_HW_MODEL, SIM_RTL, SIM_GATE],
                     dir_path='/home/gaspar/git/pyhacores/playground')


def test_experiment3():
    np.random.seed(12345)
    dut = Experiment3()
    a = list(range(32))
    b = list(range(32))

    debug_assert_sim_match(dut, None, a, b,
                     simulations=[SIM_MODEL, SIM_HW_MODEL, SIM_RTL, SIM_GATE],
                     dir_path='/home/gaspar/git/pyhacores/playground')