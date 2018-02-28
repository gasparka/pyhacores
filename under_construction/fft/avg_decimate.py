import numpy as np
import pytest
from pyha import Hardware, simulate, sims_close, Complex, resize, Sfix


class Stream(Hardware):
    def __init__(self, data, valid):
        self.data = data
        self.valid = valid

    def _pyha_on_simulation_output(self, list_data):
        ret = [x.data for x in list_data if x.valid]
        return ret


class AvgDecimate(Hardware):
    def __init__(self, decimate_by):
        self.DECIMATE_BY = decimate_by
        self.DECIMATE_BITS = int(np.log2(decimate_by))
        self.control = 0
        self.sum = Sfix(0, self.DECIMATE_BITS, -17)
        self.DELAY = self.DECIMATE_BY

    def main(self, float_in):
        if self.control == 0:
            valid = True
            self.sum = float_in
        else:
            valid = False
            self.sum += float_in

        next_control = self.control + 1
        if next_control == self.DECIMATE_BY:
            next_control = 0
        self.control = next_control

        return Stream(self.sum >> self.DECIMATE_BITS, valid)

    def model_main(self, x):
        x = np.array(x)
        x = np.split(x, len(x) // self.DECIMATE_BY)
        avg = np.average(x, axis=1)
        return avg

@pytest.mark.parametrize("M", [2, 4, 8, 16, 32, 64, 128, 256])
def test_avgdecimate(M):
    dut = AvgDecimate(M)
    inp = np.random.uniform(-1, 1, M*5)
    # inp = [0.5, 0.5, 0.2, 0.1, 0.9, 0.1]

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA', 'RTL'])
    print(sims)
    assert sims_close(sims)
