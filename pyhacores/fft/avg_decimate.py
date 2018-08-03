import numpy as np
import pytest
from pyha import Hardware, simulate, sims_close, Sfix
from pyha.common.util import is_power2
from under_construction.fft.packager import DataWithIndex


class AvgDecimate(Hardware):
    def __init__(self, decimation):
        assert is_power2(decimation)
        self.DECIMATION = decimation
        self.DECIMATE_BITS = int(np.log2(decimation))
        self.sum = Sfix(0, self.DECIMATE_BITS, -17)
        self.DELAY = self.DECIMATION + 1

        self.out = DataWithIndex(Sfix(0.0, 0, -17), index=0, valid=False)

    def main(self, inp):
        next_index = (inp.index >> self.DECIMATE_BITS) - 1
        if next_index != self.out.index:
            self.out = DataWithIndex(self.sum >> self.DECIMATE_BITS, index=next_index, valid=True)
            self.sum = inp.data
        else:
            self.out.valid = False
            self.sum += inp.data

        return self.out

    def model_main(self, x):
        x = np.array(x).T
        splits = np.split(x, len(x) // self.DECIMATION)
        avg = np.average(splits, axis=1).T

        return avg


@pytest.mark.parametrize("M", [2, 4, 8, 16, 32, 64, 128, 256])
def test_avgdecimate(M):
    dut = AvgDecimate(M)

    packages = np.random.randint(1, 4)
    inp = np.random.uniform(-1, 1, size=(packages, 1024))

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA',
                                           'RTL'
                                           ],
                    output_callback=unpackage,
                    input_callback=package)
    assert sims_close(sims)


def test_simple():
    M = 2
    dut = AvgDecimate(M)
    inp = [[0.5, 0.5, 0.2, 0.1, 0.9, 0.8], [0.5, 0.5, 0.2, 0.1, 0.6, 0.7]]

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA',
                                           'RTL'
                                           ],
                    output_callback=unpackage,
                    input_callback=package)
    assert sims_close(sims)