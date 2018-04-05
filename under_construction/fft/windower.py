import numpy as np
import pytest
from pyha import Hardware, simulate, sims_close, Complex, Sfix
from under_construction.fft.packager import DataWithIndex, Packager, unpackage, package


class Windower(Hardware):
    def __init__(self, M, window_type='hanning'):
        assert window_type == 'hanning'
        self.M = M
        self.window_pure = np.hanning(M)
        # self.WINDOW = [Sfix(x, 0, -7) for x in np.hanning(M)]
        self.WINDOW = np.hanning(M)
        self.out = DataWithIndex(Complex())
        self.DELAY = 1

    def main(self, inp):
        # calculate output
        self.out = inp
        self.out.data = inp.data * self.WINDOW[inp.index]
        return self.out

    def model_main(self, complex_in_list):
        return complex_in_list * self.window_pure


@pytest.mark.parametrize("M", [4, 8, 16, 32, 64, 128, 256])
def test_windower(M):
    dut = Windower(M)
    inp = np.random.uniform(-1, 1, size=(2, M)) + np.random.uniform(-1, 1, size=(2, M)) * 1j

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA',
                                           'RTL',
                                           # 'GATE'
                                           ],
                    conversion_path='/home/gaspar/git/pyhacores/playground',
                    output_callback=unpackage,
                    input_callback=package)

    assert sims_close(sims, rtol=1e-2)
