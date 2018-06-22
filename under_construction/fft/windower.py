import numpy as np
import pytest
from pyha import Hardware, simulate, sims_close, Complex, Sfix
from under_construction.fft.packager import DataWithIndex, Packager, unpackage, package


# 8 bit 9 was about 1k LE
# Total logic elements	631 / 39,600 ( 2 % )
# Embedded Multiplier 9-bit elements	4 / 232 ( 2 % )

class Windower(Hardware):
    def __init__(self, M, window_type='hanning', coefficient_bits=8):
        assert window_type == 'hanning'
        self.M = M
        self.window_pure = np.hanning(M)
        self.WINDOW = [Sfix(x, 0, -(coefficient_bits-1), round_style='round', overflow_style='saturate') for x in np.hanning(M)]
        # self.WINDOW = np.hanning(M)
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
    M = 1024 * 8
    dut = Windower(M)
    inp = np.random.uniform(-1, 1, size=(2, M)) + np.random.uniform(-1, 1, size=(2, M)) * 1j

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA',
                                           # 'RTL',
                                           'GATE'
                                           ],
                    conversion_path='/home/gaspar/git/pyhacores/playground',
                    output_callback=unpackage,
                    input_callback=package)

    assert sims_close(sims, rtol=1e-2, atol=1e-2)
