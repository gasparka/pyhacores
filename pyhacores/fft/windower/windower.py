import numpy as np
import pytest
from pyha import Hardware, simulate, sims_close, Complex, Sfix
from scipy.signal import get_window

from pyhacores.fft.packager import DataWithIndex, unpackage, package


# 8 bit 9 was about 1k LE
# Total logic elements	631 / 39,600 ( 2 % )
# Embedded Multiplier 9-bit elements	4 / 232 ( 2 % )

class Windower(Hardware):
    """ Windowing function determines the frequency response of FFT bins. """
    def __init__(self, M, window='hanning', coefficient_bits=8):
        self.M = M
        self.window_pure = get_window(window, M)
        self.WINDOW = [Sfix(x, 0, -(coefficient_bits-1), round_style='round', overflow_style='saturate')
                       for x in self.window_pure]
        self.out = DataWithIndex(Complex())
        self.DELAY = 1

    def main(self, inp):
        # calculate output
        self.out = inp
        self.out.data = inp.data * self.WINDOW[inp.index]
        return self.out

    def model_main(self, complex_in_list):
        return complex_in_list * self.window_pure


def test_wtf():
    from pyhacores.fft import Windower
    from pyhacores.fft.packager import DataWithIndex, Packager, unpackage, package
    # NBVAL_IGNORE_OUTPUT
    input_signal = [0.0 + 0.0j] * 512
    input_signal[0] = 1.0 + 1.0j

    input_signal = np.reshape(input_signal, (-1, 128))
    dut = Windower(128)
    sims = simulate(dut, input_signal, output_callback=unpackage, input_callback=package, simulations=['MODEL', 'PYHA'])

    pass


@pytest.mark.parametrize("M", [4, 8, 16, 32, 64, 128, 256])
def test_windower(M):
    M = 1024 * 8
    dut = Windower(M)
    inp = np.random.uniform(-1, 1, size=(2, M)) + np.random.uniform(-1, 1, size=(2, M)) * 1j

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA',
                                           # 'RTL',
                                           # 'GATE'
                                           ],
                    conversion_path='/home/gaspar/git/pyhacores/playground',
                    output_callback=unpackage,
                    input_callback=package)

    assert sims_close(sims, rtol=1e-2, atol=1e-2)