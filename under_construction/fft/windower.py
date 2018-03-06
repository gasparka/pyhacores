import numpy as np
import pytest
from pyha import Hardware, simulate, sims_close, Complex
from under_construction.fft.packager import DataWithIndex, Packager


class Windower(Hardware):
    def __init__(self, M, window_type='hanning'):
        assert window_type == 'hanning'
        self.M = M
        self.WINDOW = np.hanning(M)

        self.out = DataWithIndex(Complex(0.0, 0, -17), 0)
        self.DELAY = 1

    def main(self, inp):
        self.out = inp
        self.out.data.real = inp.data.real * self.WINDOW[inp.index]
        self.out.data.imag = inp.data.imag * self.WINDOW[inp.index]
        return self.out

    def model_main(self, complex_in_list):
        return complex_in_list * self.WINDOW


@pytest.mark.parametrize("M", [4, 8, 16, 32, 64, 128, 256])
def test_windower(M):
    dut = Windower(M)
    inp = np.random.uniform(-1, 1, size=(2, M)) + np.random.uniform(-1, 1, size=(2, M)) * 1j

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA',
                                           # 'RTL'
                                           ],
                    output_callback=DataWithIndex.unpack,
                    input_callback=DataWithIndex.pack)

    assert sims_close(sims, rtol=1e-2)
