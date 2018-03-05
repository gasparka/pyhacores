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
    class Dut(Hardware):
        def __init__(self, size):
            self.pack = Packager(size)
            self.window = Windower(size)
            self.DELAY = self.pack.DELAY + self.window.DELAY

        def main(self, data):
            out = self.pack.main(data)
            out = self.window.main(out)
            return out

        def model_main(self, data):
            out = self.pack.model_main(data)
            out = self.window.model_main(out)
            return out

    dut = Dut(M)
    inp = np.random.uniform(-1, 1, M) + np.random.uniform(-1, 1, M) * 1j

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA',
                                           # 'RTL'
                                           ])

    sims['PYHA'] = DataWithIndex.to2d(sims['PYHA'])
    assert sims_close(sims, rtol=1e-2)
