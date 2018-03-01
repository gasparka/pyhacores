from scipy import signal

import numpy as np
import pytest
from data import load_iq
from pyha import Hardware, simulate, sims_close, Complex, resize, Sfix


class Windower(Hardware):
    def __init__(self, M, window_type='hanning'):
        assert window_type == 'hanning'
        self.M = M
        self.WINDOW = np.hanning(M)
        self.control = 0
        self.out = Complex(0, 0, -17)
        self.DELAY = 1

    def main(self, complex_in):
        self.out.real = complex_in.real * self.WINDOW[self.control]
        self.out.imag = complex_in.imag * self.WINDOW[self.control]

        next_control = self.control + 1
        if next_control >= self.M:
            next_control = 0

        self.control = next_control
        return self.out

    def model_main(self, complex_in_list):
        complex_in_list = np.array(complex_in_list).reshape((-1, self.M))
        stack = np.hstack(complex_in_list * self.WINDOW)

        return stack


@pytest.mark.parametrize("M", [2, 4, 8, 16, 32, 64, 128, 256])
def test_windower(M):
    dut = Windower(M)
    inp = np.random.uniform(-1, 1, M) + np.random.uniform(-1, 1, M) * 1j

    sims = simulate(dut, inp, simulations=['MODEL', 'MODEL_PYHA', 'PYHA', 'RTL'])
    assert sims_close(sims, rtol=1e-2)


def test_fail():
    fft_points = 256
    extra_sims = ['RTL']

    inp = load_iq('/home/gaspar/git/pyhacores/data/f2404_fs16.896_one_hop.iq')
    inp = signal.decimate(inp, 8)
    inp *= 0.5
    print(len(inp))
    print(inp.max())

    # make sure input divides with fft_points
    inp = np.array(inp[:int(len(inp) // fft_points) * fft_points])

    dut = Windower(fft_points)
    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA'] + extra_sims)