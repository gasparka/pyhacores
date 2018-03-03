from pyha.common.stream import Stream
from scipy import signal
import numpy as np
import pytest
from data import load_iq
from pyha import Hardware, simulate, sims_close, Complex


class Windower(Hardware):
    def __init__(self, M, window_type='hanning'):
        assert window_type == 'hanning'
        self.M = M
        self.WINDOW = np.hanning(M)
        self.control = 0

        self.out = Stream(Complex(), True, False, False)
        self.DELAY = 1

    def main(self, inp):
        """
        :type inp: Stream(Complex)
        """
        if not inp.valid:
            return self.out

        self.out.data.real = inp.data.real * self.WINDOW[self.control]
        self.out.data.imag = inp.data.imag * self.WINDOW[self.control]
        self.out.package_start = self.control == 0
        self.out.package_end = self.control == self.M - 1
        self.out.valid = True

        next_control = self.control + 1
        if next_control >= self.M:
            next_control = 0

        self.control = next_control

        return self.out

    def model_main(self, complex_in_list):
        return complex_in_list * self.WINDOW


@pytest.mark.parametrize("M", [4, 8, 16, 32, 64, 128, 256])
def test_windower(M):
    dut = Windower(M)
    packets = np.random.randint(1, 4)
    inp = np.random.uniform(-1, 1, size=(packets, M)) + np.random.uniform(-1, 1, size=(packets, M)) * 1j

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA', 'RTL'])

    # uns = unstream(sims['PYHA'])
    assert sims_close(sims, rtol=1e-2)


def test_fail():
    fft_points = 256
    extra_sims = ['RTL', 'GATE']

    inp = load_iq('/home/gaspar/git/pyhacores/data/f2404_fs16.896_one_hop.iq')
    inp = signal.decimate(inp, 8)
    inp *= 0.5
    print(len(inp))
    print(inp.max())

    # make sure input divides with fft_points
    inp = np.array(inp[:int(len(inp) // fft_points) * fft_points])

    dut = Windower(fft_points)
    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA'] + extra_sims, conversion_path='/home/gaspar/git/pyhacores/playground')
