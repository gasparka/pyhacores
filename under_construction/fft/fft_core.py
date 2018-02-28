import timeit
from scipy import signal

import pytest
from data import load_iq
from pyha import Hardware, simulate, sims_close, Complex, resize
import numpy as np


def W(k, N):
    """ e^-j*2*PI*k*n/N, argument k = k * n """
    return np.exp(-1j * (2 * np.pi / N) * k)


class StageR2SDF(Hardware):
    def __init__(self, fft_size):
        self.FFT_SIZE = fft_size
        self.FFT_HALF = fft_size // 2

        self.CONTROL_MASK = (self.FFT_HALF - 1)
        self.shr = [Complex() for _ in range(self.FFT_HALF)]


        self.TWIDDLES = [W(i, self.FFT_SIZE) for i in range(self.FFT_HALF)]

    def main(self, x, control):

        up_real = resize(self.shr[-1].real + x.real, 0, -17)
        up_imag = resize(self.shr[-1].imag + x.imag, 0, -17)

        if self.FFT_HALF > 4:
            up_real = up_real >> 1
            up_imag = up_imag >> 1

        up = Complex(up_real, up_imag)
        # up = self.shr[-1] + x

        # down sub
        down_sub_real = resize(self.shr[-1].real - x.real, 0, -17)
        down_sub_imag = resize(self.shr[-1].imag - x.imag, 0, -17)

        twiddle = self.TWIDDLES[control & self.CONTROL_MASK]
        down_real = resize((down_sub_real * twiddle.real) - (down_sub_imag * twiddle.imag), 0, -17)
        down_imag = resize((down_sub_real * twiddle.imag) + (down_sub_imag * twiddle.real), 0, -17)

        if self.FFT_HALF > 4:
            down_real = down_real >> 1
            down_imag = down_imag >> 1

        down = Complex(down_real, down_imag)
        # down = (self.shr[-1] - x) * self.TWIDDLES[control & self.control_mask]

        if not (control & self.FFT_HALF):
            self.shr = [x] + self.shr[:-1]
            return self.shr[-1]
        else:
            self.shr = [down] + self.shr[:-1]
            return up


class R2SDF(Hardware):
    def __init__(self, fft_size):
        self.FFT_SIZE = fft_size

        self.n_bits = int(np.log2(fft_size))
        self.stages = [StageR2SDF(2 ** (pow + 1)) for pow in reversed(range(self.n_bits))]
        # self.stage32 = StageR2SDF(32)
        # self.stage16 = StageR2SDF(16)
        # self.stage8 = StageR2SDF(8)
        # self.stage4 = StageR2SDF(4)
        # self.stage2 = StageR2SDF(2)

        self.control = 0
        self.GAIN_CORRECTION = 2 ** (0 if self.n_bits - 3 < 0 else -(self.n_bits - 3))
        self.DELAY = fft_size - 1

    def main(self, x):
        next_control = self.control + 1
        if next_control == self.FFT_SIZE:
            next_control = 0

        # execute stages
        # out = x
        # out = self.stage32.main(out, self.control)
        # out = self.stage16.main(out, self.control)
        # out = self.stage8.main(out, self.control)
        # out = self.stage4.main(out, self.control)
        # out = self.stage2.main(out, self.control)
        out = x
        for stage in self.stages:
            out = stage.main(out, self.control)

        self.control = next_control
        return out

    def model_main(self, x):
        from scipy.fftpack import fft
        x = np.array(x).reshape((-1, self.FFT_SIZE))
        ffts = fft(x)

        # apply bit reversing ie. mess up the output order to match radix-2 algorithm
        # from under_construction.fft.bit_reversal import bit_reversed_indexes

        def bit_reverse(x, n_bits):
            return int(np.binary_repr(x, n_bits)[::-1], 2)

        def bit_reversed_indexes(N):
            return [bit_reverse(i, int(np.log2(N))) for i in range(N)]

        rev_index = bit_reversed_indexes(self.FFT_SIZE)
        for i, _ in enumerate(ffts):
            ffts[i] = ffts[i][rev_index]

        return np.hstack(ffts)




def test_conv():
    fft_size = 32
    dut = R2SDF(fft_size)
    inp = np.random.uniform(-1, 1, fft_size) + np.random.uniform(-1, 1, fft_size) * 1j
    inp *= 0.25 / 2

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA',
                                           'RTL', 'GATE'
                                           ], conversion_path='/home/gaspar/git/pyhacores/playground')
    assert sims_close(sims, rtol=1e-2)


@pytest.mark.parametrize("fft_size", [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 2048*2])
def test_fft(fft_size):
    dut = R2SDF(fft_size)
    inp = np.random.uniform(-1, 1, fft_size) + np.random.uniform(-1, 1, fft_size) * 1j

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA'])
    assert sims_close(sims, rtol=1e-2)


def test_fail():
    inp = load_iq('/home/gaspar/git/pyhacores/data/f2404_fs16.896_one_hop.iq')
    inp = signal.decimate(inp, 8)
    inp *= 0.25
    print(len(inp))
    print(inp.max())

    fft_points = 256
    # make sure input divides with fft_points
    inp = np.array(inp[:int(len(inp) // fft_points) * fft_points])
    dut = R2SDF(fft_points)
    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA'])

# import pyha.simulation.simulation_interface import simulate
if __name__ == '__main__':
    print(timeit.timeit('import pyha; pyha.Complex()'))
    print(timeit.timeit('from pyha.simulation.simulation_interface import simulate'))

    # fft_size = 256
    # inp = np.random.uniform(-1, 1, fft_size) + np.random.uniform(-1, 1, fft_size) * 1j
    #
    # dut = R2SDF(fft_size)
    # sims = simulate(dut, inp, simulations=['PYHA'])
    # # assert sims_close(sims, rtol=1e-2)
    #
    # #python -m plop.collector -f flamegraph fft_core.py
    # # ./git/FlameGraph/flamegraph.pl --width 5000 x.flame > flame.svg


