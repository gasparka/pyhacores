import timeit

import pytest
from pyha import Hardware, simulate, sims_close, Complex, resize
import numpy as np


def W(k, N):
    """ e^-j*2*PI*k*n/N, argument k = k * n """
    return np.exp(-1j * (2 * np.pi / N) * k)


class StageR2SDF(Hardware):
    def __init__(self, fft_size):
        self.fft_size = fft_size
        self.fft_half = fft_size // 2

        self.control_mask = (self.fft_half - 1)
        self.shr = [Complex() for _ in range(self.fft_half)]

        self.TWIDDLES = [W(i, self.fft_size) for i in range(self.fft_half)]

    def main(self, x, control):
        if not (control & self.fft_half):
            self.shr = [x] + self.shr[:-1]
            return self.shr[-1]
        else:
            up_real = resize(self.shr[-1].real + x.real, 0, -17)
            up_imag = resize(self.shr[-1].imag + x.imag, 0, -17)
            up = Complex(up_real, up_imag)
            # up = self.shr[-1] + x

            # down sub
            down_sub_real = resize(self.shr[-1].real - x.real, 0, -17)
            down_sub_imag = resize(self.shr[-1].imag - x.imag, 0, -17)

            twiddle = self.TWIDDLES[control & self.control_mask]
            down_real = resize((down_sub_real * twiddle.real) - (down_sub_imag * twiddle.imag), 0, -17)
            down_imag = resize((down_sub_real * twiddle.imag) + (down_sub_imag * twiddle.real), 0, -17)
            down = Complex(down_real, down_imag)
            # down = (self.shr[-1] - x) * self.TWIDDLES[control & self.control_mask]
            self.shr = [down] + self.shr[:-1]
            return up


class R2SDF(Hardware):
    def __init__(self, fft_size):
        self.fft_size = fft_size

        self.n_bits = int(np.log2(fft_size))
        # self.stages = [StageR2SDF(2 ** (pow + 1)) for pow in reversed(range(self.n_bits))]
        self.stage16 = StageR2SDF(16)
        self.stage8 = StageR2SDF(8)
        self.stage4 = StageR2SDF(4)
        self.stage2 = StageR2SDF(2)

        self.control = 0

        self.DELAY = fft_size - 1

    def main(self, x):
        next_control = (self.control + 1) % self.fft_size

        # execute stages
        out = x
        out = self.stage16.main(out, self.control)
        out = self.stage8.main(out, self.control)
        out = self.stage4.main(out, self.control)
        out = self.stage2.main(out, self.control)
        # out = x
        # for stage in self.stages:
        #     out = stage.main(out, self.control)

        self.control = next_control
        return out

    def model_main(self, x):
        from scipy.fftpack import fft
        x = np.array(x).reshape((-1, self.fft_size))
        stack = np.hstack(fft(x))

        # apply bit reversing ie. mess up the output order to match radix-2 algorithm
        # from under_construction.fft.bit_reversal import bit_reversed_indexes

        def bit_reverse(x, n_bits):
            return int(np.binary_repr(x, n_bits)[::-1], 2)

        def bit_reversed_indexes(N):
            return [bit_reverse(i, int(np.log2(N))) for i in range(N)]

        rev_index = bit_reversed_indexes(self.fft_size)
        if len(stack.shape) == 1:
            return stack[rev_index]
        assert 0


def test_conv():
    fft_size = 16
    dut = R2SDF(fft_size)
    inp = np.random.uniform(-1, 1, fft_size) + np.random.uniform(-1, 1, fft_size) * 1j
    inp *= 0.25 / 2

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA',
                                           'RTL'
                                           ], conversion_path='/home/gaspar/git/pyhacores/playground')
    assert sims_close(sims, rtol=1e-2)


@pytest.mark.parametrize("fft_size", [2, 4, 8, 16, 32, 64, 128, 256, 512])
def test_fft(fft_size):
    dut = R2SDF(fft_size)
    inp = np.random.uniform(-1, 1, fft_size) + np.random.uniform(-1, 1, fft_size) * 1j

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA'])
    assert sims_close(sims, rtol=1e-2)

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


