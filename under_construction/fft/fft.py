import pytest
from pyha import Hardware, simulate, sims_close, Complex
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

    def main(self, x, control):
        if not (control & self.fft_half):
            self.shr = [x] + self.shr[:-1]
            return self.shr[-1]
        else:
            up = self.shr[-1] + x
            down = (self.shr[-1] - x) * W(control & self.control_mask, self.fft_size)
            self.shr = [down] + self.shr[:-1]
            return up


class R2SDF(Hardware):
    def __init__(self, fft_size):
        self.fft_size = fft_size

        self.n_bits = int(np.log2(fft_size))
        self.stages = [StageR2SDF(2 ** (pow + 1)) for pow in reversed(range(self.n_bits))]
        self.control = 0

        self.DELAY = fft_size - 1

    def main(self, x):
        next_control = (self.control + 1) % self.fft_size

        # execute stages
        out = x
        for stage in self.stages:
            out = stage.main(out, self.control)

        self.control = next_control
        return out

    def model_main(self, x):
        from scipy.fftpack import fft
        x = np.array(x).reshape((-1, self.fft_size))
        stack = np.hstack(fft(x))

        # apply bit reversing ie. mess up the output order to match radix-2 algorithm
        from under_construction.fft.bit_reversal import bit_reversed_indexes
        rev_index = bit_reversed_indexes(self.fft_size)
        if len(stack.shape) == 1:
            return stack[rev_index]
        assert 0


@pytest.mark.parametrize("fft_size", [2, 4, 8, 16, 32, 64, 128])
def test_fft32(fft_size):
    dut = R2SDF(fft_size)
    inp = np.random.uniform(-1, 1, fft_size) + np.random.uniform(-1, 1, fft_size) * 1j

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA'])
    assert sims_close(sims, rtol=1e-2)
