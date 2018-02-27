import pytest
from pyha import Hardware, simulate, sims_close, Complex
import numpy as np


def bit_reverse(x, n_bits):
    return int(np.binary_repr(x, n_bits)[::-1], 2)


def bit_reversed_indexes(N):
    return [bit_reverse(i, int(np.log2(N))) for i in range(N)]


class BitReversal(Hardware):

    def __init__(self, fft_size, fftshift=True):
        self.FFTSHIFT = fftshift
        self.FFT_SIZE = fft_size
        self.control = 0
        self.mem0 = [Complex() for _ in range(fft_size)]
        self.mem1 = [Complex() for _ in range(fft_size)]
        self.state = False
        self.n_bits = int(np.log2(fft_size))

        self.DELAY = fft_size

    def main(self, data):
        if self.FFTSHIFT:
            if self.control < self.FFT_SIZE//2:
                reversed_index = bit_reverse(self.control + self.FFT_SIZE // 2, self.n_bits)
            else:
                reversed_index = bit_reverse(self.control - self.FFT_SIZE // 2, self.n_bits)
        else:
            reversed_index = bit_reverse(self.control, self.n_bits)

        if self.state:
            self.mem0[self.control] = data
            ret = self.mem1[reversed_index]
        else:
            self.mem1[self.control] = data
            ret = self.mem0[reversed_index]

        next_control = self.control + 1
        if next_control == self.FFT_SIZE:
            next_control = 0
            self.state = not self.state

        self.control = next_control
        return ret

    def model_main(self, complex_in_list):
        complex_in_list = np.array(complex_in_list).reshape((-1, self.FFT_SIZE))

        ret = []
        for x in complex_in_list:
            rev = x[bit_reversed_indexes(self.FFT_SIZE)]
            if self.FFTSHIFT:
                rev = np.fft.fftshift(rev)
            ret.extend(rev)

        return ret


@pytest.mark.parametrize("fftshift", [True, False])
@pytest.mark.parametrize("N", [2, 4, 8, 16, 32, 64, 128, 256])
def test_bit_reversal(N, fftshift):
    inp = np.random.uniform(-1, 1, N) + np.random.uniform(-1, 1, N) * 1j
    inp = inp[bit_reversed_indexes(N)]
    dut = BitReversal(N, fftshift)
    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA'])
    assert sims_close(sims)
