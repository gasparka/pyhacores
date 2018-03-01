import pytest
from pyha import Hardware, simulate, sims_close, Sfix, Complex
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
        self.N_BITS = int(np.log2(fft_size))

        self.DELAY = fft_size

    def bit_reverse(self, control):
        cfix = Sfix(control, self.N_BITS, 0)
        ret = cfix
        for i in range(self.N_BITS):
            ret[self.N_BITS - i - 1] = cfix[i]

        return int(ret)

    def main(self, data):
        read_index = self.control
        if self.FFTSHIFT:
            if self.control < self.FFT_SIZE // 2:
                read_index += self.FFT_SIZE // 2
            else:
                read_index -= self.FFT_SIZE // 2

        read_index = self.bit_reverse(read_index)

        if self.state:
            self.mem0[self.control] = data
            ret = self.mem1[read_index]
        else:
            self.mem1[self.control] = data
            ret = self.mem0[read_index]

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
    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA', 'RTL', 'GATE'], conversion_path='/home/gaspar/git/pyhacores/playground')
    assert sims_close(sims)
