import pytest
from pyha import Hardware, simulate, sims_close, Sfix, Complex
import numpy as np
from copy import copy
from under_construction.fft.packager import DataWithIndex


def bit_reverse(x, n_bits):
    return int(np.binary_repr(x, n_bits)[::-1], 2)


def bit_reversed_indexes(N):
    return [bit_reverse(i, int(np.log2(N))) for i in range(N)]


class BitReversal(Hardware):

    def __init__(self, fft_size, fftshift=True):
        self.FFTSHIFT = fftshift
        self.FFT_SIZE = fft_size
        self.N_BITS = int(np.log2(fft_size))
        self.DELAY = fft_size + 1

        self.mem0 = [Complex() for _ in range(fft_size)]
        self.mem1 = [Complex() for _ in range(fft_size)]
        self.state = False
        self.out = DataWithIndex(Complex(0.0, 0, -17), 0)

    def bit_reverse(self, control):
        cfix = Sfix(control, self.N_BITS, 0)
        ret = copy(cfix)
        for i in range(self.N_BITS):
            ret[self.N_BITS - i - 1] = cfix[i]

        return int(ret)

    def main(self, inp):
        read_index = inp.index
        write_index = inp.index

        if self.FFTSHIFT:
            if inp.index < self.FFT_SIZE // 2:
                read_index += self.FFT_SIZE // 2
            else:
                read_index -= self.FFT_SIZE // 2

        read_index = self.bit_reverse(read_index)

        if self.state:
            self.mem0[write_index] = inp.data
            ret = self.mem1[read_index]
        else:
            self.mem1[write_index] = inp.data
            ret = self.mem0[read_index]

        if inp.index == self.FFT_SIZE - 1:
            self.state = not self.state

        self.out.data = ret
        self.out.index = inp.index
        return self.out

    def model_main(self, complex_in_list):
        ret = []
        for x in complex_in_list:
            rev = x[bit_reversed_indexes(self.FFT_SIZE)]
            if self.FFTSHIFT:
                rev = np.fft.fftshift(rev)
            ret += [rev]

        return ret


@pytest.mark.parametrize("fftshift", [True, False])
@pytest.mark.parametrize("N", [2, 4, 8, 16, 32, 64, 128, 256])
def test_bit_reversal(N, fftshift):
    packages = np.random.randint(2, 4)
    inp = np.random.uniform(-1, 1, (packages, N)) + np.random.uniform(-1, 1, (packages, N)) * 1j
    inp = [x[bit_reversed_indexes(N)] for x in inp]

    dut = BitReversal(N, fftshift)
    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA',
                                           'RTL',
                                           # 'GATE'
                                           ],
                    output_callback=DataWithIndex._pyha_unpack,
                    input_callback=DataWithIndex._pyha_pack)
    assert sims_close(sims)


def test_simple():
    N = 16
    inp = np.array(list(range(N))).astype(complex) / 100
    inp = inp[bit_reversed_indexes(N)]
    inp = [inp, inp]
    dut = BitReversal(N)
    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA',
                                           # 'RTL',
                                           # 'GATE'
                                           ],
                    output_callback=DataWithIndex._pyha_unpack,
                    input_callback=DataWithIndex._pyha_pack
                    )
    assert sims_close(sims)
