from pyha import Hardware, simulate, sims_close
import numpy as np


def bit_reverse(x, n_bits):
    return int(np.binary_repr(x, n_bits)[::-1], 2)


def bit_reversed_indexes(N):
    return [bit_reverse(i, int(np.log2(N))) for i in range(N)]


class BitReversal(Hardware):

    def __init__(self, fft_size):
        self.fft_size = fft_size
        self.mem = [0 for _ in range(fft_size)]
        self.n_bits = int(np.log2(fft_size))

        # self.DELAY = fft_size//2

    def main(self, data, control):
        reversed_index = bit_reverse(control, self.n_bits)

        # write
        if control < self.fft_size // 2:
            self.mem[reversed_index] = data
        else:
            self.mem[control] = data

        # read
        if control & 1:
            return self.mem[reversed_index]
        else:
            return self.mem[control]


def test_bit_reversal():
    # inp = [0, 1, 2, 3]
    # expect = [0, 2, 1, 3]
    N = 4
    base = list(range(N))
    reverse = bit_reversed_indexes(N)

    dut = BitReversal(N)
    sims = simulate(dut, reverse, base, simulations=['PYHA'])

    import matplotlib.pyplot as plt
    # plt.plot(sims['MODEL'])
    # plt.plot(sims['PYHA'])
    # plt.show()
    print("\n\n\n", sims['PYHA'])
    # assert sims_close(sims)
