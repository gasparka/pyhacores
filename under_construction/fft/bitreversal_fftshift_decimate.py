import pytest

from pyha import Hardware, simulate, sims_close, Sfix
import numpy as np
from pyha.simulation.simulation_interface import get_last_trained_object
from under_construction.fft.bit_reversal_fftshift import bit_reversed_indexes
from under_construction.fft.packager import DataWithIndex, unpackage, package


def build_lut(fft_size, decimation):
    rev_index = bit_reversed_indexes(fft_size)
    orig_inp = np.array(list(range(fft_size)))
    shift = np.fft.fftshift(orig_inp)
    rev = shift[rev_index]
    lut = rev // decimation
    return lut


class BitreversalFFTshiftDecimate(Hardware):
    def __init__(self, fft_size, decimation):
        assert decimation > 1
        self.DECIMATION = decimation
        self.DECIMATION_BITS = int(np.log2(decimation))
        self.FFT_SIZE = fft_size
        self.LUT = build_lut(fft_size, decimation)

        self.state = False
        self.mem0 = [Sfix(0.0, np.log2(decimation), -17)] * (fft_size // decimation)
        self.mem1 = [Sfix(0.0, np.log2(decimation), -17)] * (fft_size // decimation)
        self.out = DataWithIndex(0.0, 0)
        self.DELAY = fft_size + 1

    def main(self, inp):
        write_index = self.LUT[inp.index]

        if self.state:
            self.mem0[write_index] += inp.data
            if inp.index < self.FFT_SIZE / self.DECIMATION:
                self.out = DataWithIndex(self.mem1[inp.index] >> self.DECIMATION_BITS, index=inp.index, valid=True)
                self.mem1[inp.index] = 0.0
            else:
                self.out.valid = False
        else:
            self.mem1[write_index] += inp.data
            if inp.index < self.FFT_SIZE / self.DECIMATION:
                self.out = DataWithIndex(self.mem0[inp.index] >> self.DECIMATION_BITS, index=inp.index, valid=True)
                self.mem0[inp.index] = 0.0
            else:
                self.out.valid = False

        if inp.index == self.FFT_SIZE - 1:
            self.state = not self.state

        return self.out

    def model_main(self, x):
        rev_index = bit_reversed_indexes(self.FFT_SIZE)
        unrev = x[:, rev_index]
        unshift = np.fft.fftshift(unrev, axes=1)
        avg = np.reshape(unshift, (len(x), self.FFT_SIZE//self.DECIMATION, self.DECIMATION))
        avg = np.mean(avg, axis=2)
        return avg


@pytest.mark.parametrize("decimation", [2, 4, 8])
@pytest.mark.parametrize("fft_size", [256, 128, 64, 32])
@pytest.mark.parametrize("packets", [4, 3, 2, 1])
def test_basic(fft_size, decimation, packets):
    orig_inp = np.array(list(range(fft_size))) / 1000
    orig_inp = [orig_inp] * packets

    rev_index = bit_reversed_indexes(fft_size)
    shift = np.fft.fftshift(orig_inp, axes=1)
    input = shift[:, rev_index]

    dut = BitreversalFFTshiftDecimate(fft_size, decimation)

    sims = simulate(dut, input, simulations=['MODEL',
                                             'PYHA',
                                             'RTL',
                                             # 'GATE'
                                             ],
                    output_callback=unpackage,
                    input_callback=package,
                    conversion_path='/home/gaspar/git/pyhacores/playground'
                    )

    mod = get_last_trained_object()
    assert sims_close(sims)
