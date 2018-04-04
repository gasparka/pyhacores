from pyha import Hardware, simulate, sims_close
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
        self.DECIMATION = decimation
        self.FFT_SIZE = fft_size
        self.LUT = build_lut(fft_size, decimation)

        self.state = False
        self.mem0 = [0.0] * (fft_size // decimation)
        self.mem1 = [0.0] * (fft_size // decimation)
        self.out = DataWithIndex(0.0, 0)
        self.DELAY = fft_size + 1

    def main(self, inp):
        write_index = self.LUT[inp.index]

        if self.state:
            self.mem0[write_index] += inp.data
            if inp.index < self.FFT_SIZE / self.DECIMATION:
                self.out = DataWithIndex(self.mem1[inp.index], index=inp.index, valid=True)
            else:
                self.out.valid = False
        else:
            self.mem1[write_index] += inp.data
            if inp.index < self.FFT_SIZE / self.DECIMATION:
                self.out = DataWithIndex(self.mem0[inp.index], index=inp.index, valid=True)
            else:
                self.out.valid = False

        if inp.index == self.FFT_SIZE - 1:
            self.state = not self.state

        return self.out

    def model_main(self, x):
        rev_index = bit_reversed_indexes(self.FFT_SIZE)

        unrev = x.copy()
        for i, row in enumerate(x):
            unrev[i] = row[rev_index]

        unshift = np.fft.fftshift(unrev, axes=1)
        avg = np.reshape(unshift, (-1, self.DECIMATION))
        avg = np.mean(avg, axis=1)
        return [avg]


def test_basic():
    fft_size = 64
    orig_inp = np.array(list(range(fft_size))) / 100
    rev_index = bit_reversed_indexes(fft_size)
    shift = np.fft.fftshift(orig_inp)
    input = shift[rev_index]
    input = [input]
    dut = BitreversalFFTshiftDecimate(fft_size, decimation=1)

    sims = simulate(dut, input, simulations=['MODEL',
                                             'PYHA',
                                             'RTL',
                                             'GATE'
                                             ],
                    output_callback=unpackage,
                    input_callback=package,
                    conversion_path='/home/gaspar/git/pyhacores/playground'
                    )

    mod = get_last_trained_object()
    assert sims_close(sims, expected=[orig_inp])
