import pytest

from pyha import Hardware, simulate, sims_close, Sfix, resize
import numpy as np

from pyha.common.ram import RAM
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

        self.state = True
        self.mem0 = RAM([Sfix(0.0, np.log2(decimation), -17)] * (fft_size // decimation))
        self.mem1 = RAM([Sfix(0.0, np.log2(decimation), -17)] * (fft_size // decimation))
        self.out = DataWithIndex(Sfix(0.0, np.log2(decimation), -17), 0)
        self.DELAY = fft_size + 1
        self.read = Sfix(0.0, np.log2(decimation), -17)

    def main(self, inp):
        write_index = self.LUT[inp.index]
        write_index_future = self.LUT[(inp.index + 1) % self.FFT_SIZE]

        if self.state:
            self.read = self.mem0.delayed_read(write_index_future)
            res = resize(self.read + inp.data, self.DECIMATION_BITS, -17)
            self.mem0.delayed_write(write_index, res)

            if inp.index < self.FFT_SIZE / self.DECIMATION:
                read = self.mem1.delayed_read(inp.index) >> self.DECIMATION_BITS
                self.out = DataWithIndex(read, index=inp.index, valid=True)
                res = Sfix(0.0, self.DECIMATION_BITS, -17)
                self.mem1.delayed_write(inp.index, res)
                # self.out = DataWithIndex(self.mem1[inp.index] >> self.DECIMATION_BITS, index=inp.index, valid=True)
                # self.mem1[inp.index] = 0.0
            else:
                self.out.valid = False

        else:
            self.read = self.mem1.delayed_read(write_index_future)
            res = resize(self.read + inp.data, self.DECIMATION_BITS, -17)
            self.mem1.delayed_write(write_index, res)
            if inp.index < self.FFT_SIZE / self.DECIMATION:
                read = self.mem0.delayed_read(inp.index) >> self.DECIMATION_BITS
                self.out = DataWithIndex(read, index=inp.index, valid=True)
                res = Sfix(0.0, self.DECIMATION_BITS, -17)
                self.mem0.delayed_write(inp.index, res)
                # self.out = DataWithIndex(self.mem0[inp.index] >> self.DECIMATION_BITS, index=inp.index, valid=True)
                # self.mem0[inp.index] = 0.0
            else:
                self.out.valid = False

        if inp.index == self.FFT_SIZE - 1:
            self.state = not self.state
            self.read = Sfix(0.0, self.DECIMATION_BITS, -17)

        return self.out

    def model_main(self, x):
        rev_index = bit_reversed_indexes(self.FFT_SIZE)
        unrev = x[:, rev_index]
        unshift = np.fft.fftshift(unrev, axes=1)
        avg = np.reshape(unshift, (len(x), self.FFT_SIZE//self.DECIMATION, self.DECIMATION))
        avg = np.mean(avg, axis=2)
        return avg


def test_basicc():
    fft_size = 128
    decimation = 4
    packets = 4
    orig_inp = np.array(list(range(fft_size))) / 100
    orig_inp = [orig_inp] * packets

    rev_index = bit_reversed_indexes(fft_size)
    shift = np.fft.fftshift(orig_inp, axes=1)
    input = shift[:, rev_index]

    dut = BitreversalFFTshiftDecimate(fft_size, decimation)

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
    assert sims_close(sims)


@pytest.mark.parametrize("decimation", [2, 4, 8])
@pytest.mark.parametrize("fft_size", [256, 128, 64, 32])
@pytest.mark.parametrize("packets", [4, 3, 2, 1])
def test_basic(fft_size, decimation, packets):
    fft_size = 1024 * 8
    decimation = 32
    packets = 2
    orig_inp = np.random.uniform(-1, 1, fft_size * packets)
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
