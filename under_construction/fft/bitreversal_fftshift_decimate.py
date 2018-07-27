import pytest

from pyha import Hardware, simulate, sims_close, Sfix, resize, scalb
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
    def __init__(self, fft_size, avg_freq_axis, avg_time_axis):
        assert avg_freq_axis > 1
        self.AVG_FREQ_AXIS = avg_freq_axis
        self.AVG_TIME_AXIS = avg_time_axis
        self.ACCUMULATION_BITS = int(np.log2(avg_freq_axis)) * avg_time_axis
        self.FFT_SIZE = fft_size
        self.LUT = build_lut(fft_size, avg_freq_axis)

        self.state = True
        self.mem0 = RAM([Sfix(0.0, 0, -35)] * (fft_size // avg_freq_axis))
        self.mem1 = RAM([Sfix(0.0, 0, -35)] * (fft_size // avg_freq_axis))
        self.out = DataWithIndex(Sfix(0.0, 0, -35), 0)
        self.DELAY = fft_size + 1

    def main(self, inp):
        write_index = self.LUT[inp.index]
        write_index_future = self.LUT[(inp.index + 1) % self.FFT_SIZE]

        if self.state:
            read = self.mem0.delayed_read(write_index_future)
            if inp.index == 0:
                read = 0.0
            res = resize(read + scalb(inp.data, -self.ACCUMULATION_BITS), 0, -35)
            self.mem0.delayed_write(write_index, res)

            if inp.index < self.FFT_SIZE / self.AVG_FREQ_AXIS:
                read = self.mem1.delayed_read(inp.index)
                self.out = DataWithIndex(read, index=inp.index, valid=True)
                res = Sfix(0.0, 0, -35)
                self.mem1.delayed_write(inp.index, res)
                # self.out = DataWithIndex(self.mem1[inp.index] >> self.DECIMATION_BITS, index=inp.index, valid=True)
                # self.mem1[inp.index] = 0.0
            else:
                self.out.valid = False

        else:
            read = self.mem1.delayed_read(write_index_future)
            if inp.index == 0:
                read = 0.0
            res = resize(read + scalb(inp.data, -self.ACCUMULATION_BITS), 0, -35)
            self.mem1.delayed_write(write_index, res)
            if inp.index < self.FFT_SIZE / self.AVG_FREQ_AXIS:
                read = self.mem0.delayed_read(inp.index)
                self.out = DataWithIndex(read, index=inp.index, valid=True)
                res = Sfix(0.0, 0, -35)
                self.mem0.delayed_write(inp.index, res)
                # self.out = DataWithIndex(self.mem0[inp.index] >> self.DECIMATION_BITS, index=inp.index, valid=True)
                # self.mem0[inp.index] = 0.0
            else:
                self.out.valid = False

        if inp.index == self.FFT_SIZE - 1:
            self.state = not self.state
            # self.read = Sfix(0.0, self.DECIMATION_BITS, -17)

        # return self.out
        out = DataWithIndex(read, index=self.out.index, valid=self.out.valid)
        if self.state:
            out.data = self.mem1.get_readregister()
        else:
            out.data = self.mem0.get_readregister()

        return out

    def model_main(self, inp):
        # apply bitreversal
        rev_index = bit_reversed_indexes(self.FFT_SIZE)
        unrev = inp[:, rev_index]

        # fftshift
        unshift = np.fft.fftshift(unrev, axes=1)

        # avg average in freq axis
        avg_y = np.split(unshift.T, len(unshift.T) // self.AVG_FREQ_AXIS)
        avg_y = np.average(avg_y, axis=1)

        # avg average in time axis
        avg_x = np.split(avg_y.T, len(avg_y.T) // self.AVG_TIME_AXIS)
        avg_x = np.average(avg_x, axis=1)
        return avg_x


@pytest.mark.parametrize("decimation", [2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("fft_size", [2048, 1024, 512, 256, 128])
@pytest.mark.parametrize("packets", [3, 2, 1])
def test_basic(fft_size, decimation, packets):
    orig_inp = np.random.uniform(-1, 1, fft_size * packets)
    orig_inp = [orig_inp] * packets

    rev_index = bit_reversed_indexes(fft_size)
    shift = np.fft.fftshift(orig_inp, axes=1)
    input = shift[:, rev_index]

    dut = BitreversalFFTshiftDecimate(fft_size, decimation, 1)

    # r = []
    # for pack in input:
    #     tmp = []
    #     for x in pack:
    #         tmp.append(Sfix(x, 0, -35))
    #     r.append(tmp)
    with Sfix._float_mode:
        sims = simulate(dut, input, simulations=['MODEL',
                                                 'PYHA',
                                                 # 'RTL',
                                                 # 'GATE'
                                                 ],
                        output_callback=unpackage,
                        input_callback=package,
                        conversion_path='/home/gaspar/git/pyhacores/playground'
                        )

    assert sims_close(sims, rtol=1e-2, atol=1e-5)


@pytest.mark.parametrize("decimation", [2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("fft_size", [2048, 1024, 512, 256, 128])
def test_low_power(fft_size, decimation):
    """ Used to force input back to traditional 0,-17 format .. that was a mistake
    because it has critical precision (everything is 0.0) loss in case of small numbers

    This tests that the block is not losing any precision..
    """

    packets = 1
    orig_inp = np.random.uniform(-1, 1, fft_size * packets) * 0.001
    orig_inp = [float(Sfix(x, 0, -17)) for x in orig_inp]  # Quantize the inputs!
    orig_inp = [orig_inp] * packets

    rev_index = bit_reversed_indexes(fft_size)
    shift = np.fft.fftshift(orig_inp, axes=1)
    input = shift[:, rev_index]

    dut = BitreversalFFTshiftDecimate(fft_size, decimation, 1)

    sims = simulate(dut, input, simulations=['MODEL',
                                             'PYHA',
                                             # 'RTL',
                                             # 'GATE'
                                             ],
                    output_callback=unpackage,
                    input_callback=package,
                    conversion_path='/home/gaspar/git/pyhacores/playground'
                    )

    assert sims_close(sims, rtol=1e-32, atol=1e-32)


def test_wtf():
    """ Used to force input back to traditional 0,-17 format .. that was a mistake
    because it has critical precision (everything is 0.0) loss in case of small numbers

    This tests that the block is not losing any precision..
    """
    fft_size = 4
    decimation = 2
    packets = 1
    orig_inp = np.random.uniform(-1, 1, fft_size * packets) * 0.001
    orig_inp = [0.0001, 0.0002, 0.0003, 0.0004]
    orig_inp = [float(Sfix(x, 0, -17)) for x in orig_inp]  # Quantize the inputs!
    orig_inp = [orig_inp] * packets

    rev_index = bit_reversed_indexes(fft_size)
    shift = np.fft.fftshift(orig_inp, axes=1)
    input = shift[:, rev_index]

    dut = BitreversalFFTshiftDecimate(fft_size, decimation)

    sims = simulate(dut, input, simulations=['MODEL',
                                             'PYHA',
                                             # 'RTL',
                                             # 'GATE'
                                             ],
                    output_callback=unpackage,
                    input_callback=package,
                    conversion_path='/home/gaspar/git/pyhacores/playground'
                    )

    assert sims_close(sims, rtol=1e-32, atol=1e-32)
