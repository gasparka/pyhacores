import pytest
from pyha import Hardware, simulate, sims_close, Sfix, resize, scalb
import numpy as np
from pyha.common.ram import RAM
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
        self.ACCUMULATION_BITS = int(np.log2(avg_freq_axis * avg_time_axis))
        self.FFT_SIZE = fft_size
        self.LUT = build_lut(fft_size, avg_freq_axis)
        self.DELAY = fft_size + 1

        self.time_axis_counter = self.AVG_TIME_AXIS
        self.state = True
        self.ram = [RAM([Sfix(0.0, 0, -35)] * (fft_size // avg_freq_axis)),
                    RAM([Sfix(0.0, 0, -35)] * (fft_size // avg_freq_axis))]
        self.out_index = 0
        self.out_valid = False

    def work_ram(self, inp, write_ram, read_ram):
        """ Warning: synth tools may not infer ram if stuff is changed here """
        # READ-MODIFY-WRITE
        write_index = self.LUT[inp.index]
        write_index_future = self.LUT[(inp.index + 1) % self.FFT_SIZE]
        read = self.ram[write_ram].delayed_read(write_index_future)
        res = resize(read + scalb(inp.data, -self.ACCUMULATION_BITS), 0, -35)
        self.ram[write_ram].delayed_write(write_index, res)

        # output stage
        self.out_valid = False
        if inp.index < self.FFT_SIZE / self.AVG_FREQ_AXIS and self.time_axis_counter == self.AVG_TIME_AXIS:
            _ = self.ram[read_ram].delayed_read(inp.index)
            self.out_index = inp.index
            self.out_valid = True

            # clear memory
            self.ram[read_ram].delayed_write(inp.index, Sfix(0.0, 0, -35))

    def main(self, inp):
        # Quartus wants this IF to infer RAM...
        if self.state:
            self.work_ram(inp, 0, 1)
            read = self.ram[1].get_readregister()
        else:
            self.work_ram(inp, 1, 0)
            read = self.ram[0].get_readregister()

        if inp.index >= self.FFT_SIZE - 1:
            next_counter = self.time_axis_counter - 1
            if next_counter == 0:
                next_counter = self.AVG_TIME_AXIS
                self.state = not self.state

            self.time_axis_counter = next_counter

        out = DataWithIndex(read, index=self.out_index, valid=self.out_valid)
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


@pytest.mark.parametrize("avg_freq_axis", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("avg_time_axis", [1, 2, 4, 8])
@pytest.mark.parametrize("fft_size", [512, 256, 128])
def test_basic(fft_size, avg_freq_axis, avg_time_axis):
    packets = avg_time_axis * 4
    orig_inp = np.random.uniform(-1, 1, fft_size * packets)
    orig_inp = [orig_inp] * packets

    rev_index = bit_reversed_indexes(fft_size)
    shift = np.fft.fftshift(orig_inp, axes=1)
    input = shift[:, rev_index]

    dut = BitreversalFFTshiftDecimate(fft_size, avg_freq_axis, avg_time_axis)

    sims = simulate(dut, input, simulations=['MODEL',
                                             'PYHA',
                                             'RTL',
                                             # 'GATE'
                                             ],
                    output_callback=unpackage,
                    input_callback=package,
                    # conversion_path='/home/gaspar/git/pyhacores/playground'
                    )

    assert sims_close(sims, rtol=1e-2, atol=1e-5)


def test_synth():
    pytest.skip()
    # fft_size = 1024*8
    # avg_freq_axis = 16
    # avg_time_axis = 4
    # Flow Status	Successful - Fri Jul 27 15:31:58 2018
    # Quartus Prime Version	17.1.0 Build 590 10/25/2017 SJ Lite Edition
    # Revision Name	quartus_project
    # Top-level Entity Name	top
    # Family	Cyclone IV E
    # Device	EP4CE40F23C8
    # Timing Models	Final
    # Total logic elements	338 / 39,600 ( < 1 % )
    # Total registers	174
    # Total pins	122 / 329 ( 37 % )
    # Total virtual pins	0
    # Total memory bits	36,864 / 1,161,216 ( 3 % )
    # Embedded Multiplier 9-bit elements	0 / 232 ( 0 % )
    # Total PLLs	0 / 4 ( 0 % )
    #
    # INFO:sim:Analysis & Synthesis Status : Successful - Fri Jul 27 15:33:58 2018
    # INFO:sim:Quartus Prime Version : 17.1.0 Build 590 10/25/2017 SJ Lite Edition
    # INFO:sim:Revision Name : quartus_project
    # INFO:sim:Top-level Entity Name : top
    # INFO:sim:Family : Cyclone IV E
    # INFO:sim:Total logic elements : 334
    # INFO:sim:    Total combinational functions : 302
    # INFO:sim:    Dedicated logic registers : 174
    # INFO:sim:Total registers : 174
    # INFO:sim:Total pins : 122
    # INFO:sim:Total virtual pins : 0
    # INFO:sim:Total memory bits : 36,864
    # INFO:sim:Embedded Multiplier 9-bit elements : 0
    # INFO:sim:Total PLLs : 0
    # INFO:sim:Running netlist writer.
    # fft_size = 1024*8
    # avg_freq_axis = 16
    # avg_time_axis = 2

    # INFO:sim:Analysis & Synthesis Status : Successful - Fri Jul 27 15:35:12 2018
    # INFO:sim:Quartus Prime Version : 17.1.0 Build 590 10/25/2017 SJ Lite Edition
    # INFO:sim:Revision Name : quartus_project
    # INFO:sim:Top-level Entity Name : top
    # INFO:sim:Family : Cyclone IV E
    # INFO:sim:Total logic elements : 332
    # INFO:sim:    Total combinational functions : 300
    # INFO:sim:    Dedicated logic registers : 170
    # INFO:sim:Total registers : 170
    # INFO:sim:Total pins : 122
    # INFO:sim:Total virtual pins : 0
    # INFO:sim:Total memory bits : 18,432
    # INFO:sim:Embedded Multiplier 9-bit elements : 0
    # INFO:sim:Total PLLs : 0
    # INFO:sim:Running netlist writer.
    # fft_size = 1024*8
    # avg_freq_axis = 32
    # avg_time_axis = 4

    fft_size = 1024 * 8
    avg_freq_axis = 32
    avg_time_axis = 4

    packets = avg_time_axis
    orig_inp = np.random.uniform(-1, 1, fft_size * packets)
    orig_inp = [orig_inp] * packets

    rev_index = bit_reversed_indexes(fft_size)
    shift = np.fft.fftshift(orig_inp, axes=1)
    input = shift[:, rev_index]

    dut = BitreversalFFTshiftDecimate(fft_size, avg_freq_axis, avg_time_axis)

    sims = simulate(dut, input, simulations=['MODEL',
                                             'PYHA',
                                             # 'RTL',
                                             'GATE'
                                             ],
                    output_callback=unpackage,
                    input_callback=package,
                    conversion_path='/home/gaspar/git/pyhacores/playground'
                    )

    assert sims_close(sims, rtol=1e-2, atol=1e-5)


@pytest.mark.parametrize("avg_freq_axis", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("avg_time_axis", [1, 2, 4, 8])
@pytest.mark.parametrize("fft_size", [512, 256, 128])
def test_low_power(fft_size, avg_freq_axis, avg_time_axis):
    """ Used to force input back to traditional 0,-17 format .. that was a mistake
    because it has critical precision (everything is 0.0) loss in case of small numbers

    This tests that the block is not losing any precision..
    """

    packets = avg_time_axis
    orig_inp = np.random.uniform(-1, 1, fft_size * packets) * 0.001
    orig_inp = [float(Sfix(x, 0, -17)) for x in orig_inp]  # Quantize the inputs!
    orig_inp = [orig_inp] * packets

    rev_index = bit_reversed_indexes(fft_size)
    shift = np.fft.fftshift(orig_inp, axes=1)
    input = shift[:, rev_index]

    dut = BitreversalFFTshiftDecimate(fft_size, avg_freq_axis, avg_time_axis)

    sims = simulate(dut, input, simulations=['MODEL',
                                             'PYHA',
                                             # 'RTL',
                                             # 'GATE'
                                             ],
                    output_callback=unpackage,
                    input_callback=package,
                    # conversion_path='/home/gaspar/git/pyhacores/playground'
                    )

    assert sims_close(sims, rtol=1e-32, atol=1e-32)
