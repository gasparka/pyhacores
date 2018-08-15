import pytest
from pyha import Hardware, simulate, sims_close, Sfix, resize, scalb, Complex
import numpy as np
from pyha.common.ram import RAM

from pyhacores.fft.packager import DataWithIndex, unpackage, package
from pyhacores.fft.util import toggle_bit_reverse


def build_lut(fft_size, freq_axis_decimation):
    """ This LUT fixes the bit-reversal and performs fftshift. It defines th RAM write addresses."""
    orig_inp = np.array(list(range(fft_size)))
    shift = np.fft.fftshift(orig_inp)
    rev = toggle_bit_reverse(shift)
    lut = rev // freq_axis_decimation
    return lut


class BitreversalFFTshiftAVGPool(Hardware):
    """ This core is meant to be used in spectrogram applications.
    It performs bitreversal, fftshift and average pooling in one memory.
    """
    def __init__(self, fft_size, avg_freq_axis, avg_time_axis):
        self._simulation_input_types = DataWithIndex(Sfix(0.0, 0, -35))

        assert not (avg_freq_axis == 1 and avg_freq_axis == 1)
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
        unrev = toggle_bit_reverse(inp)

        # fftshift
        unshift = np.fft.fftshift(unrev, axes=1)

        # average in freq axis
        avg_y = np.split(unshift.T, len(unshift.T) // self.AVG_FREQ_AXIS)
        avg_y = np.average(avg_y, axis=1)

        # average in time axis
        avg_x = np.split(avg_y.T, len(avg_y.T) // self.AVG_TIME_AXIS)
        avg_x = np.average(avg_x, axis=1)
        return avg_x


def test_shit():
    from scipy import signal
    fft_size = 128
    avg_freq_axis = 2
    file = '/home/gaspar/git/pyhacores/data/low_power_ph3.raw'
    from pyhacores.utils import load_iq
    orig_inp = load_iq(file)[2000000:2100000]
    orig_inp -= np.mean(orig_inp)
    # orig_inp = orig_inp[:len(orig_inp)//8]

    _, _, spectro_out = signal.spectrogram(orig_inp, 1, nperseg=fft_size, return_onesided=False, detrend=False,
                                           noverlap=0, window='hann')

    # fftshift
    spectro_out = np.roll(spectro_out, fft_size // 2, axis=0)

    # avg decimation
    x = np.split(spectro_out, len(spectro_out) // avg_freq_axis)
    golden_output = np.average(x, axis=1)


    from pyhacores.fft.util import toggle_bit_reverse

    input_signal = toggle_bit_reverse(spectro_out.T).T
    input_signal = np.fft.fftshift(input_signal)

    dut = BitreversalFFTshiftAVGPool(fft_size, avg_freq_axis, 1)
    sims = simulate(dut, input_signal.T, input_types=[Sfix(0, 0, -35)], simulations=['MODEL', 'PYHA'], output_callback=unpackage, input_callback=package)
    assert sims_close(sims, rtol=1e-2, atol=1e-5)




@pytest.mark.parametrize("avg_freq_axis", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("avg_time_axis", [1, 2, 4, 8])
@pytest.mark.parametrize("fft_size", [512, 256, 128])
def test_all(fft_size, avg_freq_axis, avg_time_axis):
    packets = avg_time_axis * 4
    orig_inp = np.random.uniform(-1, 1, size=(packets, fft_size))
    dut = BitreversalFFTshiftAVGPool(fft_size, avg_freq_axis, avg_time_axis)
    sims = simulate(dut, orig_inp, simulations=['MODEL', 'PYHA'], output_callback=unpackage, input_callback=package)
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

    dut = BitreversalFFTshiftAVGPool(fft_size, avg_freq_axis, avg_time_axis)

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

    dut = BitreversalFFTshiftAVGPool(fft_size, avg_freq_axis, avg_time_axis)

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
