import logging
import pickle
import pytest

from data import load_iq
from pyha import Hardware, simulate, sims_close, Complex, resize, scalb
import numpy as np
from pyha.common.shift_register import ShiftRegister
from under_construction.fft.packager import DataWithIndex, unpackage, package

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fft')


def W(k, N):
    """ e^-j*2*PI*k*n/N, argument k = k * n """
    return np.exp(-1j * (2 * np.pi / N) * k)


class StageR2SDF(Hardware):
    def __init__(self, fft_size, twiddle_bits=18):
        self.FFT_SIZE = fft_size
        self.FFT_HALF = fft_size // 2

        self.CONTROL_MASK = (self.FFT_HALF - 1)
        self.shr = ShiftRegister([Complex() for _ in range(self.FFT_HALF)])

        self.TWIDDLES = [
            Complex(W(i, self.FFT_SIZE), 0, -(twiddle_bits - 1), overflow_style='saturate', round_style='round') for i
            in range(self.FFT_HALF)]

        self.out = Complex()

    def butterfly(self, in_up, in_down, twiddle):
        if self.FFT_HALF > 4:
            up = resize(scalb(in_up + in_down, -1), 0, -17, round_style='round')
            down_part = resize(in_up - in_down, 0, -17)
            down = resize(scalb(down_part * twiddle, -1), 0, -17, round_style='round')
        else:
            up = resize(in_up + in_down, 0, -17)
            down_part = resize(in_up - in_down, 0, -17)
            down = resize(down_part * twiddle, 0, -17, round_style='round')
        return up, down

        # TODO: Bug..negative integer index?
        # up = in_up + in_down
        # down_part = resize(in_up - in_down, 0, -17)
        # down = down_part * twiddle
        #
        # if self.FFT_HALF > 4:
        #     up = scalb(up, -1)
        #     down = scalb(down, -1)
        #
        # up = resize(up, 0, -17, round_style='round')
        # down = resize(down, 0, -17, round_style='round')
        # return up, down

    def main(self, x, control):

        # logger.info(f'{self.FFT_SIZE} - {control}')
        if not (control & self.FFT_HALF):
            self.shr.push_next(x)
            self.out = self.shr.peek()
        else:
            twid = self.TWIDDLES[control & self.CONTROL_MASK]
            up, down = self.butterfly(self.shr.peek(), x, twid)
            self.shr.push_next(down)
            self.out = up

        return self.out


class R2SDF(Hardware):
    def __init__(self, fft_size, twiddle_bits=9):
        self.FFT_SIZE = fft_size

        self.N_STAGES = int(np.log2(fft_size))
        self.stages = [StageR2SDF(2 ** (pow + 1), twiddle_bits) for pow in reversed(range(self.N_STAGES))]

        # Note: it is NOT correct to use this gain after the magnitude/abs operation, it has to be applied to complex values
        self.GAIN_CORRECTION = 2 ** (0 if self.N_STAGES - 3 < 0 else -(self.N_STAGES - 3))
        self.DELAY = (fft_size - 1) + 1 + self.N_STAGES  # +1 is output register

        self.out = DataWithIndex(Complex(0.0, 0, -17, round_style='round'), 0)

    def main(self, x):
        # execute stages
        out = x.data
        for i in range(len(self.stages)):
            index = (x.index - i) % self.FFT_SIZE
            out = self.stages[i].main(out, index)
        # for stage in self.stages:
        #     out = stage.main(out, x.index)

        self.out.data = out
        self.out.index = (x.index - (self.N_STAGES - 1)) % self.FFT_SIZE
        self.out.valid = x.valid
        return self.out

    def model_main(self, x):
        x = x.reshape(-1, self.FFT_SIZE)
        ffts = np.fft.fft(x, self.FFT_SIZE)

        # apply bit reversing ie. mess up the output order to match radix-2 algorithm
        # from under_construction.fft.bit_reversal import bit_reversed_indexes
        def bit_reverse(x, n_bits):
            return int(np.binary_repr(x, n_bits)[::-1], 2)

        def bit_reversed_indexes(N):
            return [bit_reverse(i, int(np.log2(N))) for i in range(N)]

        rev_index = bit_reversed_indexes(self.FFT_SIZE)
        for i, _ in enumerate(ffts):
            ffts[i] = ffts[i][rev_index]

        # apply gain control (to avoid overflows in hardware)
        ffts *= self.GAIN_CORRECTION

        return ffts


def test_pipeline():
    fft_size = 32
    dut = R2SDF(fft_size, twiddle_bits=18)

    inp = np.random.uniform(-1, 1, size=(2, fft_size)) + np.random.uniform(-1, 1, size=(2, fft_size)) * 1j
    inp *= 0.125
    # inp = np.array([0.1 + 0.1j, 0.2 + 0.2j, 0.3 + 0.3j, 0.4 + 0.4j] * 4) * 0.1

    sims = simulate(dut, inp, simulations=[
        'MODEL',
        'PYHA',
        # 'RTL',
        # 'GATE'
    ],
                    output_callback=unpackage,
                    input_callback=package)

    assert sims_close(sims, rtol=1e-1, atol=1e-4)


@pytest.mark.parametrize("fft_size", [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
def test_fft(fft_size):
    np.random.seed(0)
    dut = R2SDF(fft_size, twiddle_bits=12)
    inp = np.random.uniform(-1, 1, size=(2, fft_size)) + np.random.uniform(-1, 1, size=(2, fft_size)) * 1j
    inp *= 0.25

    sims = simulate(dut, inp, simulations=[
        'MODEL',
        'PYHA',
        # 'RTL',
        # 'GATE'
    ],
                    conversion_path='/home/gaspar/git/pyhacores/playground',
                    output_callback=unpackage,
                    input_callback=package)
    assert sims_close(sims, rtol=1e-1, atol=1e-4)


def test_synth():
    # 9 bit (MAX)
    # MAX clk: 22MHz
    # Max Sine peak: ~56 dB
    # Total logic elements	8,677 / 15,840 ( 55 % )
    # Total registers	341
    # Total memory bits	293,976 / 562,176 ( 52 % )
    # Embedded Multiplier 9-bit elements	90 / 90 ( 100 % )

    # 8 bit
    # Total logic elements	6,424 / 39,600 ( 16 % )

    # 7 bit
    # Total logic elements	5,442

    # 11 bit
    # Total logic elements	12,905 / 39,600 ( 33 % )
    # FMAX: 16M ( 58M with 2 up/down regs only, retime +0, 52.82 MHz (out reg only) )
    # OLD NO ROUND:
    # FMAX : 23M (simple registers, 125M)
    # Device	EP4CE40F23C8
    # Timing Models	Final
    # Total logic elements	11,480 / 39,600 ( 29 % )
    # Total registers	341
    # Total pins	140 / 329 ( 43 % )
    # Total virtual pins	0
    # Total memory bits	293,976 / 1,161,216 ( 25 % )
    # Embedded Multiplier 9-bit elements	96 / 232 ( 41 % )

    # 12 bit
    # INFO:sim:Total logic elements : 13,079

    # 9 bit
    # INFO:sim:Analysis & Synthesis Status : Successful - Tue Jun 19 10:24:46 2018
    # INFO:sim:Quartus Prime Version : 17.1.0 Build 590 10/25/2017 SJ Lite Edition
    # INFO:sim:Revision Name : quartus_project
    # INFO:sim:Top-level Entity Name : top
    # INFO:sim:Family : Cyclone IV E
    # INFO:sim:Total logic elements : 9,112
    # INFO:sim:    Total combinational functions : 9,038
    # INFO:sim:    Dedicated logic registers : 790
    # INFO:sim:Total registers : 790
    # INFO:sim:Total pins : 140
    # INFO:sim:Total virtual pins : 0
    # INFO:sim:Total memory bits : 293,976
    # INFO:sim:Embedded Multiplier 9-bit elements : 96
    # INFO:sim:Total PLLs : 0
    # INFO:sim:Running netlist writer.

    # 10 bit
    # INFO:sim:Analysis & Synthesis Status : Successful - Tue Jun 19 10:43:00 2018
    # INFO:sim:Quartus Prime Version : 17.1.0 Build 590 10/25/2017 SJ Lite Edition
    # INFO:sim:Revision Name : quartus_project
    # INFO:sim:Top-level Entity Name : top
    # INFO:sim:Family : Cyclone IV E
    # INFO:sim:Total logic elements : 10,685
    # INFO:sim:    Total combinational functions : 10,611
    # INFO:sim:    Dedicated logic registers : 790
    # INFO:sim:Total registers : 790
    # INFO:sim:Total pins : 140
    # INFO:sim:Total virtual pins : 0
    # INFO:sim:Total memory bits : 293,976
    # INFO:sim:Embedded Multiplier 9-bit elements : 96
    # INFO:sim:Total PLLs : 0
    # INFO:sim:Running netlist writer.

    fft_size = 1024 * 2 * 2 * 2
    np.random.seed(0)
    dut = R2SDF(fft_size, twiddle_bits=10)
    inp = np.random.uniform(-1, 1, size=(1, fft_size)) + np.random.uniform(-1, 1, size=(1, fft_size)) * 1j
    inp *= 0.25

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA',
                                           # 'RTL',
                                           'GATE'
                                           ],
                    conversion_path='/home/gaspar/git/pyhacores/playground',
                    output_callback=unpackage,
                    input_callback=package)
    assert sims_close(sims, rtol=1e-1, atol=1e-4)


# @pytest.mark.parametrize("file", glob.glob('/run/media/gaspar/maxtor/measurement 13.03.2018/mavic_tele/qdetector_20180313122024455464_far_10m_regular/**/*.raw', recursive=True))
# def test_realsig(file):
def test_realsig():
    file = '/run/media/gaspar/maxtor/measurement 13.03.2018/mavic_tele/qdetector_20180313122024455464_far_10m_regular/1520936452.2426_fs=20000000.0_bw=20000000.0_fc=2431000000.0_d=0_g=033000.raw'
    fft_size = 1024 * 2 * 2 * 2

    iq = load_iq(file)
    sig = iq.reshape(-1, fft_size)
    sig *= np.hanning(fft_size)
    sig = sig.flatten()

    dut = R2SDF(fft_size)
    sims = simulate(dut, sig, simulations=['MODEL', 'PYHA'],
                    output_callback=unpackage,
                    input_callback=package)

    with open(f'{file}.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(sims, f, pickle.HIGHEST_PROTOCOL)


def test_simple():
    fft_size = 2
    np.random.seed(0)
    dut = R2SDF(fft_size)

    inp = np.array([0 + 0.123j, 1 + 0.123j]).astype(complex) * 0.1
    inp = [inp, inp]
    # inp = np.random.uniform(-1, 1, size=(2, fft_size)) + np.random.uniform(-1, 1, size=(2, fft_size)) * 1j
    # inp *= 0.1

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA',
                                           # 'GATE'
                                           ],
                    # conversion_path='/home/gaspar/git/pyhacores/playground',
                    output_callback=DataWithIndex._pyha_unpack,
                    input_callback=DataWithIndex._pyha_pack)
    assert sims_close(sims, rtol=1e-1, atol=1e-4)


# import pyha.simulation.simulation_interface import simulate
if __name__ == '__main__':
    fft_size = 1024 * 2 * 2 * 2
    np.random.seed(0)
    dut = R2SDF(fft_size)
    inp = np.random.uniform(-1, 1, size=(1, fft_size)) + np.random.uniform(-1, 1, size=(1, fft_size)) * 1j
    inp *= 0.25

    sims = simulate(dut, inp, simulations=[
        # 'MODEL',
        'PYHA',
        # 'RTL',
        'GATE'
    ],
                    conversion_path='/home/gaspar/git/pyhacores/playground',
                    output_callback=unpackage,
                    input_callback=package)

    # mod = get_last_trained_object()
    # Conversion(mod)
    # vhdl_sim = VHDLSimulation(Path(conversion_path), fix_model, 'RTL')
    # assert sims_close(sims, rtol=1e-1, atol=1e-4)

    # print(timeit.timeit('import pyha; pyha.Complex()'))
    # print(timeit.timeit('from pyha.simulation.simulation_interface import simulate'))
    #
    # # fft_size = 256
    # # inp = np.random.uniform(-1, 1, fft_size) + np.random.uniform(-1, 1, fft_size) * 1j
    # #
    # # dut = R2SDF(fft_size)
    # # sims = simulate(dut, inp, simulations=['PYHA'])
    # # # assert sims_close(sims, rtol=1e-2)
    # #
    # # #python -m plop.collector -f flamegraph fft_core.py
    # # # ./git/FlameGraph/flamegraph.pl --width 5000 x.flame > flame.svg
