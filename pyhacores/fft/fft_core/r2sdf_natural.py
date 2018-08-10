import logging
import pickle
import pytest

from data import load_iq
from pyha import Hardware, simulate, sims_close, Complex, resize, scalb, Sfix
import numpy as np
from pyha.common.shift_register import ShiftRegister
from pyhacores.fft.packager import DataWithIndex, unpackage, package
from pyhacores.fft.util import toggle_bit_reverse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fft')


def W(k, N, inverse=False):
    """ e^-j*2*PI*k*n/N, argument k = k * n """
    r = np.exp(-1j * (2 * np.pi / N) * k)
    if inverse:
        return np.conjugate(r)
    return r


def test_shit2222():
    # INFO:sim:Analysis & Synthesis Status : Successful - Fri Aug 10 15:10:42 2018
    # INFO:sim:Quartus Prime Version : 17.1.0 Build 590 10/25/2017 SJ Lite Edition
    # INFO:sim:Revision Name : quartus_project
    # INFO:sim:Top-level Entity Name : top
    # INFO:sim:Family : Cyclone IV E
    # INFO:sim:Total logic elements : 2,726
    # INFO:sim:    Total combinational functions : 2,472
    # INFO:sim:    Dedicated logic registers : 1,519
    # INFO:sim:Total registers : 1519
    # INFO:sim:Total pins : 140
    # INFO:sim:Total virtual pins : 0
    # INFO:sim:Total memory bits : 8,718
    # INFO:sim:Embedded Multiplier 9-bit elements : 56
    # INFO:sim:Total PLLs : 0
    # INFO:sim:Running netlist writer.

    # INFO:sim:Analysis & Synthesis Status : Successful - Fri Aug 10 15:35:02 2018
    # INFO:sim:Quartus Prime Version : 17.1.0 Build 590 10/25/2017 SJ Lite Edition
    # INFO:sim:Revision Name : quartus_project
    # INFO:sim:Top-level Entity Name : top
    # INFO:sim:Family : Cyclone IV E
    # INFO:sim:Total logic elements : 2,598
    # INFO:sim:    Total combinational functions : 2,525
    # INFO:sim:    Dedicated logic registers : 1,347
    # INFO:sim:Total registers : 1347
    # INFO:sim:Total pins : 140
    # INFO:sim:Total virtual pins : 0
    # INFO:sim:Total memory bits : 8,712
    # INFO:sim:Embedded Multiplier 9-bit elements : 56
    # INFO:sim:Total PLLs : 0
    # INFO:sim:Running netlist writer.

    # INFO:sim:Analysis & Synthesis Status : Successful - Fri Aug 10 15:43:23 2018
    # INFO:sim:Quartus Prime Version : 17.1.0 Build 590 10/25/2017 SJ Lite Edition
    # INFO:sim:Revision Name : quartus_project
    # INFO:sim:Top-level Entity Name : top
    # INFO:sim:Family : Cyclone IV E
    # INFO:sim:Total logic elements : 2,621
    # INFO:sim:    Total combinational functions : 2,533
    # INFO:sim:    Dedicated logic registers : 1,408
    # INFO:sim:Total registers : 1408
    # INFO:sim:Total pins : 140
    # INFO:sim:Total virtual pins : 0
    # INFO:sim:Total memory bits : 8,712
    # INFO:sim:Embedded Multiplier 9-bit elements : 56
    # INFO:sim:Total PLLs : 0
    # INFO:sim:Running netlist writer.

    np.random.seed(0)
    fft_size = 256
    input_signal = np.random.uniform(-1, 1, fft_size) + np.random.uniform(-1, 1, fft_size) * 1j
    input_signal *= 0.125

    dut = R2SDFNATURAL(fft_size, twiddle_bits=18)
    rev_sims = simulate(dut, input_signal, input_callback=package, output_callback=unpackage,
                        simulations=['MODEL', 'PYHA', 'GATE'])
    assert sims_close(rev_sims, rtol=1e-3)


def test_shit2():
    fft_size = 8
    input_signal = np.array(
        [0.01 + 0.01j, 0.02 + 0.02j, 0.03 + 0.03j, 0.04 + 0.04j, 0.05 + 0.05j, 0.06 + 0.06j, 0.07 + 0.07j,
         0.08 + 0.08j])

    dut = R2SDFNATURAL(fft_size, twiddle_bits=18)
    rev_sims = simulate(dut, input_signal, input_callback=package,
                        output_callback=unpackage,
                        simulations=['MODEL', 'PYHA'])
    assert sims_close(rev_sims)


def test_shit16():
    fft_size = 16
    input_signal = np.random.uniform(-1, 1, fft_size) + np.random.uniform(-1, 1, fft_size) * 1j
    input_signal *= 0.125

    dut = R2SDFNATURAL(fft_size, twiddle_bits=18)
    rev_sims = simulate(dut, input_signal, input_callback=package, output_callback=unpackage,
                        simulations=['MODEL', 'PYHA'])
    assert sims_close(rev_sims)


class StageR2SDFNATURAL(Hardware):
    def __init__(self, fft_size, global_fft_size, twiddle_bits=18):
        self.GLOBAL_FFT_SIZE = global_fft_size
        self.FFT_SIZE = fft_size
        self.FFT_HALF = fft_size // 2

        self.CONTROL_MASK = (self.FFT_HALF - 1)
        self.shr = ShiftRegister([Complex() for _ in range(self.FFT_HALF)])

        self.TWIDDLES = [
            Complex(W(i, self.FFT_SIZE), 0, -(twiddle_bits - 1), overflow_style='saturate', round_style='round') for i
            in range(self.FFT_HALF)]

        self.DELAY = 3 + self.FFT_HALF  # 3 comes from stage registers

        self.twiddle = self.TWIDDLES[0]
        self.stage1_out = Complex(0, 0, -17)
        self.stage2_out = Complex(0, 0, -17 - (twiddle_bits - 1))
        self.stage3_out = Complex(0, 0, -17, round_style='round')
        self.output_index = 0
        self.mode_delay = False

    def butterfly(self, in_up, in_down):
        up = resize(in_up + in_down, 0, -17)
        down = resize(in_up - in_down, 0, -17)
        return up, down

    def main(self, x, control):
        # Stage 1: handle the loopback memory, to calculate butterfly additions.
        # Also fetch the twiddle factor.
        self.twiddle = self.TWIDDLES[control & self.CONTROL_MASK]

        mode = not (control & self.FFT_HALF)
        self.mode_delay = mode
        if mode:
            self.shr.push_next(x)
            self.stage1_out = self.shr.peek()
        else:
            up, down = self.butterfly(self.shr.peek(), x)
            self.shr.push_next(down)
            self.stage1_out = up

        # Stage 2: complex multiply, only the botton line
        if self.mode_delay and self.FFT_HALF != 1:
            self.stage2_out = self.stage1_out * self.twiddle
        else:
            self.stage2_out = self.stage1_out

        # Stage 3: gain control and rounding
        if self.FFT_HALF > 4:
            self.stage3_out = scalb(self.stage2_out, -1)
        else:
            self.stage3_out = self.stage2_out

        # delay index by same amount as data
        self.output_index = (control - (self.DELAY - 1)) % self.GLOBAL_FFT_SIZE
        return self.stage3_out, self.output_index


class R2SDFNATURAL(Hardware):
    def __init__(self, fft_size, twiddle_bits=18):
        self.FFT_SIZE = fft_size

        self.N_STAGES = int(np.log2(fft_size))
        self.stages = [StageR2SDFNATURAL(2 ** (pow + 1), fft_size, twiddle_bits) for pow in
                       reversed(range(self.N_STAGES))]

        # Note: it is NOT correct to use this gain after the magnitude/abs operation, it has to be applied to complex values
        self.GAIN_CORRECTION = 2 ** (0 if self.N_STAGES - 3 < 0 else -(self.N_STAGES - 3))
        self.DELAY = (fft_size - 1) + (self.N_STAGES * 3) + 1

        self.out = DataWithIndex(Complex(0.0, 0, -17, round_style='round'), 0)

    def main(self, x):
        # execute stages
        out = x.data
        out_index = x.index
        for stage in self.stages:
            out, out_index = stage.main(out, out_index)

        self.out.data = out
        self.out.index = out_index
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
    fft_size = 4
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


@pytest.mark.parametrize("fft_size", [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
def test_fft(fft_size):
    np.random.seed(0)
    dut = R2SDF(fft_size, twiddle_bits=14)
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
    assert sims_close(sims, rtol=1e-3, atol=1e-4)


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

    # PIPELINED
    # 10 bit
    # Fmax = 95M
    # Family	Cyclone IV E
    # Total logic elements	9,845
    # Total registers	2013
    # Total pins	140
    # Total virtual pins	0
    # Total memory bits	294,048
    # Embedded Multiplier 9-bit elements	96
    # Total PLLs	0

    fft_size = 1024 * 2 * 2 * 2
    # fft_size = 8
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
