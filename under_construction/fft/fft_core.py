import timeit

from pyha.common.stream import Stream
from scipy import signal

import pytest
from data import load_iq
from pyha import Hardware, simulate, sims_close, Complex, resize, Sfix
import numpy as np

from under_construction.fft.packager import DataWithIndex, Packager


def W(k, N):
    """ e^-j*2*PI*k*n/N, argument k = k * n """
    return np.exp(-1j * (2 * np.pi / N) * k)


class StageR2SDF(Hardware):
    def __init__(self, fft_size):
        self.FFT_SIZE = fft_size
        self.FFT_HALF = fft_size // 2

        self.CONTROL_MASK = (self.FFT_HALF - 1)
        self.shr = [Complex() for _ in range(self.FFT_HALF)]

        self.TWIDDLES = [Complex(W(i, self.FFT_SIZE), 0, -8, overflow_style='saturate', round_style='round') for i in range(self.FFT_HALF)]

        self.twiddle_buffer = Complex()

    def butterfly(self, in_up, in_down, twiddle):
        up_real = resize(in_up.real + in_down.real, 0, -17)
        up_imag = resize(in_up.imag + in_down.imag, 0, -17)
        up = Complex(up_real, up_imag)

        # down sub
        down_sub_real = resize(in_up.real - in_down.real, 0, -17)
        down_sub_imag = resize(in_up.imag - in_down.imag, 0, -17)

        down_real = resize((down_sub_real * twiddle.real) - (down_sub_imag * twiddle.imag), 0, -17)
        down_imag = resize((down_sub_real * twiddle.imag) + (down_sub_imag * twiddle.real), 0, -17)
        down = Complex(down_real, down_imag)
        return up, down

    def main(self, x, control):
        self.twiddle_buffer = self.TWIDDLES[(control + 1) & self.CONTROL_MASK]
        up, down = self.butterfly(self.shr[-1], x, self.twiddle_buffer)

        if self.FFT_HALF > 4:
            down.real = down.real >> 1
            down.imag = down.imag >> 1
            up.real = up.real >> 1
            up.imag = up.imag >> 1

        if not (control & self.FFT_HALF):
            self.shr = [x] + self.shr[:-1]
            return self.shr[-1]
        else:
            self.shr = [down] + self.shr[:-1]
            return up


class R2SDF(Hardware):
    def __init__(self, fft_size):
        self.FFT_SIZE = fft_size

        self.n_bits = int(np.log2(fft_size))
        self.stages = [StageR2SDF(2 ** (pow + 1)) for pow in reversed(range(self.n_bits))]
        #
        # self.stage4000 = StageR2SDF(1024*4)
        # self.stage2000 = StageR2SDF(1024*2)
        # self.stage1024 = StageR2SDF(1024)
        # self.stage512 = StageR2SDF(512)
        # self.stage256 = StageR2SDF(256)
        # self.stage128 = StageR2SDF(128)
        # self.stage64 = StageR2SDF(64)
        # self.stage32 = StageR2SDF(32)
        # self.stage16 = StageR2SDF(16)
        # self.stage8 = StageR2SDF(8)
        # self.stage4 = StageR2SDF(4)
        # self.stage2 = StageR2SDF(2)

        # Note: it is NOT correct to use this gain after the magnitude/abs operation, it has to be applied to complex values
        self.GAIN_CORRECTION = 2 ** (0 if self.n_bits - 3 < 0 else -(self.n_bits - 3))
        self.DELAY = (fft_size - 1) + 1  # +1 is output register

        self.out = DataWithIndex(Complex(0.0, 0, -17), 0)

    def main(self, x):
        # #execute stages
        out = x.data
        # out = self.stage4000.main(out, x.index)
        # out = self.stage2000.main(out, x.index)
        # out = self.stage1024.main(out, x.index)
        # out = self.stage512.main(out, x.index)
        # out = self.stage256.main(out, x.index)
        # out = self.stage128.main(out, x.index)
        # out = self.stage64.main(out, x.index)
        # out = self.stage32.main(out, x.index)
        # out = self.stage16.main(out, x.index)
        # out = self.stage8.main(out, x.index)
        # out = self.stage4.main(out, x.index)
        # out = self.stage2.main(out, x.index)

        # new_index = (x.index + self.DELAY + 1) # BUGG

        out = x.data
        for stage in self.stages:
            out = stage.main(out, x.index)

        new_index = (x.index + self.DELAY + 1) % self.FFT_SIZE
        self.out.data = out
        self.out.index = new_index
        self.out.valid = x.valid
        return self.out

    def model_main(self, x):
        ffts = np.fft.fft(x)

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


@pytest.mark.parametrize("fft_size", [2, 4, 8, 16, 32, 64, 128, 256])
def test_fft(fft_size):
    np.random.seed(0)
    dut = R2SDF(fft_size)
    inp = np.random.uniform(-1, 1, size=(2, fft_size)) + np.random.uniform(-1, 1, size=(2, fft_size)) * 1j
    inp *= 0.25

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA',
                                           'RTL'
                                           # 'GATE'
                                           ],
                    conversion_path='/home/gaspar/git/pyhacores/playground',
                    output_callback=DataWithIndex._pyha_unpack,
                    input_callback=DataWithIndex._pyha_pack)
    assert sims_close(sims, rtol=1e-1, atol=1e-4)



def test_simple():
    fft_size = 2
    np.random.seed(0)
    dut = R2SDF(fft_size)

    inp = np.array([0+0.123j, 1+0.123j]).astype(complex) * 0.1
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
    print(timeit.timeit('import pyha; pyha.Complex()'))
    print(timeit.timeit('from pyha.simulation.simulation_interface import simulate'))

    # fft_size = 256
    # inp = np.random.uniform(-1, 1, fft_size) + np.random.uniform(-1, 1, fft_size) * 1j
    #
    # dut = R2SDF(fft_size)
    # sims = simulate(dut, inp, simulations=['PYHA'])
    # # assert sims_close(sims, rtol=1e-2)
    #
    # #python -m plop.collector -f flamegraph fft_core.py
    # # ./git/FlameGraph/flamegraph.pl --width 5000 x.flame > flame.svg
