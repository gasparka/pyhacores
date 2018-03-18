import timeit
import pytest
from numba import jit

from pyha import Hardware, simulate, sims_close, Complex, resize, Sfix
import numpy as np

from pyha.common.shift_register import ShiftRegister
from pyha.conversion.conversion import get_conversion, get_objects_rednode, Conversion
from pyha.simulation.simulation_interface import get_last_trained_object
from pyha.simulation.vhdl_simulation import VHDLSimulation
from under_construction.fft.packager import DataWithIndex, unpackage, package


def W(k, N):
    """ e^-j*2*PI*k*n/N, argument k = k * n """
    return np.exp(-1j * (2 * np.pi / N) * k)


class StageR2SDF(Hardware):
    def __init__(self, fft_size):
        self.FFT_SIZE = fft_size
        self.FFT_HALF = fft_size // 2

        self.CONTROL_MASK = (self.FFT_HALF - 1)
        self.shr = ShiftRegister([Complex() for _ in range(self.FFT_HALF)])

        # self.TWIDDLES = [Complex(W(i, self.FFT_SIZE), 0, -7, overflow_style='saturate', round_style='round') for i in range(self.FFT_HALF)]

        self.TWIDDLES = [W(i, self.FFT_SIZE) for i in range(self.FFT_HALF)]

    def butterfly(self, in_up, in_down, twiddle):
        up = resize(in_up + in_down, 0, -17) # make 0, -17 default? 

        down_part = resize(in_up - in_down, 0, -17)
        down = resize(down_part * twiddle, 0, -17)
        return up, down

    def main(self, x, control):
        if not (control & self.FFT_HALF):
            self.shr.push_next(x)
            return self.shr.peek()
        else:
            twid = self.TWIDDLES[control & self.CONTROL_MASK]
            up, down = self.butterfly(self.shr.peek(), x, twid)

            if self.FFT_HALF > 4:
                down >>= 1
                up >>= 1

            self.shr.push_next(down)
            return up


class R2SDF(Hardware):
    def __init__(self, fft_size):
        self.FFT_SIZE = fft_size

        self.n_bits = int(np.log2(fft_size))
        self.stages = [StageR2SDF(2 ** (pow + 1)) for pow in reversed(range(self.n_bits))]

        # Note: it is NOT correct to use this gain after the magnitude/abs operation, it has to be applied to complex values
        self.GAIN_CORRECTION = 2 ** (0 if self.n_bits - 3 < 0 else -(self.n_bits - 3))
        self.DELAY = (fft_size - 1) + 1  # +1 is output register

        self.out = DataWithIndex(Complex(0.0, 0, -17), 0)

    def main(self, x):
        # #execute stages
        out = x.data
        for stage in self.stages:
            out = stage.main(out, x.index)

        self.out.data = out
        self.out.index = (x.index + self.DELAY + 1) % self.FFT_SIZE
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


@pytest.mark.parametrize("fft_size", [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1024 * 2, 1024 * 4, 1024 * 8])
def test_fft(fft_size):
    np.random.seed(0)
    dut = R2SDF(fft_size)
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
    fft_size = 1024 * 2 * 2
    np.random.seed(0)
    dut = R2SDF(fft_size)
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
    fft_size = 1024
    np.random.seed(0)
    dut = R2SDF(fft_size)
    inp = np.random.uniform(-1, 1, size=(1, fft_size)) + np.random.uniform(-1, 1, size=(1, fft_size)) * 1j
    inp *= 0.25

    sims = simulate(dut, inp, simulations=[
        # 'MODEL',
        'PYHA',
        # 'RTL',
        # 'GATE'
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
