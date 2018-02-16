import pytest
from pyha import Hardware, simulate, sims_close, Complex
import numpy as np


# todo:
# * One input instead of N
# * Each butterfly stage needs 2 inputs
# * Input order needs to be adjusted
# * Butterfly outputs need to be *corrected* (hardest?)
# * Butterfly needs to loop between coefficents
# * Output should be single item not dual, as is from last butterfly
# * Output ordering is wrong (radix-2 problem)

def W(k, N):
    """ e^-j*2*PI*k*n/N, argument k = k * n """
    return np.exp(-1j * (2 * np.pi / N) * k)


class Butterfly(Hardware):

    def __init__(self, fft_size):
        self.fft_size = fft_size
        self.input_delay = [0] * (fft_size//2)

    def butterfly(self, a, b):
        up = a + b
        down = (a - b) * W(0, 2)
        return up, down

    def main(self, a, b):
        pass

    def model_main(self, al, bl):
        from scipy.fftpack import fft
        i = np.array([al, bl]).T
        return fft(i, n=2).T


class FFT(Hardware):
    def __init__(self, fft_size):
        self.fft_size = fft_size
        self.input_delay = [Complex()] * 1
        self.output_delay = 0
        self.state = True
        self.DELAY = 1

    def transform_input(self, complex_in):
        self.input_delay = [complex_in] + self.input_delay[:-1]
        return self.input_delay[-1], complex_in

    def butterfly(self, a, b):
        up = a + b
        down = (a - b) * W(0, 2)
        return up, down

    def dummy(self, a, b):
        return a, b

    def main(self, complex_in):
        a, b = self.transform_input(complex_in)
        if self.state:
            output, self.output_delay = self.butterfly(a, b)
        else:
            output = self.output_delay

        self.state = not self.state
        return output

    def model_main(self, x):
        from scipy.fftpack import fft
        return fft(x)


@pytest.mark.parametrize("fft_size", [2, 4, 8, 16])
def test_fft(fft_size):
    dut = FFT(fft_size)

    inp = np.random.uniform(-1, 1, fft_size) + np.random.uniform(-1, 1, fft_size)*1j
    # inp = [0.1 + 0.2j, 0.3 + 0.4j]

    # inp = [0.1, 0.2]

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA'])
    assert sims_close(sims)


class TestFFT:
    def test_basic(self):
        dut = FFT()
        a = list(range(16))[:1]

        sims = simulate(dut, a, simulations=['MODEL', 'PYHA'])
        assert sims_close(sims)

class TestButterfly:
    def test_simple(self):
        dut = Butterfly()
        a = 0.1 + 0.2j
        b = 0.2 + 0.3j

        sims = simulate(dut, a, b, simulations=['MODEL', 'MODEL_SIM'])
        assert sims_close(sims)

    def test_simple_list(self):
        dut = Butterfly()
        a = [0.1 + 0.2j] * 2
        b = [0.2 + 0.3j] * 2

        sims = simulate(dut, a, b, simulations=['MODEL', 'MODEL_SIM'])
        assert sims_close(sims)
