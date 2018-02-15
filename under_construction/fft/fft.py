from pyha import Hardware, simulate, sims_close
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


    def model_main(self, al, bl):
        from scipy.fftpack import fft
        i = np.array([al, bl]).T
        return fft(i, n=2).T


class FFT(Hardware):
    def __init__(self):
        self.input_delay = 0
        self.output_delay = 0
        self.butterfly = Butterfly()
        self.state = False

    def dummy(self, a, b):
        return a, b

    def main(self, complex_in):
        self.input_delay = complex_in
        if self.state:
            a, self.output_delay = self.dummy(complex_in, self.input_delay)
            self.state = False
            return a
        else:
            self.state = True
            return self.output_delay



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
