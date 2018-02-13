from pyha import Hardware, simulate, sims_close
import numpy as np


def W(k, N):
    """ e^-j*2*PI*k*n/N, argument k = k * n """
    return np.exp(-1j * (2 * np.pi / N) * k)


class Butterfly(Hardware):
    def main(self, a, b):
        up = a + b
        down = (a - b) * W(0, 2)
        return up, down

    def model_main(self, al, bl):
        from scipy.fftpack import fft
        i = np.array([al, bl]).T
        return fft(i, n=2).T


class FFT(Hardware):
    def __init__(self):
        self.d = [0] * 2
        self.butterfly = Butterfly()

    def main(self, complex_in):
        self.d = [complex_in] + self.d[:-1]


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
