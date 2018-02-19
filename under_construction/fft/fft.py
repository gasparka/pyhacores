from copy import deepcopy

import pytest
from pyha import Hardware, simulate, sims_close, Complex
import numpy as np


class StageR2SDF(Hardware):
    def __init__(self, fft_size):
        self.fft_size = fft_size
        self.fft_half = fft_size // 2

        self.control_mask = (self.fft_half - 1)
        self.mem = [Complex() for _ in range(self.fft_half)]

    def main(self, x, control):
        if not (control & self.fft_half):
            self.mem = [x] + self.mem[:-1]
            return self.mem[-1]
        else:
            up = self.mem[-1] + x
            down = (self.mem[-1] - x) * W(control & self.control_mask, self.fft_size)
            self.mem = [down] + self.mem[:-1]
            return up

def bit_reverse(x, n_bits):
    return int(np.binary_repr(x, n_bits)[::-1], 2)

class R2SDF(Hardware):
    def __init__(self, fft_size):
        self.fft_size = fft_size

        self.n_bits = int(np.log2(fft_size))
        self.stages = [StageR2SDF(2 ** (pow+1)) for pow in reversed(range(self.n_bits))]
        self.control = 0

        self.correct_output = [Complex() for _ in range(fft_size)]
        self.cod = [0] * (fft_size//2+1)

        self.DELAY = (fft_size - 1)  + (5)

    def main(self, x):
        next_control = (self.control + 1) % self.fft_size
        self.control = next_control

        tmp = x
        for stage in self.stages:
            tmp = stage.main(tmp, self.control)


        out = tmp
        # print(next_control, out)
        reversed_index = bit_reverse(next_control, self.n_bits)

        # BUGGGG -> test dual port memory!
        self.correct_output[reversed_index] = deepcopy(tmp)
        print(reversed_index, self.cod[-1])

        self.cod = [next_control] + self.cod[:-1]
        ro = self.cod[-1]
        out = self.correct_output[self.cod[-1]]

        return out

    def model_main(self, x):
        from scipy.fftpack import fft
        x = np.array(x).reshape((-1, self.fft_size))
        return np.hstack(fft(x))


def test_fft8():
    fft_size = 8
    dut = R2SDF(fft_size)

    inp = np.random.uniform(-1, 1, 32) + np.random.uniform(-1, 1, 32) * 1j
    inp = [0.1 + 0.2j, 0.3 + 0.4j, 0.1 + 0.2j, 0.3 + 0.4j, 0.1 + 0.2j, 0.3 + 0.4j, 0.1 + 0.2j, 0.3 + 0.4j]

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA'])

    import matplotlib.pyplot as plt
    plt.plot(sims['MODEL'])
    plt.plot(sims['PYHA'])
    plt.show()
    assert sims_close(sims)


def test_fft4():
    fft_size = 4
    dut = R2SDF(fft_size)

    # inp = np.random.uniform(-1, 1, fft_size) + np.random.uniform(-1, 1, fft_size)*1j
    inp = [0.1 + 0.2j, 0.3 + 0.4j, 0.1 + 0.2j, 0.3 + 0.4j]
    # inp = list(range(fft_size))

    # inp = [0.1, 0.2]

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA'])
    assert sims_close(sims)


def test_fft2():
    fft_size = 2
    dut = R2SDF(fft_size)

    # inp = np.random.uniform(-1, 1, fft_size) + np.random.uniform(-1, 1, fft_size)*1j
    inp = [0.1 + 0.2j, 0.3 + 0.4j]
    # inp = list(range(fft_size))

    # inp = [0.1, 0.2]

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA'])
    assert sims_close(sims)


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
        self.twiddles = [W(i, fft_size) for i in range(fft_size // 2)]
        self.twiddle_index = 0

    def main(self, a, b):
        up = a + b
        down = (a - b) * W(0, 2)
        return up, down

    def model_main(self, al, bl):
        from scipy.fftpack import fft
        i = np.array([al, bl]).T
        return fft(i, n=2).T


class FFT(Hardware):
    def __init__(self, fft_size):
        self.fft_size = fft_size
        self.input_delay = [0] * (fft_size // 2)
        self.output_delay = 0
        self.state = True
        self.DELAY = fft_size // 2

        self.butterfly1 = Butterfly(2)
        self.butterfly2 = Butterfly(4)

    def transform_input(self, complex_in):
        self.input_delay = [complex_in] + self.input_delay[:-1]
        return self.input_delay[-1], complex_in

    def main(self, complex_in):
        a, b = self.transform_input(complex_in)
        if self.state:
            output, self.output_delay = self.butterfly1.main(a, b)
        else:
            output = self.output_delay

        self.state = not self.state
        return output

    def model_main(self, x):
        from scipy.fftpack import fft
        return fft(x)


class Stream(Hardware):
    def __init__(self, data, valid):
        self.data = data
        self.valid = valid

    def _pyha_on_simulation_output(self):
        if self.valid:
            return self.data


class InputStage(Hardware):
    def __init__(self, fft_size):
        self.fft_size = fft_size
        self.a_delayed = [-2] * (fft_size // 2)
        self.state = 0
        self.DELAY = fft_size // 2

    def switcher(self, x):
        self.state = (self.state + 1) % self.fft_size
        if self.state < (self.fft_size // 2):
            a = x
            b = -1
            valid = False
        else:
            a = -1
            b = x
            valid = True
        return a, b, valid

    def main(self, x):
        a, b, valid = self.switcher(x)

        self.a_delayed = [a] + self.a_delayed[:-1]
        out = Stream([self.a_delayed[-1], b], valid)
        return out


def test_input4():
    dut = InputStage(4)

    inp = [0, 1, 2, 3, 0, 1, 2, 3]
    expect = [[0, 2], [1, 3], [0, 2], [1, 3]]

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA'])
    assert sims_close(sims, expect)


# def test_fft4():
#     fft_size = 4
#     dut = FFT(4)
#
#     # inp = np.random.uniform(-1, 1, fft_size) + np.random.uniform(-1, 1, fft_size)*1j
#     inp = [0.1 + 0.2j, 0.3 + 0.4j]
#     inp = [0, 1, 2, 3, 0, 1, 2, 3]
#
#     sims = simulate(dut, inp, simulations=['MODEL', 'PYHA'])
#     assert sims_close(sims)


@pytest.mark.parametrize("fft_size", [2, 4, 8, 16])
def test_fft(fft_size):
    dut = FFT(fft_size)

    # inp = np.random.uniform(-1, 1, fft_size) + np.random.uniform(-1, 1, fft_size)*1j
    inp = [0.1 + 0.2j, 0.3 + 0.4j]
    inp = list(range(fft_size))

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
