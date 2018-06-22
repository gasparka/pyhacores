from scipy import signal

import pytest
from pyha import Hardware, Sfix, simulate, sims_close, Complex
from pyha.common.util import is_power2
import numpy as np


class MovingAverage(Hardware):
    """
    Moving average algorithm.
    This can be used for signal smoothing (low pass filter) or matched filter/detector for rectangular signals.

    :param window_len: Size of the moving average window, must be power of 2
    """

    def __init__(self, window_len):
        self.WINDOW_LEN = window_len
        if window_len < 2:
            raise AttributeError('Window length must be >= 2')

        if not is_power2(window_len):
            raise AttributeError('Window length must be power of 2')

        self.mem = [Complex()] * self.WINDOW_LEN
        self.WINDOW_POW = int(np.log2(window_len))
        self.sum = Complex(0, self.WINDOW_POW, -17)

        self.DELAY = 1

    def main(self, x):
        """
        This works by keeping a history of 'window_len' elements and the sum of them.
        Every clock last element will be subtracted and new added to the sum.
        More good infos: https://www.dsprelated.com/showarticle/58.php

        :param x: input to average
        :return: averaged output
        """
        # add new element to shift register
        self.mem = [x] + self.mem[:-1]

        # calculate new sum
        self.sum = self.sum + x - self.mem[-1]
        return self.sum >> self.WINDOW_POW

    def model_main(self, inputs):
        # MA expressed as FIR filter
        taps = [1 / self.WINDOW_LEN] * self.WINDOW_LEN
        return signal.lfilter(taps, [1.0], inputs)


def test_incorrect_conf():
    with pytest.raises(AttributeError):
        mov = MovingAverage(window_len=1)  # too small window

    with pytest.raises(AttributeError):
        mov = MovingAverage(window_len=3)  # not power of 2


def test_window2():
    dut = MovingAverage(window_len=2)
    x = [0.0, 0.1, 0.2, 0.3, 0.4]
    expected = [0.0, 0.05, 0.15, 0.25, 0.35]
    sim_out = simulate(dut, x)
    assert sims_close(sim_out, expected)


def test_window4():
    dut = MovingAverage(window_len=4)
    x = [-0.2, 0.05, 1.0, -0.9571, 0.0987]
    expected = [-0.05, -0.0375, 0.2125, -0.026775, 0.0479]
    sim_out = simulate(dut, x)
    assert sims_close(sim_out, expected)


def test_max():
    dut = MovingAverage(window_len=4)
    x = [1., 1., 1., 1., 1., 1.]
    expected = [0.25, 0.5, 0.75, 1., 1., 1.]
    sim_out = simulate(dut, x)
    assert sims_close(sim_out, expected)


def test_min():
    dut = MovingAverage(window_len=8)
    x = [-1., -1., -1., -1., -1., -1., -1., -1., -1.]
    expected = [-0.125, -0.25, -0.375, -0.5, -0.625, -0.75, -0.875, -1., -1.]
    sim_out = simulate(dut, x)
    assert sims_close(sim_out, expected)


def test_noisy_signal():
    np.random.seed(0)
    dut = MovingAverage(window_len=8)
    x = np.linspace(0, 2 * 2 * np.pi, 512)
    y = 0.7 * np.sin(x)
    noise = 0.1 * np.random.normal(size=512)
    y += noise

    sim_out = simulate(dut, y)
    assert sims_close(sim_out, atol=1e-4)


def test_noisy_complex():
    np.random.seed(0)
    dut = MovingAverage(window_len=8)
    x = np.linspace(0, 2 * 2 * np.pi, 512)
    y = 0.7 * np.sin(x)
    noise = 0.1 * np.random.normal(size=512)
    y += noise

    sim_out = simulate(dut, y)
    assert sims_close(sim_out, atol=1e-4)
