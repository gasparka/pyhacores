import numpy as np

from pyha.common.const import Const
from pyha.common.hwsim import HW, default_sfix
from pyha.common.sfix import Sfix, left_index, right_index, fixed_wrap, fixed_truncate
from pyha.common.sfix import resize
from pyha.common.util import is_power2


class MovingAverage(HW):
    """
    Moving average algorithm.

    :param window_len: Size of the moving average window, must be power of 2
    """
    def __init__(self, window_len):
        self.window_len = window_len
        if window_len < 2:
            raise AttributeError('Window length must be >= 2')

        if not is_power2(window_len):
            raise AttributeError('Window length must be power of 2')

        self.mem = [Sfix()] * self.window_len
        self.sum = Sfix(0, 0, -17, overflow_style=fixed_wrap)

        self.window_pow = Const(int(np.log2(window_len)))

        self._delay = 1

    def main(self, x):
        """
        This works by keeping a history of 'window_len' elements and sum of them.
        Every clock last element will be subtracted and new added to the sum. These are already scaled before.
        More good infos: https://www.dsprelated.com/showarticle/58.php

        :param x: input to average
        :return: averaged output
        :rtype: Sfix
        """
        # divide by window_pow
        div = x >> self.window_pow

        # add new element to shift register
        self.mem = [div] + self.mem[:-1]

        # calculate new sum
        self.sum = self.sum + div - self.mem[-1]
        return self.sum

    def model_main(self, inputs):
        taps = [1 / self.window_len] * self.window_len
        ret = np.convolve(inputs, taps, mode='full')
        return ret[:-self.window_len + 1]
