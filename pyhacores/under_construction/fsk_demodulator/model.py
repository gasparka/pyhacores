import numpy as np
from pyha.common.hwsim import HW
from pyha.common.sfix import Sfix, ComplexSfix
from pyhacores.moving_average.model import MovingAverage
from pyhacores.under_construction.quadrature_demodulator.model import QuadratureDemodulator


class FSKDemodulator(HW):
    """
    Takes in complex signal and gives out bits. It uses Quadrature demodulator followed by
    matched filter (moving average). M&M clock recovery is the last DSP block, it performs timing recovery.

    .. note:: M&M clock recovery is currently not implemented
    """

    def __init__(self, deviation, fs, sps):
        self.fs = fs
        self.deviation = deviation

        self.gain = fs / (2 * np.pi * self.deviation) / np.pi

        self.demod = QuadratureDemodulator(self.gain)
        self.match = MovingAverage(sps)

        # constants
        self._delay = self.demod._delay + self.match._delay

    def main(self, input):
        """
        :type  input: ComplexSfix
        :rtype: Sfix
        """
        demod = self.demod.main(input)
        match = self.match.main(demod)

        return match

    def model_main(self, input_list):
        demod = self.demod.model_main(input_list)
        match = self.match.model_main(demod)

        return match


