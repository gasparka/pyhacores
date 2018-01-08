from pathlib import Path

import numpy as np
import pytest
from pyha import Hardware, Sfix, simulate, sims_close, Complex

import pyhacores
from data import load_iq
from pyhacores.cordic import Angle
from pyhacores.util import ComplexConjugate, ComplexMultiply


class QuadratureDemodulator(Hardware):
    """
    http://gnuradio.org/doc/doxygen-3.7/classgr_1_1analog_1_1quadrature__demod__cf.html#details

    """
    def __init__(self, gain=1.0):
        """
        :param gain: inverse of tx sensitivity
        """
        self.gain = gain
        # components / registers
        self.conjugate = ComplexConjugate()
        self.complex_mult = ComplexMultiply()
        self.angle = Angle()
        self.y = Sfix(0, 0, -17, overflow_style='saturate')

        # pi term gets us to -1 to +1
        self.GAIN_SFIX = Sfix(gain * np.pi, 3, -17, round_style='round', overflow_style='saturate')

        self.DELAY = self.conjugate.DELAY + \
                     self.complex_mult.DELAY + \
                     self.angle.DELAY + 1

    def main(self, c):
        """
        :type c: Complex
        :rtype: Sfix
        """
        conj = self.conjugate.main(c)
        mult = self.complex_mult.main(c, conj)
        angle = self.angle.main(mult)

        self.y = self.GAIN_SFIX * angle
        return self.y

    def model_main(self, c):
        # this eats one input i.e output has one less element than input
        demod = np.angle(c[1:] * np.conjugate(c[:-1]))
        fix_gain = self.gain * demod
        return fix_gain


def test_fm_demodulation():
    def make_fm(fs, deviation):
        # data signal
        periods = 1
        data_freq = 20
        time = np.linspace(0, periods, fs * periods, endpoint=False)
        data = np.cos(2 * np.pi * data_freq * time) * 0.5

        # modulate
        sensitivity = 2 * np.pi * deviation / fs
        phl = np.cumsum(sensitivity * data)
        mod = np.exp(phl * 1j) * 0.5

        return mod, data

    fs = 1e3
    deviation = fs / 3
    demod_gain = fs / (2 * np.pi * deviation)

    inp, expect = make_fm(fs, deviation)
    expect = expect[1:] # because model eats one sample

    dut = QuadratureDemodulator(gain=demod_gain)
    sims = simulate(dut, inp)
    assert sims_close(sims, expected=expect, rtol=1e-3)


def test_demod_phantom2_signal():
    path = Path(pyhacores.__path__[0]) / '../data/f2404_fs16.896_one_hop.iq'
    iq = load_iq(str(path))[19000:20000] # this part has only bits..i.e no noisy stuff

    dut = QuadratureDemodulator(gain=1/np.pi)
    sims = simulate(dut, iq)

    # import matplotlib.pyplot as plt
    # plt.plot(sims['MODEL'], label='MODEL')
    # plt.plot(sims['PYHA'], label='PYHA')
    # plt.plot(sims['RTL'], label='RTL')
    # plt.legend()
    # plt.show()

    assert sims_close(sims, atol=1e-4)


def test_demod_phantom2_noise():
    pytest.xfail('cant match noisy stuff with fixed point :(')
    path = Path(pyhacores.__path__[0]) / '../data/f2404_fs16.896_one_hop.iq'
    iq = load_iq(str(path))[:500] # ONLY NOISE

    dut = QuadratureDemodulator(gain=1/np.pi)
    sims = simulate(dut, iq)

    # import matplotlib.pyplot as plt
    # plt.plot(sims['MODEL'], label='MODEL')
    # plt.plot(sims['PYHA'], label='PYHA')
    # plt.plot(sims['RTL'], label='RTL')
    # plt.legend()
    # plt.show()

    assert sims_close(sims, atol=1e-4)
