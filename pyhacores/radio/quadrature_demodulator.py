import numpy as np
import pytest
from pyha import Hardware, Sfix, simulate, sims_close

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
        self.GAIN_SFIX = Sfix(self.gain * np.pi, 3, -14, round_style='round', overflow_style='saturate')

        self.DELAY = self.conjugate.DELAY + \
                     self.complex_mult.DELAY + \
                     self.angle.DELAY + 1

    def main(self, c):
        """
        :type c: ComplexSfix
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
    pytest.xfail('Has RTL/HWSIM mismatch in noise region..TODO')
    def make_fm(fs, deviation):
        # data signal
        periods = 1
        data_freq = 20
        time = np.linspace(0, periods, fs * periods, endpoint=False)
        data = np.cos(2 * np.pi * data_freq * time)

        # modulate
        sensitivity = 2 * np.pi * deviation / fs
        phl = np.cumsum(sensitivity * data)
        mod = np.exp(phl * 1j) * 0.9

        return mod, data

    fs = 1e3
    deviation = fs / 3
    demod_gain = fs / (2 * np.pi * deviation)

    inp, expect = make_fm(fs, deviation)
    expect = expect[1:] # because model eats one sample

    dut = QuadratureDemodulator(gain=demod_gain)
    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA', 'RTL'])
    assert sims_close(sims, expected=expect, rtol=1e-3)

    # assert sims_close(sim_out, rtol=1e-3)
    # # assert_sim_match(dut,
    # #                  expect, inputs,
    # #                  rtol=1e-3,
    # #                  atol=1e-3,
    # #                  # dir_path='/home/gaspar/git/pyha/playground/example'
    # #                  )
    # import matplotlib.pyplot as plt
    # plt.plot(sims['MODEL'], label='MODEL')
    # plt.plot(sims['PYHA'], label='PYHA')
    # plt.plot(sims['RTL'], label='RTL')
    # plt.legend()
    # plt.show()


# class TestQuadratureDemodulator_taranis:
#     def test_demod(self):
#         x = np.load('data/taranis_sps16_fs1600000.0_band1000000.0_far_onepackage_filter0.1.npy')
#
#         # x = x[:1000]
#
#         dut = QuadratureDemodulator()
#         debug_assert_sim_match(dut, None, x, simulations=[SIM_HW_MODEL])
#
# if __name__ == '__main__':
#     dut = TestQuadratureDemodulator_taranis()
#     dut.test_demod()







# class TestPhantom2:
#     """ Uses one chunk of Phantom 2 transmission """
#
#     def setup(self):
#         path = Path(__file__).parent / 'data/f2404_fs16.896_one_hop.iq'
#         inputs = load_gnuradio_file(str(path))
#         inputs = inputs[18000:19000]
#         self.mod = inputs
#         self.demod_gain = 1.5
#
#     def test_demod(self):
#         pytest.xfail('Has RTL/HWSIM mismatch in noise region..TODO')
#         inputs = self.mod
#
#         dut = QuadratureDemodulator(gain=self.demod_gain)
#         plot_assert_sim_match(dut, [ComplexSfix(left=0, right=-17)],
#                               None, inputs,
#                               rtol=1e-4,
#                               atol=1e-4,
#                               )