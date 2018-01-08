import pytest
from pyha import Hardware, Sfix, Complex, simulate, sims_close
import numpy as np

from pyhacores.cordic import CordicMode, Cordic


class NCO(Hardware):
    """
    Baseband signal generator. Integrated phase accumulator.
    """

    def __init__(self):
        self.cordic = Cordic(17, CordicMode.ROTATION)
        self.phase_acc = Sfix(0, 0, -24, wrap_is_ok=True)
        self.out = Complex(0, 0, -17, overflow_style='saturate')
        self.DELAY = self.cordic.ITERATIONS + 1 + 1
        self.INIT_X = 1.0 / 1.646760  # gets rid of cordic gain, could add amplitude modulation here

    def main(self, phase_inc):
        """
        :param phase_inc: amount of rotation applied for next clock cycle, must be normalized to -1 to 1.
        :rtype: Complex
        """
        self.phase_acc = self.phase_acc + phase_inc

        start_x = self.INIT_X
        start_y = Sfix(0.0, 0, -17)

        x, y, phase = self.cordic.main(start_x, start_y, self.phase_acc)

        self.out.real = x
        self.out.imag = y

        return self.out

    def model_main(self, phase_list):
        p = np.cumsum(np.array(phase_list) * np.pi)
        return np.exp(p * 1j)


def test_basic():
    inputs = [0.01] * 4
    expect = [np.exp(0.01j * np.pi), np.exp(0.02j * np.pi), np.exp(0.03j * np.pi), np.exp(0.04j * np.pi)]

    dut = NCO()
    sim_out = simulate(dut, inputs)
    assert sims_close(sim_out, expect)


@pytest.mark.parametrize('period', [0.25, 0.50, 0.75, 1, 2, 4, 8, 16])
def test_nco(period):
    fs = 64
    freq = 1
    phase_inc = 2 * np.pi * freq / fs
    phase_cumsum = np.arange(0, period * fs * phase_inc, phase_inc)

    ref = np.exp(phase_cumsum * 1j)

    pil = np.diff(phase_cumsum) / np.pi
    pil = np.insert(pil, 0, [0.0])

    inputs = pil
    expect = ref

    dut = NCO()
    sims = ['MODEL', 'PYHA', 'RTL']
    if period == 16:
        sims.append('GATE')

    sim_out = simulate(dut, inputs, simulations=sims)
    assert sims_close(sim_out, expect)
