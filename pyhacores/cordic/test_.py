import numpy as np
import pytest
from scipy.signal import chirp, hilbert

from pyha.simulation.simulation_interface import assert_sim_match, SIM_MODEL, SIM_HW_MODEL, SIM_RTL, SIM_GATE

from pyhacores.cordic.model import Cordic, ToPolar, Angle, Abs, NCO
from pyhacores.cordic.model import CordicMode


def chirp_stimul():
    """ Amplitude modulated chirp signal """
    duration = 1.0
    fs = 256
    samples = int(fs * duration)
    t = np.arange(samples) / fs
    signal = chirp(t, 20.0, t[-1], 100.0)
    signal *= (1.0 + 0.5 * np.sin(2.0 * np.pi * 3.0 * t))
    # import matplotlib.pyplot as plt
    # plt.plot(signal)
    # plt.show()
    analytic_signal = hilbert(signal) * 0.5
    ref_abs = np.abs(analytic_signal)
    ref_instantaneous_phase = np.angle(analytic_signal)
    return analytic_signal, ref_abs, ref_instantaneous_phase


class TestCordic:
    def test_small_vectoring(self):
        inputs = (np.random.rand(3, 8) * 2 - 1)

        dut = Cordic(16, CordicMode.VECTORING)
        assert_sim_match(dut, None, *inputs, atol=1e-4)

    def test_vectoring(self):
        np.random.seed(123456)
        inputs = 0.5 * (np.random.rand(3, 128) * 2 - 1)

        dut = Cordic(16, CordicMode.VECTORING)
        assert_sim_match(dut, None, *inputs, atol=1e-4)

    def test_rotation(self):
        np.random.seed(123456)
        inputs = 0.5 * (np.random.rand(3, 128) * 2 - 1)

        dut = Cordic(16, CordicMode.ROTATION)
        assert_sim_match(dut, None, *inputs, atol=1e-4)


class TestToPolar:
    def test_polar_quadrant_i(self):
        inputs = [0.234 + 0.92j]
        expect = [np.abs(inputs), np.angle(inputs) / np.pi]

        dut = ToPolar()
        assert_sim_match(dut, expect, inputs)

    def test_polar_quadrant_ii(self):
        inputs = [-0.234 + 0.92j]
        expect = [np.abs(inputs), np.angle(inputs) / np.pi]

        dut = ToPolar()
        assert_sim_match(dut, expect, inputs)

    def test_polar_quadrant_iii(self):
        inputs = [-0.234 - 0.92j]
        expect = [np.abs(inputs), np.angle(inputs) / np.pi]

        dut = ToPolar()
        assert_sim_match(dut, expect, inputs, rtol=1e-4)

    def test_polar_quadrant_iv(self):
        inputs = [0.234 - 0.92j]
        expect = [np.abs(inputs), np.angle(inputs) / np.pi]

        dut = ToPolar()
        assert_sim_match(dut, expect, inputs, rtol=1e-4)

    def test_overflow_condition(self):
        pytest.xfail('abs would be > 1 (1.84)')
        inputs = [0.92j + 0.92j]
        expect = [np.abs(inputs), np.angle(inputs) / np.pi]
        dut = ToPolar()
        assert_sim_match(dut, expect, inputs)

    def test_chirp(self):
        analytic_signal, ref_abs, ref_instantaneous_phase = chirp_stimul()

        inputs = analytic_signal
        expect = [ref_abs, ref_instantaneous_phase / np.pi]

        dut = ToPolar()
        assert_sim_match(dut, expect, inputs, atol=1e-4)


class TestAngle:
    def test_basic(self):
        inputs = [np.exp(0.5j), np.exp(0.1j)]
        expect = [0.5 / np.pi, 0.1 / np.pi]

        dut = Angle()
        assert_sim_match(dut, expect, inputs, atol=1e-4)

    def test_chirp(self):
        analytic_signal, ref_abs, ref_instantaneous_phase = chirp_stimul()

        inputs = analytic_signal
        expect = ref_instantaneous_phase / np.pi

        dut = Angle()
        assert_sim_match(dut, expect, inputs, atol=1e-4)


class TestAbs:
    def test_basic(self):
        inputs = [-0.25 * np.exp(0.5j), 0.123 * np.exp(0.1j)]
        expect = [0.25, 0.123]

        dut = Abs()
        assert_sim_match(dut, expect, inputs)

    def test_chirp(self):
        analytic_signal, ref_abs, ref_instantaneous_phase = chirp_stimul()

        inputs = analytic_signal
        expect = ref_abs

        dut = Abs()
        assert_sim_match(dut, expect, inputs)


class TestNCO:
    def test_basic(self):
        inputs = [0.01] * 4
        expect = [np.exp(0.01j*np.pi), np.exp(0.02j*np.pi), np.exp(0.03j*np.pi), np.exp(0.04j*np.pi)]

        dut = NCO()
        assert_sim_match(dut, expect, inputs)

    @pytest.mark.parametrize('period', [0.25, 0.50, 0.75, 1, 2, 4])
    def test_nco(self, period):
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
        sims = [SIM_MODEL, SIM_HW_MODEL, SIM_RTL]
        if period == 4:
            sims = [SIM_MODEL, SIM_HW_MODEL, SIM_RTL, SIM_GATE]

        assert_sim_match(dut, expect, inputs, simulations=sims)

