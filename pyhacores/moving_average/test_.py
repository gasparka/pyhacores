import numpy as np
import pytest

from pyha.simulation.simulation_interface import assert_sim_match, SIM_HW_MODEL
from pyhacores.moving_average.model import MovingAverage


class TestMovingAverage:
    def test_window1(self):
        with pytest.raises(AttributeError):
            mov = MovingAverage(window_len=1)

    def test_window2(self):
        mov = MovingAverage(window_len=2)
        x = [0.0, 0.1, 0.2, 0.3, 0.4]
        expected = [0.0, 0.05, 0.15, 0.25, 0.35]
        assert_sim_match(mov, expected, x)

    def test_window3(self):
        mov = MovingAverage(window_len=4)
        x = [-0.2, 0.05, 1.0, -0.9571, 0.0987]
        expected = [-0.05, -0.0375, 0.2125, -0.026775, 0.0479]
        assert_sim_match(mov, expected, x)


    def test_max(self):
        mov = MovingAverage(window_len=4)
        x = [1., 1., 1., 1., 1., 1.]
        expected = [0.25, 0.5, 0.75, 1., 1., 1.]
        assert_sim_match(mov, expected, x)

    def test_min(self):
        mov = MovingAverage(window_len=8)
        x = [-1., -1., -1., -1., -1., -1., -1., -1., -1.]
        expected = [-0.125, -0.25, -0.375, -0.5, -0.625, -0.75, -0.875, -1., -1.]
        assert_sim_match(mov, expected, x)

    def test_noisy_signal(self):
        np.random.seed(0)
        mov = MovingAverage(window_len=8)
        x = np.linspace(0, 2 * 2 * np.pi, 512)
        y = 0.7 * np.sin(x)
        noise = 0.1 * np.random.normal(size=512)
        y += noise

        assert_sim_match(mov, None, y)
