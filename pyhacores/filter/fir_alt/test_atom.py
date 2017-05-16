import numpy as np
from pyha.common.util import plot_freqz
from pyha.simulation.simulation_interface import assert_sim_match
from scipy import signal

from pyhacores.filter.fir_alt.model_atom import FIR_atom


def test_demo():
    # design filter
    b = signal.remez(4, [0, 0.1, 0.3, 0.5], [1, 0])
    # plot_freqz(b)

    dut = FIR_atom(b)
    inp = np.random.uniform(-1, 1, 64)

    # assert_sim_match(dut, None, inp, dir_path='/home/gaspar/git/thesis/playground' )


def test_symmetric():
    taps = [0.01, 0.02, 0.03, 0.04, 0.03, 0.02, 0.01]
    dut = FIR_atom(taps)
    inp = np.random.uniform(-1, 1, 64)

    assert_sim_match(dut, None, inp)


# def test_sfix_bug():
#     """ There was Sfix None bound based bug that made only 5. output different """
#     np.random.seed(4)
#     taps = [0.01, 0.02, 0.03, 0.04, 0.03, 0.02, 0.01]
#     dut = FIR(taps)
#     inp = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
#
#     assert_sim_match(dut, None, inp)
#
#
# def test_non_symmetric():
#     taps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
#     dut = FIR(taps)
#     inp = np.random.uniform(-1, 1, 128)
#
#     assert_sim_match(dut, None, inp)
#
#
# def test_remez16():
#     np.random.seed(0)
#     taps = signal.remez(16, [0, 0.1, 0.2, 0.5], [1, 0])
#     dut = FIR(taps)
#     inp = np.random.uniform(-1, 1, 64)
#
#     assert_sim_match(dut, None, inp)
#
#
# def test_remez32():
#     np.random.seed(1)
#     taps = signal.remez(32, [0, 0.1, 0.2, 0.5], [1, 0])
#     dut = FIR(taps)
#     inp = np.random.uniform(-1, 1, 64) * 0.9
#
#     assert_sim_match(dut, None, inp)
#
#
# def test_remez128():
#     np.random.seed(2)
#     taps = signal.remez(128, [0, 0.1, 0.2, 0.5], [1, 0])
#     dut = FIR(taps)
#     inp = np.random.uniform(-1, 1, 128)
#
#     assert_sim_match(dut, None, inp)