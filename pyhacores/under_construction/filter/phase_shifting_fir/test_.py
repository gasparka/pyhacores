import numpy as np
from pyha.simulation.simulation_interface import plot_assert_sim_match, SIM_MODEL, SIM_HW_MODEL, debug_assert_sim_match

from pyhacores.under_construction.filter.phase_shifting_fir.model import PhaseShiftingFIR
import matplotlib.pyplot as plt


def test_basic():
    data = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    nrz = [1 if x == 1 else -1 for x in data]
    sps = 4
    nrz_data = np.array([[x] * sps for x in nrz]).flatten()
    # plt.plot(nrz_data)
    # plt.show()

    taps = [1 / sps] * sps
    match_filtered = np.convolve(nrz_data, taps, mode='full')

    i = PhaseShiftingFIR()

    mu = [0.5] * len(match_filtered)
    mu = np.array(range(len(match_filtered))) / len(match_filtered)
    # mu = [0.1] * 8 + [0.9] * 8
    # mu = [0.] * (len(match_filtered)//2) + [0.99] * (len(match_filtered)//2)
    # mu = [0.] * 4 + [0.25] *4 + [0.5] *4 + [0.75] *4  + [1.0] *4
    # mu = [0.5] * 16
    # mu = [0.2] * 16

    plot_assert_sim_match(i, None, match_filtered, mu, simulations=[SIM_MODEL, SIM_HW_MODEL])

    # plt.plot(iff[3:])
    # plt.plot(match_filtered)
    # plt.show()