from astropy.tests.helper import pytest
from pyha.common.sfix import Sfix
from pyha.common.util import bools_to_bitstr, hex_to_bitstr
from pyha.simulation.simulation_interface import debug_assert_sim_match, SIM_HW_MODEL, SIM_MODEL, SIM_RTL, SIM_GATE

from pyhacores.under_construction.clock_recovery.simple_gardner.gardner import SimpleGardnerTimingRecovery
from pyhacores.under_construction.clock_recovery.test_ import data_gen
import numpy as np
import matplotlib.pyplot as plt

from pylab import rcParams
rcParams['figure.figsize'] = 50, 10


class TestSimpleGardnerTimingRecovery:

    @pytest.mark.parametrize('sps', [8, 10, 12, 14, 16])
    @pytest.mark.parametrize('int_delay, fract_delay, noise_amp', [
        [0, 0, 0.1],
        [1, 0.245, 0.024],
        [2, 0.945, 0.08],
        [3, 0.5, 0.05],
        [9, 0.888, 0.1],
        [5, 0.34, 0.1],
    ])
    def test_data(self, sps, int_delay, fract_delay, noise_amp):
        self.sps = sps
        packet = '123456789abcdef'
        insig = data_gen(f'aaaaaaaa{packet}aa', self.sps, int_delay, fract_delay, noise_amp)
        dut = SimpleGardnerTimingRecovery(self.sps)

        r = debug_assert_sim_match(dut, None, insig, simulations=[SIM_MODEL, SIM_HW_MODEL],
                                   types=[Sfix(0, 1, -17)])


        # filter out invalid data
        rt = np.transpose([x for x in np.transpose(r[1]) if x[3]])

        # plt.subplot(1, 3, 1)
        # plt.plot(rt[0])
        # plt.plot(r[0][0])
        #
        # plt.subplot(1, 3, 2)
        # plt.plot(rt[1])
        # plt.plot(r[0][1])
        #
        # plt.subplot(1, 3, 3)
        # plt.plot(rt[2])
        # plt.plot(r[0][2])
        #
        # plt.show()
        packet_bits = hex_to_bitstr(packet)

        bits_model = bools_to_bitstr([1 if x > 0 else 0 for x in r[0][0]])
        bits_hwmodel = bools_to_bitstr([1 if x > 0 else 0 for x in rt[0]])
        assert packet_bits in bits_model
        assert packet_bits in bits_hwmodel



def test_debug():
    sps = 8
    packet = '123456789abcdef'
    insig = data_gen(f'aaaaaa{packet}aa', sps, 1, 0.9, 0.01)
    dut = SimpleGardnerTimingRecovery(sps)

    eng = True
    sims = [SIM_MODEL, SIM_HW_MODEL]
    if eng:
        sims = [SIM_MODEL, SIM_HW_MODEL, SIM_RTL, SIM_GATE]

    r = debug_assert_sim_match(dut, None, insig,
                               simulations=sims,
                               dir_path='/home/gaspar/git/pyhacores/playground')


    # filter out invalid data
    rhw = np.transpose([x for x in np.transpose(r[1]) if x[3]])

    if eng:
        rrtl = np.transpose([x for x in np.transpose(r[2]) if x[3]])
        rgate = np.transpose([x for x in np.transpose(r[3]) if x[3]])


    plt.subplot(1,3,1)
    plt.plot(rhw[0], label='HW')
    plt.plot(r[0][0], label='MODEL')
    if eng:
        plt.plot(rrtl[0], label='RTL')
        plt.plot(rgate[0], label='GATE')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(rhw[1], label='HW')
    plt.plot(r[0][1], label='MODEL')
    if eng:
        plt.plot(rrtl[1], label='RTL')
        plt.plot(rgate[1], label='GATE')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(rhw[2], label='HW')
    plt.plot(r[0][2], label='MODEL')
    if eng:
        plt.plot(rrtl[2], label='RTL')
        plt.plot(rgate[2], label='GATE')
    plt.legend()

    plt.show()
    pass