from astropy.tests.helper import pytest
from pyha.common.util import bools_to_bitstr, hex_to_bitstr
from pyha.simulation.simulation_interface import debug_assert_sim_match, SIM_HW_MODEL, SIM_MODEL

from pyhacores.under_construction.clock_recovery.simple_gardner.gardner import SimpleGardnerTimingRecovery
from pyhacores.under_construction.clock_recovery.test_ import data_gen
import numpy as np
import matplotlib.pyplot as plt


class TestSimpleGardnerTimingRecovery:

    @pytest.mark.parametrize('sps', [4, 8, 16])
    @pytest.mark.parametrize('int_delay, fract_delay, noise_amp', [
        [0, 0, 0.1],
        [1, 0.245, 0.2],
        [2, 0.945, 0.3],
        [3, 0.5, 0.05],
        [9, 0.888, 0.1],
        [5, 0.34, 0.25],
    ])
    def test_data(self, sps, int_delay, fract_delay, noise_amp):
        self.sps = sps
        packet = '123456789abcdef'
        insig = data_gen(f'aaaaaa{packet}aa', self.sps, int_delay, fract_delay, noise_amp)
        recover = SimpleGardnerTimingRecovery(self.sps)

        ret, err, mu = recover.model_main(insig)
        # plt.plot(ret, label='ret')
        # plt.plot(err, label='err')
        # plt.plot(mu, label='mu')
        # plt.grid()
        # plt.legend()
        # plt.show()

        bits = bools_to_bitstr([1 if x > 0 else 0 for x in ret])
        packet_bits = hex_to_bitstr(packet)
        assert packet_bits in bits



def test_hw_model():
    sps = 8
    packet = '123456789abcdef'
    insig = data_gen(f'aaaaaa{packet}aa', sps, 1, 0.3, 0)
    dut = SimpleGardnerTimingRecovery(sps)

    r = debug_assert_sim_match(dut, None, insig, simulations=[SIM_MODEL, SIM_HW_MODEL])

    rt = np.transpose(r[1])

    rt = np.transpose([x for x in rt if x[3]])
    from pylab import rcParams
    rcParams['figure.figsize'] = 50, 10
    plt.subplot(1,3,1)
    plt.plot(r[0][0])
    plt.plot(rt[0])

    plt.subplot(1, 3, 2)
    plt.plot(rt[1])
    plt.plot(r[0][1])

    plt.subplot(1, 3, 3)
    plt.plot(rt[2])
    plt.plot(r[0][2])

    plt.show()
    pass