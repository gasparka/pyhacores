
from astropy.tests.helper import pytest
from pyha.common.util import hex_to_bool_list, bools_to_bitstr, hex_to_bitstr

from pyhacores.under_construction.clock_recovery.simple_gardner.gardner import SimpleGardnerTimingRecovery
from pyhacores.under_construction.clock_recovery.simple_gardner.test_ import data_gen


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

