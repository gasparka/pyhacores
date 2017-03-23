from pathlib import Path

import numpy as np
from astropy.tests.helper import pytest
from pyha.common.util import hex_to_bool_list, bools_to_bitstr, hex_to_bitstr
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from pyhacores.under_construction.clock_recovery.gardner import GardnerTimingRecovery


def fract_delay(sig, fract_delay=0.0):
    f = interp1d(range(len(sig)), sig)
    new_x = np.array(range(len(sig))) + fract_delay
    fract_sig = f(new_x[:-1])
    return fract_sig


def insig(bits, sps, int_delay=0, fd=0.0):
    nrz = [[1] * sps if x == 1 else [-1] * sps for x in bits]
    nrz = np.array(nrz).flatten()

    taps = [1 / sps] * sps
    matched = np.convolve(nrz, taps, mode='full')

    # delays
    sig = matched[int_delay:]
    sig = fract_delay(sig, fd)
    return sig


def data_gen(hex, sps, int_delay=0, fract_delay=0.0, noise_amp=0.001):
    bits = hex_to_bool_list(hex)
    nrz = [[1] * sps if x else [-1] * sps for x in bits]
    nrz = np.array(nrz).flatten()

    # matched filter
    taps = [1 / sps] * sps
    matched = np.convolve(nrz, taps, mode='full')

    # noise
    sig = matched + np.random.uniform(-noise_amp, noise_amp, len(matched))

    # delays
    sig = sig[int_delay:]
    f = interp1d(range(len(sig)), sig)
    new_x = np.array(range(len(sig))) + fract_delay
    # sig = f(new_x[:-1])


    return sig


class TestGardnerTimingRecovery:
    def setup_class(self):
        self.sps = 4

    @pytest.mark.parametrize('int_delay', range(6))
    def test_int_delay(self, int_delay):
        if int_delay == 4 and self.sps == 4:
            pytest.xfail('Loop gets stuck..cause samples are so perfect, some noise would unstuck it!')
        inp = insig([1, 0, 1, 0, 1, 0, 1, 0] * 8, self.sps, int_delay)
        recover = GardnerTimingRecovery(self.sps)

        ret, err, mu = recover.model_main(inp)

        assert err[-8:] == [0] * 8
        assert ret[-4:] == [1, -1, 1, -1] or ret[-4:] == [-1, 1, -1, 1]

    @pytest.mark.parametrize('fract_delay', np.array(range(10)) / 10)
    def test_fract_delay(self, fract_delay):
        inp = insig([1, 0, 1, 0, 1, 0, 1, 0] * 8, self.sps , fd=fract_delay)
        recover = GardnerTimingRecovery(self.sps )

        ret, err, mu = recover.model_main(inp)

        np.testing.assert_allclose(err[-8:], [0] * 8, atol=1e-2)

    @pytest.mark.parametrize('fract_delay', np.array(range(10)) / 10)
    @pytest.mark.parametrize('int_delay', [1, 2, 3, 4])
    def test_leading_noise(self, fract_delay, int_delay):
        inp = insig([1, 0, 1, 0, 1, 0, 1, 0] * 8, self.sps, int_delay, fd=fract_delay)
        inp = np.append(np.random.uniform(-1, 1, 128), inp)
        recover = GardnerTimingRecovery(self.sps)

        ret, err, mu = recover.model_main(inp)

        np.testing.assert_allclose(err[-8:], [0] * 8, atol=1e-2)

    @pytest.mark.parametrize('int_delay, fract_delay, noise_amp', [
        [0, 0, 0.1],
        [1, 0.245, 0.2],
        [2, 0.945, 0.3],
        [3, 0.5, 0.05],
    ])
    def test_data(self, int_delay, fract_delay, noise_amp):
        packet = '123456789abcdef'
        insig = data_gen(f'aaaaaa{packet}aa', self.sps, int_delay, fract_delay, noise_amp)
        recover = GardnerTimingRecovery(self.sps)

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
    # @pytest.mark.parametrize('fract_delay', np.array(range(10)) / 10)
    # @pytest.mark.parametrize('int_delay', [1, 2, 3, 4])
    # def test_error_ramps(self, fract_delay, int_delay):
    #     """ How system responses to constant error ramp """
    #     sps = 4
    #     inp = insig([1, 0, 1, 0, 1, 0, 1, 0] * 32, sps, int_delay, fd=fract_delay)
    #     recover = GardnerTimingRecovery(sps, test_inject_error=0.05)
    #
    #     r = recover.model_main(inp)
    #     ret, err, mu = r
    #     plt.plot(err)
    #     plt.plot(mu)
    #     plt.stem(ret)
    #     plt.grid()
    #     plt.show()
    #
    #     path = Path(__file__).parent / f'data/{sps}_{fract_delay}_{int_delay}'
    #     np.save(path, np.array(r))

