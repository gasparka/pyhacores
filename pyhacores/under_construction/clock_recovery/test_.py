import numpy as np
from astropy.tests.helper import pytest
from scipy.interpolate import interp1d

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


class TestGardnerTimingRecovery:
    @pytest.mark.parametrize('sps', [2, 4])
    @pytest.mark.parametrize('int_delay', range(8))
    def test_int_delay(self, sps, int_delay):
        inp = insig([1, 0, 1, 0, 1, 0, 1, 0] * 8, sps, int_delay)
        recover = GardnerTimingRecovery(sps)

        ret, err, mu = recover.model_main(inp)

        assert err[-8:] == [0] * 8
        assert ret[-4:] == [1, -1, 1, -1] or ret[-4:] == [-1, 1, -1, 1]

    @pytest.mark.parametrize('sps', [2, 4])
    @pytest.mark.parametrize('fract_delay', np.array(range(10)) / 10)
    def test_fract_delay(self, sps, fract_delay):
        inp = insig([1, 0, 1, 0, 1, 0, 1, 0] * 8, sps, fd=fract_delay)
        recover = GardnerTimingRecovery(sps)

        ret, err, mu = recover.model_main(inp)

        np.testing.assert_allclose(err[-8:], [0] * 8, atol=1e-2)

