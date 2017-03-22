import numpy as np
from astropy.tests.helper import pytest
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


class TestGardnerTimingRecovery:
    def setup_class(self):
        self.sps = 4

    @pytest.mark.parametrize('int_delay', range(6))
    def test_int_delay(self, int_delay):
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



def test_debug():
    # bug 3->4
    sps = 4
    inp = insig([1, 0, 1, 0, 1, 0, 1, 0] * 64, sps, 0, fd=0.0)
    recover = GardnerTimingRecovery(sps, test_inject_error=0.05)

    ret, err, mu = recover.model_main(inp)
    plt.plot(err)
    plt.plot(mu)
    # plt.plot(np.array(mu)%1)
    # plt.plot(np.diff(err))
    # plt.plot(recover.out_int)
    plt.grid()
    plt.show()


def test_up2(self):
    # bug 3->4
    inp = insig([1, 0, 1, 0, 1, 0, 1, 0] * 64, self.sps, 2, fd=0.3)
    recover = GardnerTimingRecovery(self.sps)

    ret, err, mu = recover.model_main(inp)
    plt.plot(err)
    plt.plot(mu)
    plt.plot(np.array(mu)%1)
    plt.grid()
    plt.show()

def test_upbug1(self):
    # bug 4->5
    # 3->4 OK
    inp = insig([1, 0, 1, 0, 1, 0, 1, 0] * 8, self.sps, 3, fd=0.3)
    recover = GardnerTimingRecovery(self.sps)

    ret, err, mu = recover.model_main(inp)
    plt.plot(err)
    plt.plot(mu)
    plt.plot(np.array(mu)%1)
    plt.grid()
    plt.show()

def test_downbug2(self):
    # bug 4->3
    inp = insig([1, 0, 1, 0, 1, 0, 1, 0] * 8, self.sps, 1, fd=0.3)
    recover = GardnerTimingRecovery(self.sps)
    recover.d = 4

    ret, err, mu = recover.model_main(inp)
    plt.plot(err)
    plt.plot(mu)
    plt.plot(np.array(mu)%1)
    plt.grid()
    plt.show()

def test_upbug3(self):
    # bug 5->6
    # 4-5 OK
    inp = insig([1, 0, 1, 0, 1, 0, 1, 0] * 8, self.sps, 0, fd=0.3)
    recover = GardnerTimingRecovery(self.sps)
    recover.d = 4

    ret, err, mu = recover.model_main(inp)
    plt.plot(err)
    plt.plot(mu)
    plt.plot(np.array(mu)%1)
    plt.grid()
    plt.show()

def test_upbug44(self):
    # bug 2->3 TOTAL OSCILLATION
    inp = insig([1, 0, 1, 0, 1, 0, 1, 0] * 8, self.sps, 1, fd=0.3)
    recover = GardnerTimingRecovery(self.sps)
    recover.d = 2

    ret, err, mu = recover.model_main(inp)
    plt.plot(err)
    plt.plot(mu)
    plt.plot(np.array(mu)%1)
    plt.grid()
    plt.show()