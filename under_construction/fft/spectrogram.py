import numpy as np
import pytest

from data import load_iq
from pyha import Hardware, simulate, hardware_sims_equal, sims_close
from under_construction.fft.avg_decimate import AvgDecimate
from under_construction.fft.bit_reversal_fftshift import BitReversal
from under_construction.fft.conjmult import ConjMult
from under_construction.fft.fft_core import R2SDF
from under_construction.fft.windower import Windower
from scipy import signal


class Spectrogram(Hardware):
    def __init__(self, nfft, window_type='hanning', decimate_by=2):
        self.decimate_by = decimate_by
        self.nfft = nfft
        self.window_type = window_type

        # components
        self.windower = Windower(nfft, self.window_type)
        self.fft = R2SDF(nfft)
        # self.bit_reversal_fftshift = BitReversal(nfft, fftshift=True)
        # self.abs = ConjMult()
        # self.avg_decimate = AvgDecimate(decimate_by)

        self.DELAY = self.windower.DELAY + self.fft.DELAY

    def main(self, x):
        window_out = self.windower.main(x)
        fft_out = self.fft.main(window_out)
        return fft_out
        # bitshift_out = self.bit_reversal_fftshift.main(fft_out)
        # mag_out = self.abs.main(bitshift_out)
        # dec_out = self.avg_decimate.main(mag_out)
        # return dec_out

    def model_main(self, x):
        window_out = self.windower.model_main(x)
        fft_out = self.fft.model_main(window_out)
        return fft_out

        # _, _, spectro_out = signal.spectrogram(x, 1, nperseg=self.nfft, return_onesided=False, detrend=False,
        #                                        noverlap=0, window='hann')
        #
        # # fftshift
        # spectro_out = np.roll(spectro_out, self.nfft // 2, axis=0)
        #
        # # avg decimation
        # x = np.split(spectro_out, len(spectro_out) // self.decimate_by)
        # golden_output = np.average(x, axis=1)
        #
        # return golden_output


@pytest.mark.parametrize("fft_points", [2, 4, 8, 16, 32, 64, 128, 256])
def test_ph2(fft_points):
    inp = load_iq('/home/gaspar/git/pyhacores/data/f2404_fs16.896_one_hop.iq')
    inp = signal.decimate(inp, 8)[:len(inp) // 64]
    inp /= inp.max() * 2
    inp = np.array(inp[:int(len(inp) // fft_points) * fft_points])
    print(inp.shape)


    dut = Spectrogram(nfft=fft_points)
    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA'])

    # import matplotlib.pyplot as plt
    # plt.plot(sims['MODEL'][0])
    # plt.plot(sims['PYHA'][0])
    # plt.show()

    assert sims_close(sims, rtol=1e-1, atol=1e-4)

