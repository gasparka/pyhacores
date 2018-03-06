import numpy as np
import pytest

from data import load_iq
from pyha import Hardware, simulate, hardware_sims_equal, sims_close
from under_construction.fft.avg_decimate import AvgDecimate
from under_construction.fft.bit_reversal_fftshift import BitReversal
from under_construction.fft.conjmult import ConjMult
from under_construction.fft.fft_core import R2SDF
from under_construction.fft.packager import DataWithIndex, Packager
from under_construction.fft.windower import Windower
from scipy import signal


class Spectrogram(Hardware):
    """ The gain of main/model_main wont match"""
    def __init__(self, nfft, window_type='hanning', decimate_by=2):
        self.decimate_by = decimate_by
        self.nfft = nfft
        self.window_type = window_type

        # components
        self.pack = Packager(self.nfft)
        self.windower = Windower(nfft, self.window_type)
        self.fft = R2SDF(nfft)
        self.bitshift = BitReversal(nfft, fftshift=True)
        self.abs = ConjMult()
        self.avg_decimate = AvgDecimate(decimate_by)

        self.DELAY = self.pack.DELAY + self.windower.DELAY + self.fft.DELAY + self.bitshift.DELAY + self.abs.DELAY + self.avg_decimate.DELAY

    def main(self, x):
        pack_out = self.pack.main(x)
        window_out = self.windower.main(pack_out)
        fft_out = self.fft.main(window_out)
        bitshift_out = self.bitshift.main(fft_out)
        mag_out = self.abs.main(bitshift_out)
        dec_out = self.avg_decimate.main(mag_out)
        return dec_out

    def model_main(self, x):
        _, _, spectro_out = signal.spectrogram(x, 1, nperseg=self.nfft, return_onesided=False, detrend=False,
                                               noverlap=0, window='hanning')

        # fftshift
        shifted = np.roll(spectro_out, self.nfft // 2, axis=0)

        # # avg decimation
        l = np.split(shifted, len(shifted) // self.decimate_by)
        golden_output = np.average(l, axis=1).T

        return golden_output


def test_simple():
    fft_size=128
    dut = Spectrogram(fft_size)

    packets = 2
    inp = np.random.uniform(-1, 1, fft_size * packets) + np.random.uniform(-1, 1, fft_size * packets) * 1j
    inp *= 0.25

    sims = simulate(dut, inp,
                    output_callback=DataWithIndex.unpack,
                    simulations=['MODEL', 'PYHA',
                                                                                 # 'RTL'
                                                                                 ])

    sims['MODEL'] = np.array(sims['MODEL']) / np.array(sims['MODEL']).max()
    sims['PYHA'] = np.array(sims['PYHA']) / np.array(sims['PYHA']).max()
    assert sims_close(sims, rtol=1e-1, atol=1e-4)
