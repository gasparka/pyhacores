from pathlib import Path

import numpy as np
import pyhacores
import pytest
from data import load_iq
from pyha import Hardware, simulate, sims_close, Complex, resize, Sfix
from under_construction.fft.avg_decimate import AvgDecimate
from under_construction.fft.bit_reversal_fftshift import BitReversal
from under_construction.fft.conjmult import ConjMult
from under_construction.fft.fft_core import R2SDF
from under_construction.fft.windower import Windower


class Spectrogram(Hardware):
    def __init__(self, nfft, window_type='hanning', decimate_by=2):
        self.nfft = nfft
        self.window_type = window_type

        # components
        self.windower = Windower(nfft, self.window_type)
        self.fft = R2SDF(nfft)
        self.bit_reversal_fftshift = BitReversal(nfft, fftshift=True)
        self.abs = ConjMult()
        # self.avg_decimate = AvgDecimate(decimate_by)


    def main(self, x):
        window_out = self.windower.main(x)
        fft_out = self.fft.main(window_out)
        bitshift_out = self.bit_reversal_fftshift.main(fft_out)
        mag_out = self.abs.main(bitshift_out)
        return window_out

    def model_main(self, x):

        x = x[:int(len(x) // self.nfft) * self.nfft]

        # divide to chunks of size 'nfft'
        chunks = np.split(x, len(x) // self.nfft)

        # apply window
        windowed = chunks * np.hanning(self.nfft)

        # take fft, this also fixes ordering
        ffts = np.fft.fft(windowed)

        # take magnitude
        mag = np.conjugate(ffts) * ffts
        mag = mag.real

        # transpose
        t = mag.T

        # shift
        shifted = np.roll(t, self.nfft//2, axis=0)

        # decimate by averaging
        # adec =


        return windowed


def test_():
    fft_points = 256
    dut = Spectrogram(nfft=fft_points)

    path = Path(pyhacores.__path__[0]) / '../data/f2404_fs16.896_one_hop.iq'
    # inp = load_iq(str(path))[19000:20000] # this part has only bits..i.e no noisy stuff

    inp = load_iq(str(path))


    from scipy import signal

    inp = signal.decimate(inp, 8)
    print(len(inp))
    print(inp.max())

    # _, _, out = signal.spectrogram(inp, 1, nperseg=fft_points, return_onesided=False, detrend=False,
    #                                noverlap=0, window='hann')
    #
    # out = np.roll(out, fft_points//2, axis=0)


    # sims = simulate(dut, inp, simulations=['MODEL', 'PYHA'])
    #
    # import matplotlib.pyplot as plt
    #
    # plt.imshow(sims['MODEL'], interpolation='nearest', aspect='auto', origin='lower')
    # plt.tight_layout()
    # plt.show()
    #
    # plt.imshow(sims['PYHA'], interpolation='nearest', aspect='auto', origin='lower')
    # plt.tight_layout()
    # plt.show()

    # assert sims_close(sims, rtol=1e-2)