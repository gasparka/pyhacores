import pickle

import numpy as np
from data import load_iq

from pyha import Hardware, simulate, hardware_sims_equal, sims_close, Sfix
from pyhacores.filter import DCRemoval
from under_construction.fft.bitreversal_fftshift_decimate import BitreversalFFTshiftDecimate
from under_construction.fft.conjmult import ConjMult
from under_construction.fft.fft_core import R2SDF
from under_construction.fft.packager import Packager, unpackage, DataWithIndex
from under_construction.fft.windower import Windower
from scipy import signal


class Spectrogram(Hardware):
    """ The gain of main/model_main wont match"""

    def __init__(self, nfft, avg_freq_axis=2, avg_time_axis=1, window_type='hanning', fft_twiddle_bits=18, window_bits=18):
        self.DECIMATE_BY = avg_freq_axis
        self.NFFT = nfft
        self.WINDOW_TYPE = window_type

        # components
        self.dc_removal = DCRemoval(256)
        self.pack = Packager(self.NFFT)
        self.windower = Windower(nfft, self.WINDOW_TYPE, coefficient_bits=window_bits)
        self.fft = R2SDF(nfft, twiddle_bits=fft_twiddle_bits)
        self.abs = ConjMult()
        self.dec = BitreversalFFTshiftDecimate(nfft, avg_freq_axis, avg_time_axis)

        self.DELAY = self.dc_removal.DELAY + self.pack.DELAY + self.windower.DELAY + self.fft.DELAY + self.abs.DELAY + self.dec.DELAY

    def main(self, x):
        dc_out = self.dc_removal.main(x)
        pack_out = self.pack.main(dc_out)
        window_out = self.windower.main(pack_out)
        fft_out = self.fft.main(window_out)
        mag_out = self.abs.main(fft_out)
        dec_out = self.dec.main(mag_out)
        return dec_out

    def model_main(self, x):
        _, _, spectro_out = signal.spectrogram(x, 1, nperseg=self.NFFT, return_onesided=False, detrend=False,
                                               noverlap=0, window='hanning')

        # fftshift
        shifted = np.roll(spectro_out, self.NFFT // 2, axis=0)

        # # avg decimation
        l = np.split(shifted, len(shifted) // self.DECIMATE_BY)
        golden_output = np.average(l, axis=1).T

        return golden_output


def test_simple():
    # TWID: 10b, WINDOW: 8b
    # Flow Status	Successful - Wed Jun 20 12:38:29 2018
    # Quartus Prime Version	17.1.0 Build 590 10/25/2017 SJ Lite Edition
    # Revision Name	quartus_project
    # Top-level Entity Name	top
    # Family	Cyclone IV E
    # Device	EP4CE40F23C8
    # Timing Models	Final
    # Total logic elements	11,729 / 39,600 ( 30 % )
    # Total registers	955
    # Total pins	107 / 329 ( 33 % )
    # Total virtual pins	0
    # Total memory bits	312,408 / 1,161,216 ( 27 % )
    # Embedded Multiplier 9-bit elements	104 / 232 ( 45 % )
    # Total PLLs	0 / 4 ( 0 % )

    # NO DC REMOVAL
    # TWID: 9b, WINDOW: 8b
    # Family	Cyclone IV E
    # Device	EP4CE40F23C8
    # Timing Models	Final
    # Total logic elements	10,146 / 39,600 ( 26 % )
    # Total registers	955
    # Total pins	107 / 329 ( 33 % )
    # Total virtual pins	0
    # Total memory bits	312,408 / 1,161,216 ( 27 % )
    # Embedded Multiplier 9-bit elements	104 / 232 ( 45 % )
    # Total PLLs	0 / 4 ( 0 % )

    # WITH DC REMOVAL + PIPELINED FFT, FMAX 80M
    # TWID: 9b, WINDOW: 8b
    # Device	EP4CE40F23C8
    # Timing Models	Final
    # Total logic elements	10,206 / 39,600 ( 26 % )
    # Total registers	2782
    # Total pins	107 / 329 ( 33 % )
    # Total virtual pins	0
    # Total memory bits	349,179 / 1,161,216 ( 30 % )
    # Embedded Multiplier 9-bit elements	104 / 232 ( 45 % )
    # Total PLLs	0 / 4 ( 0 % )


    np.random.seed(0)
    fft_size = 1024 * 8
    avg_time_axis = 4

    dut = Spectrogram(fft_size, avg_freq_axis=32, avg_time_axis=avg_time_axis, fft_twiddle_bits=9, window_bits=8)

    packets = avg_time_axis
    inp = np.random.uniform(-1, 1, fft_size * packets) + np.random.uniform(-1, 1, fft_size * packets) * 1j
    inp *= 0.5 * 0.001

    sims = simulate(dut, inp,
                    output_callback=unpackage,
                    simulations=['MODEL', 'PYHA',
                                 'GATE',
                                 # 'RTL'
                                 ],
                    conversion_path='/home/gaspar/git/pyhacores/playground')

    # import matplotlib.pyplot as plt
    # plt.plot(np.hstack(sims['MODEL']))
    # plt.plot(np.hstack(sims['PYHA']))
    # plt.plot(np.hstack(sims['RTL']))
    # plt.show()

    sims['MODEL'] = np.array(sims['MODEL']) / np.array(sims['MODEL']).max()
    sims['PYHA'] = np.array(sims['PYHA']) / np.array(sims['PYHA']).max()
    # sims['RTL'] = np.array(sims['RTL']) / np.array(sims['RTL']).max()
    # sims['GATE'] = np.array(sims['GATE']) / np.array(sims['GATE']).max()
    assert sims_close(sims, rtol=1e-1, atol=1e-4)


def test_realsig():
    # file = '/run/media/gaspar/maxtor/measurement 13.03.2018/mavic_tele/qdetector_20180313122024455464_far_10m_regular/1520936452.2426_fs=20000000.0_bw=20000000.0_fc=2431000000.0_d=0_g=033000.raw'
    file = '/home/gaspar/Documents/low_power_ph3.raw'
    fft_size = 1024 * 2 * 2 * 2
    decimation = 32
    print(file)

    iq = load_iq(file)
    iq = iq[:len(iq) // 8]

    dut = Spectrogram(fft_size, decimation)
    sims = simulate(dut, iq, simulations=['MODEL', 'PYHA'],
                    output_callback=unpackage)

    with open(f'{file}_spectro_TST.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(sims, f, pickle.HIGHEST_PROTOCOL)

def test_realsig2():
    file = '/run/media/gaspar/maxtor/measurement 13.03.2018/mavic_tele/qdetector_20180313120601081997_noremote_medium_10m/1520935599.0396_fs=20000000.0_bw=20000000.0_fc=2410000000.0_d=3_g=063015.raw'

    fft_size = 1024 * 2 * 2 * 2
    decimation = 32
    print(file)

    iq = load_iq(file)
    # iq = iq[:len(iq) // 8]

    dut = Spectrogram(fft_size, decimation)
    sims = simulate(dut, iq, simulations=['MODEL', 'PYHA'],
                    output_callback=unpackage)

    with open(f'{file}_spectro.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(sims, f, pickle.HIGHEST_PROTOCOL)



def test_real_life():
    fft_size = 1024 * 8
    decimation = 32
    dut = Spectrogram(fft_size, decimation)

    file = '/run/media/gaspar/maxtor/measurement 13.03.2018/mavic_tele/qdetector_20180313122024455464_far_10m_regular/1520936452.2426_fs=20000000.0_bw=20000000.0_fc=2431000000.0_d=0_g=033000.raw'

    orig_inp = load_iq(file)

    orig_inp = orig_inp[:len(orig_inp) // (1024 * 4)]
    # orig_inp /= orig_inp.max() * 2

    # orig_inp = np.random.uniform(-1, 1, fft_size * 1) + np.random.uniform(-1, 1, fft_size * 1) * 1j
    # orig_inp *= 0.5

    sims = simulate(dut, orig_inp,
                    output_callback=unpackage,
                    simulations=['MODEL', 'PYHA',
                                 # 'RTL',
                                 # 'GATE'
                                 ],
                    conversion_path='/home/gaspar/git/pyhacores/playground')

    # import matplotlib.pyplot as plt
    # plt.plot(np.hstack(sims['MODEL']))
    # plt.plot(np.hstack(sims['PYHA']))
    # plt.plot(np.hstack(sims['RTL']))
    # plt.show()

    sims['MODEL'] = np.array(sims['MODEL']) / np.array(sims['MODEL']).max()
    sims['PYHA'] = np.array(sims['PYHA']) / np.array(sims['PYHA']).max()
    # sims['RTL'] = np.array(sims['RTL']) / np.array(sims['RTL']).max()
    # sims['GATE'] = np.array(sims['GATE']) / np.array(sims['GATE']).max()
    assert hardware_sims_equal(sims)
    assert sims_close(sims, rtol=1e-1, atol=1e-4)
