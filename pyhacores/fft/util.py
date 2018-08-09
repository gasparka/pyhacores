import numpy as np


def toggle_bit_reverse(ffts, fft_size):
    def bit_reverse(x, n_bits):
        return int(np.binary_repr(x, n_bits)[::-1], 2)
    ffts = np.array(ffts)
    bit_reversed_indexes = [bit_reverse(i, int(np.log2(fft_size))) for i in range(fft_size)]
    reversed_ffts = ffts[bit_reversed_indexes]

    return reversed_ffts


def postprocess(x, fft_size):
    pyh = x
    pyh = pyh[bit_reversed_indexes(fft_size)]
    pyh = np.fft.fftshift(pyh)
    pyh = [complex(x) for x in pyh]
    #     pyh = np.abs(pyh)
    pyh = pyh * np.conjugate(pyh)
    #     pyh = pyh / pyh.max()
    return pyh.real
