import numpy as np


def toggle_bit_reverse(ffts):
    def bit_reverse(x, n_bits):
        return int(np.binary_repr(x, n_bits)[::-1], 2)
    ffts = np.array(ffts)

    fft_size = ffts.shape[-1]
    bit_reversed_indexes = [bit_reverse(i, int(np.log2(fft_size))) for i in range(fft_size)]
    if len(ffts.shape) == 2:
        reversed_ffts = ffts[:,bit_reversed_indexes]
    else:
        reversed_ffts = ffts[bit_reversed_indexes]

    return reversed_ffts

