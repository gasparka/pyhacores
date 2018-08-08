import numpy as np
import pytest

from pyhacores.fft.util import toggle_bit_reverse


def W(k, N, inverse=False):
    """ e^-j*2*PI*k*n/N, argument k = k * n """
    r = np.exp(-1j * (2 * np.pi / N) * k)
    if inverse:
        return np.conjugate(r)
    return r


def pyfft_natural(inp, fft_size):
    """ Inputs natural, outputs bitreversed"""
    out = np.array([0. + 0.j] * fft_size)

    sample_offset = fft_size//2
    for i in range(fft_size//2):
        out[i] = inp[i] + inp[i + sample_offset]
        out[i+sample_offset] = (inp[i] - inp[i + sample_offset]) * W(i, fft_size)

    if sample_offset != 1:
        first = pyfft_natural(out[:sample_offset], sample_offset)
        second = pyfft_natural(out[sample_offset:], sample_offset)

        out = np.hstack([first, second])
    return out


def butterfly(a, b, twiddle):
    aa = a + b
    bb = (a - b) * twiddle
    return aa, bb

def pyfft_rev(inp, fft_size):
    out = np.array([0. + 0.j] * fft_size)
    if fft_size == 2:
        out[0], out[1] = butterfly(inp[0], inp[1], W(0, 2))
    elif fft_size == 4:
        out0 = np.array([0. + 0.j] * 4)
        out0[0], out0[1] = butterfly(inp[0], inp[1], W(0, 4))
        out0[2], out0[3] = butterfly(inp[2], inp[3], W(1, 4))

        out1 = np.array([0. + 0.j] * 4)
        out1[0], out1[2] = butterfly(out0[0], out0[2], W(0, 2))
        out1[1], out1[3] = butterfly(out0[1], out0[3], W(0, 2))
        return out1

    elif fft_size == 8:
        out0 = np.array([0. + 0.j] * 8)
        out0[0], out0[1] = butterfly(inp[0], inp[1], W(0, 8))
        out0[2], out0[3] = butterfly(inp[2], inp[3], W(2, 8))
        out0[4], out0[5] = butterfly(inp[4], inp[5], W(1, 8))
        out0[6], out0[7] = butterfly(inp[6], inp[7], W(3, 8))

        out1 = np.array([0. + 0.j] * 8)
        out1[0], out1[2] = butterfly(out0[0], out0[2], W(0, 4))
        out1[1], out1[3] = butterfly(out0[1], out0[3], W(0, 4))
        out1[4], out1[6] = butterfly(out0[4], out0[6], W(1, 4))
        out1[5], out1[7] = butterfly(out0[5], out0[7], W(1, 4))

        out2 = np.array([0. + 0.j] * 8)
        out2[0], out2[4] = butterfly(out1[0], out1[4], W(0, 2))
        out2[1], out2[5] = butterfly(out1[1], out1[5], W(0, 2))
        out2[2], out2[6] = butterfly(out1[2], out1[6], W(0, 2))
        out2[3], out2[7] = butterfly(out1[3], out1[7], W(0, 2))
        return out2

    return out


@pytest.mark.parametrize("fft_size", [2, 4, 8, 16, 32, 64, 128])
def test_pyfft(fft_size):
    input_signal = np.random.uniform(-1, 1, fft_size) + np.random.uniform(-1, 1, fft_size) * 1j
    my = pyfft_natural(input_signal, fft_size)
    ref = np.fft.fft(input_signal, fft_size)
    ref = toggle_bit_reverse(ref, fft_size)

    np.testing.assert_allclose(ref, my)



@pytest.mark.parametrize("fft_size", [2, 4, 8])
def test_rev(fft_size):
    input_signal = np.random.uniform(-1, 1, fft_size) + np.random.uniform(-1, 1, fft_size) * 1j
    my = pyfft_rev(input_signal, fft_size)
    input_signal = toggle_bit_reverse(input_signal, fft_size)
    ref = np.fft.fft(input_signal, fft_size)

    np.testing.assert_allclose(ref, my)

