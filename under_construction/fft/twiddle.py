from pyha import Hardware, Complex, simulate, sims_close, resize
import numpy as np


def W(k, N):
    """ e^-j*2*PI*k*n/N, argument k = k * n """
    return np.exp(-1j * (2 * np.pi / N) * k)


class Twiddle(Hardware):
    def __init__(self, fft_size, twiddle_bits=18, mode_simple=True):
        self.MODE_SIMPLE = mode_simple
        self.FFT_SIZE = fft_size
        self.CONTROL_MASK = (fft_size // 4 - 1)
        self.CONTROL_MASK2 = (fft_size // 2 - 1)

        if self.MODE_SIMPLE:
            self.TWIDDLES = [
                Complex(W(i, fft_size), 0, -(twiddle_bits - 1), overflow_style='saturate', round_style='round')
                for i in range(fft_size // 2)]
        else:
            self.TWIDDLES = [
                Complex(W(i, fft_size), 0, -(twiddle_bits - 1), overflow_style='saturate', round_style='round')
                for i in range(fft_size // 8 + 1)]

            self.BANK_SPLIT = fft_size // 8 + 1
            self.BANK_SPLIT2 = fft_size // 4
            self.BANK_SPLIT3 = fft_size // 4 + fft_size // 8 + 1
            self.SECOND_BANK_OFFSET = self.CONTROL_MASK

    def main_simple(self, control):
        return self.TWIDDLES[control & self.CONTROL_MASK]

    def main_complex(self, control):
        addr = control & self.CONTROL_MASK
        alt_addr = self.SECOND_BANK_OFFSET - addr + 1
        if (control & self.CONTROL_MASK2) < self.BANK_SPLIT:
            use_alt_addr = False
            swap = False
            negate_real = False
            negate_imag = False
        elif (control & self.CONTROL_MASK2) < self.BANK_SPLIT2:
            use_alt_addr = True
            swap = True
            negate_real = True
            negate_imag = True
        elif (control & self.CONTROL_MASK2) < self.BANK_SPLIT3:
            use_alt_addr = False
            swap = True
            negate_real = False
            negate_imag = True
        else:
            use_alt_addr = True
            swap = False
            negate_real = True
            negate_imag = False

        dir = addr
        if use_alt_addr:
            dir = alt_addr

        twid = self.TWIDDLES[dir]
        if swap:
            real = twid.imag
            imag = twid.real
        else:
            real = twid.real
            imag = twid.imag

        if negate_real:
            real = resize(-real, real)

        if negate_imag:
            imag = resize(-imag, imag)

        return Complex(real, imag)

    def main(self, control):
        if self.MODE_SIMPLE:
            return self.main_simple(control)
        else:
            return self.main_complex(control)

    def model_main(self, i):
        return [W(i, self.FFT_SIZE) for i in range(self.FFT_SIZE // 2)]


def test_basic():
    # 18b 8k    4500    2342
    # 12b 8k    1996    1077
    fft_size = 1024 * 2 * 2 * 2
    dut = Twiddle(fft_size, twiddle_bits=12, mode_simple=False)
    inp = list(range(fft_size // 2))
    ret = simulate(dut, inp, simulations=['MODEL', 'PYHA', 'RTL', 'GATE'],
                   conversion_path='/home/gaspar/git/pyhacores/playground')

    # assert sims_close(ret)
