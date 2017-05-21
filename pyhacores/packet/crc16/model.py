from pyha.common.const import Const
from pyha.common.hwsim import HW


class CRC16(HW):
    """ Calculate 16 bit CRC, galois based """

    def __init__(self, init_galois, xor):
        """
        :param init_galois: initial value for LFSR. **This must be in galois form, many tools report it in fibo mode only**
        :param xor: feedback value
        """
        self.xor = Const(xor)
        # NB! tools generally report fibo init value...need to convert it!
        self.init_galois = Const(init_galois)
        self.lfsr = init_galois

        self._delay = 1

    def main(self, din, reload):
        """
        :param din: bit in
        :param reload: when True, reloads the initial value to LFSR
        :return: current LFSR value
        """
        if reload:
            lfsr = self.init_galois
        else:
            lfsr = self.lfsr
        out = lfsr & 0x8000
        next = ((lfsr << 1) | din) & 0xFFFF
        if out != 0:
            next = next ^ self.xor
        self.lfsr = next
        return self.lfsr
