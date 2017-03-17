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
        self.next.lfsr = ((lfsr << 1) | din) & 0xFFFF
        if out != 0:
            self.next.lfsr = self.next.lfsr ^ self.xor
        return self.lfsr

    def model_main(self, data, reload):
        ret = []
        lfsr = self.init_galois
        for din, rl in zip(data, reload):
            if rl:
                lfsr = self.init_galois

            out = lfsr & 0x8000
            lfsr = ((lfsr << 1) | din) & 0xFFFF
            if out:
                lfsr ^= self.xor
            ret.append(lfsr)
        return ret
