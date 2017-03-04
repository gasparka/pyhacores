from pyha.common.const import Const
from pyha.common.hwsim import HW
from pyha.common.util import hex_to_bool_list


class HeaderCorrelator(HW):
    """ Correlate againtst 16 bit header
    Once header is found, 'packet_len' bits are skipped before next header can be correlated!
    """
    def __init__(self, header, packet_len):
        """

        :param header: 16 bit header
        :param packet_len: this is used as a cooldown, to not discover packets inside packets
        """

        self.cooldown_reset = Const(packet_len - 1)
        self.header = Const(hex_to_bool_list(header))

        self.cooldown = 0
        self.shr = [False] * 16

        self._delay = 16

    def main(self, din):
        """

        :param din: bit in
        :return: True if 100% correlation
        """
        self.next.shr = self.shr[1:] + [din]
        ret = False
        if self.cooldown == 0:
            if self.shr == self.header:
                self.next.cooldown = self.cooldown_reset
                ret = True
        else:
            self.next.cooldown = self.next.cooldown - 1
        return ret

    def model_main(self, data):
        rets = [False] * len(data)
        i = 0
        while i < len(data):
            word = data[i:i + 16]
            if word == self.header:
                rets[i] = True
                i += self.cooldown_reset
            i += 1
        return rets