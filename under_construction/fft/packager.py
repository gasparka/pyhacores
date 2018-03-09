import numpy as np
import pytest
from pyha import Hardware, simulate, sims_close, Complex, Sfix


class DataWithIndex(Hardware):
    def __init__(self, data, index=0, valid=True):
        self.data = data
        self.index = index
        self.valid = valid

    @staticmethod
    def _pyha_unpack(data):
        ret = []
        sublist = []
        for elem in data:
            if not elem.valid:
                continue

            if int(elem.index) == 0:
                if len(sublist):
                    ret.append(sublist)
                sublist = [elem.data]
            else:
                sublist.append(elem.data)

        ret.append(sublist)
        return ret

    @staticmethod
    def _pyha_pack(data):
        ret = []
        for row in data:
            ret += [DataWithIndex(elem, i) for i, elem in enumerate(row)]

        return ret


class Packager(Hardware):
    def __init__(self, packet_size):
        self.PACKET_SIZE = packet_size
        self.DELAY = 1

        self.out = DataWithIndex(Complex(), Sfix(self.PACKET_SIZE-1, int(np.log2(self.PACKET_SIZE)), 0, signed=False))

    def main(self, data):
        """
        :type data: Complex
        :rtype: DataWithIndex
        """

        index = (self.out.index + 1) % self.PACKET_SIZE
        self.out = DataWithIndex(data, index, valid=True)

        return self.out

    def model_main(self, inp_list):
        out = np.array(inp_list).reshape((-1, self.PACKET_SIZE))
        return out

# Flow Status	Successful - Fri Mar  9 19:05:36 2018
# Quartus Prime Version	17.1.0 Build 590 10/25/2017 SJ Lite Edition
# Revision Name	quartus_project
# Top-level Entity Name	top
# Family	Cyclone IV E
# Device	EP4CE40F23C8
# Timing Models	Final
# Total logic elements	118 / 39,600 ( < 1 % )
# Total registers	100
# Total pins	107 / 329 ( 33 % )
# Total virtual pins	0
# Total memory bits	0 / 1,161,216 ( 0 % )
# Embedded Multiplier 9-bit elements	0 / 232 ( 0 % )
# Total PLLs	0 / 4 ( 0 % )

@pytest.mark.parametrize("M", [4, 8, 16, 32, 64, 128, 256])
def test_packager(M):
    M = 1024 * 8
    dut = Packager(M)

    packets = np.random.randint(1, 4)
    inp = np.random.uniform(-1, 1, M * packets) + np.random.uniform(-1, 1, M * packets) * 1j

    sims = simulate(dut, inp,
                    output_callback=DataWithIndex._pyha_unpack,
                    simulations=['MODEL', 'PYHA',
                                                                                       # 'RTL',
                                                                                       'GATE'
                                                                                        ],
                    conversion_path='/home/gaspar/git/pyha/playground')
    assert sims_close(sims, rtol=1e-2)
