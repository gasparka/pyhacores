import numpy as np
import pytest
from pyha import Hardware, simulate, sims_close, Complex


class DataWithIndex(Hardware):
    def __init__(self, data, index, valid=True):
        self.data = data
        self.index = index
        self.valid = valid

    @staticmethod
    def unpack(data):
        ret = []
        sublist = []
        for elem in data:
            if not elem.valid:
                continue

            if elem.index == 0:
                if len(sublist):
                    ret.append(sublist)
                sublist = [elem.data]
            else:
                sublist.append(elem.data)

        ret.append(sublist)
        return ret

    @staticmethod
    def pack(data):
        ret = []
        for row in data:
            ret += [DataWithIndex(elem, i) for i, elem in enumerate(row)]

        return ret


class Packager(Hardware):
    def __init__(self, packet_size):
        self.PACKET_SIZE = packet_size
        self.counter = 0

        self.out = DataWithIndex(Complex(), 0)
        self.DELAY = 1

    def main(self, data):
        """
        :type data: Complex
        :rtype: DataWithIndex
        """

        self.out = DataWithIndex(data, index=self.counter, valid=True)

        next_counter = self.counter + 1
        if next_counter >= self.PACKET_SIZE:
            next_counter = 0

        self.counter = next_counter

        return self.out

    def model_main(self, inp_list):
        out = np.array(inp_list).reshape((-1, self.PACKET_SIZE))
        return out


@pytest.mark.parametrize("M", [4, 8, 16, 32, 64, 128, 256])
def test_packager(M):
    dut = Packager(M)

    packets = np.random.randint(1, 4)
    inp = np.random.uniform(-1, 1, M * packets) + np.random.uniform(-1, 1, M * packets) * 1j

    sims = simulate(dut, inp, output_callback=DataWithIndex.unpack, simulations=['MODEL', 'PYHA',
                                                                                 # 'RTL'
                                                                                 ])
    assert sims_close(sims, rtol=1e-2)
