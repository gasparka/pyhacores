import numpy as np
import pytest
from pyha import Hardware, simulate, sims_close, Complex, Sfix


class DataIndexValid(Hardware):
    def __init__(self, data, index=0, valid=True):
        self.data = data
        self.index = index
        self.valid = valid


class DataIndexValidPackager:
    def __init__(self, dtype=None, package_size=None):
        self.package_size = package_size
        self.dtype = dtype

    def __call__(self, inputs):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        if self.package_size is None:
            self.package_size = inputs.shape[-1]

        # TODO: throw away samples to fit package_size?

        ret = []
        if isinstance(inputs[0], (list, np.ndarray)):
            for row in inputs:
                ret += [DataIndexValid(self.dtype(elem), i) for i, elem in enumerate(row)]
        else:
            ret += [DataIndexValid(elem, i) for i, elem in enumerate(inputs)]

        return ret


class DataIndexValidDePackager:
    """ Discards invalid samples and turns the stream into 2D array, each row is one package """
    def __call__(self, outputs):
        ret = []
        sublist = []
        for elem in outputs:
            if not elem.valid:  # discard all invalid samples
                continue

            if int(elem.index) == 0:
                if len(sublist):
                    ret.append(sublist)
                sublist = [elem.data]
            else:
                sublist.append(elem.data)

        ret.append(sublist)
        return np.array(ret)


def package(data):
    # ret = []
    # index_bits = np.log2(len(data[0]))
    # for row in data:
    #     ret += [DataWithIndex(elem, Sfix(float(i), index_bits, 0, signed=False)) for i, elem in enumerate(row)]
    #
    # return ret

    ret = []
    if isinstance(data[0], (list, np.ndarray)):
        for row in data:
            ret += [DataIndexValid(elem, i) for i, elem in enumerate(row)]
    else:
        ret += [DataIndexValid(elem, i) for i, elem in enumerate(data)]

    return ret


def unpackage(data):
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


class Packager(Hardware):
    def __init__(self, packet_size):
        self.PACKET_SIZE = packet_size
        self.DELAY = 1

        self.out = DataIndexValid(Complex(), index=self.PACKET_SIZE - 1)

    def main(self, data):
        """
        :type data: Complex
        :rtype: DataIndexValid
        """

        index = (self.out.index + 1) % self.PACKET_SIZE
        self.out = DataIndexValid(data, index, valid=True)

        return self.out

    def model_main(self, inp_list):
        out = np.array(inp_list).reshape((-1, self.PACKET_SIZE))
        return out


@pytest.mark.parametrize("M", [4, 8, 16, 32, 64, 128, 256])
def test_packager(M):
    # M = 1024 * 8
    dut = Packager(M)

    packets = np.random.randint(1, 4)
    inp = np.random.uniform(-1, 1, M * packets) + np.random.uniform(-1, 1, M * packets) * 1j

    sims = simulate(dut, inp,
                    output_callback=unpackage,
                    simulations=['MODEL', 'PYHA',
                                 'RTL',
                                 # 'GATE'
                                 ],
                    conversion_path='/home/gaspar/git/pyha/playground')
    assert sims_close(sims, rtol=1e-2)
