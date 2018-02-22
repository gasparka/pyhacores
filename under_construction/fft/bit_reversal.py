from pyha import Hardware, simulate, sims_close


class BitReversalNode(Hardware):
    def __init__(self, size):
        self.shr = [0] * size

    def main(self, data, control):
        if control:
            self.shr = [data] + self.shr[:-1]
            return self.shr[-1]
        else:
            self.shr = [self.shr[-1]] + self.shr[:-1]
            return data


class BitReversal(Hardware):

    def __init__(self):
        self.node0 = BitReversalNode(7)
        self.node1 = BitReversalNode(2)
        # self.DELAY = 7

    def main(self, data, control):
        # c = bool(control & 8) != bool(control & 1)
        c = (not bool(control & 8)) or bool(control & 1)
        out = self.node0.main(data, c)
        # c = bool(control & 4) != bool(control & 2)
        # control -= 7
        c = (not bool(control & 4)) or bool(control & 2)
        out2 = self.node1.main(out, c)
        return out, out2

def test_bit_reversal():
    # inp = [0, 1, 2, 3]
    # expect = [0, 2, 1, 3]
    data_in = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15] * 2
    index_in = list(range(32))


    dut = BitReversal()
    sims = simulate(dut, data_in, data_in, simulations=['PYHA'])

    import matplotlib.pyplot as plt
    # plt.plot(sims['MODEL'])
    # plt.plot(sims['PYHA'])
    # plt.show()
    print("\n\n\n", sims['PYHA'][0])
    print(sims['PYHA'][1])
    # assert sims_close(sims)
