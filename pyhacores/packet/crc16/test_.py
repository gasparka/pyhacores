from pyha.common.util import hex_to_bool_list
from pyha.simulation.simulation_interface import assert_sim_match

from pyhacores.packet.crc16.model import CRC16


class TestCRC16:
    def setup(self):
        self.dut = CRC16(init_galois=0x48f9, xor=0x1021)

    def test_simple_one(self):
        data = hex_to_bool_list('8dfc4ff97dffdb11ff438aee29243910365e908970b9475e')
        reload = [False] * len(data)
        model = self.dut.model_main(data, reload)
        assert model[-1] == 0
        assert_sim_match(self.dut, [bool, bool], None, data, reload)

    def test_reset(self):
        data = hex_to_bool_list('A8dfc4ff97dffdb11ff438aee2524391039a4908970b91cdb')
        reload = [False, False, False, False, True] + [False] * 191

        model = self.dut.model_main(data, reload)
        assert model[-1] == 0

        assert_sim_match(self.dut, [bool, bool], None, data, reload)

    def test_reset_two(self):
        data = hex_to_bool_list('8dfc4ff97dffdb11ff438aee2524391039a4908970b91cdb'
                                '8dfc4ff97dffdb11ff438aee2524391039a4908970b91cdb')
        reload = [False] * 192 + [True] + [False] * 191

        model = self.dut.model_main(data, reload)
        assert model[191] == 0
        assert model[-1] == 0

        assert_sim_match(self.dut, [bool, bool], None, data, reload)