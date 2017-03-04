from pyha.common.util import hex_to_bool_list
from pyha.simulation.simulation_interface import assert_sim_match

from pyhacores.packet.header_correlator.model import HeaderCorrelator


class TestHeaderCorrelator:
    def setup(self):
        self.dut = HeaderCorrelator(header=0x8dfc, packet_len=12 * 16)

    def test_one_packet(self):
        inputs = hex_to_bool_list('8dfc4ff97dffdb11ff438aee2524391039a4908970b91cdb')
        expect = [True] + [False] * 191
        assert_sim_match(self.dut, [bool], expect, inputs)

    def test_two_packet(self):
        inputs = hex_to_bool_list('8dfc4ff97dffdb11ff438aee2524391039a4908970b91cdb'
                                  '8dfc4ff97dffdb11ff438aee2524391039a4908970b91cdb')
        expect = [True] + [False] * 191 + [True] + [False] * 191
        assert_sim_match(self.dut, [bool], expect, inputs)