from pathlib import Path

import numpy as np

from pyha.common.sfix import Sfix
from pyha.simulation.simulation_interface import assert_sim_match

from pyhacores.adapters.bladerf.model import Source, Sink


def _load_file(file_name):
    path = Path(__file__).parent / file_name
    return np.load(str(path))


class TestSource:
    def test_basic(self):
        c = _load_file('data/signaltap_balderf_iq.npy')

        dut = Source()
        assert_sim_match(dut,
                         None, c.real, c.imag,
                         types=[Sfix(left=0, right=-15)] * 2,
                         rtol=1e-9,
                         atol=1e-9)

    def test_low_power_bug(self):
        """ RTL sim did not work if inputs low power """
        c = _load_file('data/low_power.npy')

        dut = Source()
        assert_sim_match(dut,
                         None, c.real, c.imag,
                         types=[Sfix(left=0, right=-15)] * 2,
                         rtol=1e-9,
                         atol=1e-9)


class TestSink:
    def test_basic(self):
        input = [0.15 + 0.69j, -0.8+0.2]
        dut = Sink()
        assert_sim_match(dut, None, input)
