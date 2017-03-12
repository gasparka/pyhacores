from pathlib import Path

import numpy as np

from pyha.common.sfix import Sfix
from pyha.simulation.simulation_interface import assert_sim_match

from pyhacores.adapters.bladerf.model import ComplexSource, FloatSink


def _load_file(file_name):
    path = Path(__file__).parent / file_name
    return np.load(str(path))


def test_to_complex():
    c = _load_file('data/signaltap_balderf_iq.npy')

    dut = ComplexSource()
    assert_sim_match(dut,
                     None, c.real, c.imag,
                     types= [Sfix(left=0, right=-15)] * 2,
                     rtol=1e-9,
                     atol=1e-9)


def test_to_complex_low_power_bug():
    """ RTL sim did not work if inputs low power """
    c = _load_file('data/low_power.npy')

    dut = ComplexSource()
    assert_sim_match(dut,
                     None, c.real, c.imag,
                     types=[Sfix(left=0, right=-15)] * 2,
                     rtol=1e-9,
                     atol=1e-9)


def test_normal_to_blade():
    input = np.random.rand(128) * 2 - 1
    dut = FloatSink()
    assert_sim_match(dut, None, input)
