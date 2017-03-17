from pyha.simulation.simulation_interface import assert_sim_match

from pyhacores.under_construction.fsk_modulator.model import FSKModulator


def test_basic():
    samples_per_symbol = 4
    fs = 300e3
    deviation = 70e3  # deviation from center frequency

    symbols = [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0]

    # apply SPS
    data = []
    for x in symbols:
        data.extend([x] * samples_per_symbol)

    dut = FSKModulator(deviation, fs)

    assert_sim_match(dut, [bool],
                     None, data,
                     rtol=1e-4)