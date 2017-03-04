from pyha.common.sfix import ComplexSfix
from pyha.simulation.simulation_interface import assert_sim_match
from pyhacores.fsk_demodulator.model import FSKDemodulator
import numpy as np


def test_basic():
    # test signal
    from pyha.components.fsk_modulator import FSKModulator
    samples_per_symbol = 4
    symbols = [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0]

    data = []
    for x in symbols:
        data.extend([x] * samples_per_symbol)

    fs = 300e3
    deviation = 70e3
    mod = FSKModulator(deviation, fs)

    tx_signal = mod.model_main(data)

    # awgn channel (add some noise)
    # todo: this random stuff is not performing well
    np.random.seed(1)
    tx_signal = 0.5 * (tx_signal + np.random.normal(scale=np.sqrt(0.1)))

    dut = FSKDemodulator(deviation, fs, samples_per_symbol)

    assert_sim_match(dut, [ComplexSfix(left=0, right=-17)],
                     None, tx_signal,
                     rtol=1e-4,
                     atol=1e-4,
                     skip_first=samples_per_symbol,  # skip first moving average transient
                     )