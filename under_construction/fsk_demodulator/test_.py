import numpy as np
from pyha.common.sfix import Complex
from pyha.simulation.simulation_interface import assert_sim_match, debug_assert_sim_match, SIM_MODEL, SIM_HW_MODEL

from pyhacores.under_construction.fsk_demodulator.model import FSKDemodulator
from pyhacores.under_construction.fsk_modulator.model import FSKModulator


class TestFSKDemodulator:
    pass


def input_signal():
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
    return deviation, fs, samples_per_symbol, tx_signal


def test_basic():
    # test signal
    deviation, fs, samples_per_symbol, tx_signal = input_signal()

    dut = FSKDemodulator(deviation, fs, samples_per_symbol)

    assert_sim_match(dut,
                     None, tx_signal,
                     atol=1e-4,
                     skip_first=samples_per_symbol,  # skip first moving average transient
                     dir_path='/home/gaspar/git/pyhacores/playground'
                     )



def test_quad_demod():
    deviation, fs, samples_per_symbol, tx_signal = input_signal()

    dut = FSKDemodulator(deviation, fs, samples_per_symbol)

    r = debug_assert_sim_match(dut, None, tx_signal,
                               simulations=[SIM_MODEL, SIM_HW_MODEL]
                     )

    import matplotlib.pyplot as plt
    plt.plot(dut.demod._outputs)

    dm =dut.demod.model_main(tx_signal)
    plt.plot(dm)

    plt.show()
    pass

