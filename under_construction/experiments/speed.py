import numpy as np
from pyha.simulation.simulation_interface import assert_sim_match, SIM_HW_MODEL
from scipy import signal

from pyhacores.filter.fir.model import FIR

np.random.seed(2)
taps = signal.remez(128, [0, 0.1, 0.2, 0.5], [1, 0])
dut = FIR(taps)
inp = np.random.uniform(-1, 1, 1024*2*2)

assert_sim_match(dut, None, inp, simulations=[SIM_HW_MODEL], atol=1e-4)


# python -m vmprof --web pyhacores/under_construction/filter/fir/speed.py

# time python pyhacores/under_construction/filter/fir/speed.py
# real    0m43.068s
# user    0m43.012s
# sys     0m0.648s
