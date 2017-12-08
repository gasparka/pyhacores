import numpy as np
def load_iq(file):
    f = np.fromfile(open(str(file)), dtype=np.complex64)
    return f