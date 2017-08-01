from pyha.common.hwsim import HW
from pyha.common.sfix import Sfix
from pyhacores.moving_average.model import MovingAverage
import numpy as np


class DCRemovalSimple(HW):
    """
    Based on: https://www.dsprelated.com/showarticle/58.php
    Change is that the delay is not matched on the subtract operation, since the DC is ~constant it does not matter
    """
    def __init__(self, window_len):
        self.mavg = [MovingAverage(window_len), MovingAverage(window_len),
                     MovingAverage(window_len), MovingAverage(window_len)]
        self.y = Sfix(0, 0, -17)

        self.DELAY = 1

    def main(self, x):
        # run input signal over all the MA's
        tmp = x

        for mav in self.mavg:
            tmp = mav.main(tmp)

        # dc-free signal
        self.y = x - tmp
        return self.y

    def model_main(self, xl):
        tmp = xl
        for mav in self.mavg:
            tmp = mav.model_main(tmp)

        # this actually not quite equal to main, delay issues?
        y = xl - np.array([0, 0, 0] + tmp.tolist()[:-3])
        return y



class DCRemoval(HW):
    """
    Based on: https://www.dsprelated.com/showarticle/58.php
    """

    def __init__(self, window_len, averagers):

        self.mavg = [MovingAverage(window_len) for _ in range(averagers)]

        # this is total delay of moving averages
        hardware_delay = averagers * MovingAverage(window_len).DELAY
        self.group_delay = int(averagers * (window_len-1)/2)
        total_delay = hardware_delay +  self.group_delay

        # registers
        self.input_shr = [Sfix()] * total_delay
        self.out = Sfix(0, 0, -17)

        # module delay
        self.DELAY = total_delay + 1

    def main(self, x):
        # run signal over all moving averagers
        tmp = x
        for mav in self.mavg:
            tmp = mav.main(tmp)

        # subtract from delayed input
        self.input_shr = [x] + self.input_shr[:-1]
        self.out = self.input_shr[-1] - tmp
        return self.out

    def model_main(self, x):
        # run signal over all moving averagers
        tmp = x
        for mav in self.mavg:
            tmp = mav.model_main(tmp)

        # subtract from delayed input
        return x[:-self.group_delay] - tmp[self.group_delay:]
