from pyha.common.hwsim import HW
from pyha.common.sfix import Sfix
from pyhacores.moving_average.model import MovingAverage


class DCRemoval(HW):
    """
    Based on: https://www.dsprelated.com/showarticle/58.php
    """

    def __init__(self, window_len, averagers):

        self.mavg = [MovingAverage(window_len) for _ in range(averagers)]

        # this is total delay of moving averages
        hardware_delay = averagers * MovingAverage(window_len)._delay
        self.group_delay = int(averagers * MovingAverage(window_len)._group_delay)
        total_delay = hardware_delay +  self.group_delay

        # registers
        self.input_shr = [Sfix()] * total_delay
        self.out = Sfix(0, 0, -17)

        # module delay
        self._delay = total_delay + 1

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
