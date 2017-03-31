from pyha.common.hwsim import HW
from pyha.common.sfix import Sfix
from pyhacores.moving_average.model import MovingAverage


class DCRemoval(HW):
    """
    Based on: https://www.dsprelated.com/showarticle/58.php
    """
    def __init__(self, window_len):
        self.group_delay = int((window_len-1)/2*2)
        self.mavg = [MovingAverage(window_len) for _ in range(2)]
        self.delay = [Sfix()] * (self.group_delay + 2)
        self.out = Sfix(0, 0, -17)

        self._delay = self.group_delay + 1 + 2

    def main(self, x):

        # run sample trough all moving averages
        tmp = x
        for mav in self.mavg:
            tmp = mav.main(tmp)

        self.next.delay = [x] + self.delay[:-1]
        self.next.out = self.delay[-1] - tmp
        return self.out

    def model_main(self, x):
        tmp = x
        for mav in self.mavg:
            tmp = mav.model_main(tmp)

        # for long moving average this could be equal to:
        # x - np.mean(x)
        return x[:-self.group_delay] - tmp[self.group_delay:]
