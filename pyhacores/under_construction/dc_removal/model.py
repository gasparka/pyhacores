from pyha.common.hwsim import HW
from pyha.common.sfix import Sfix
from pyhacores.moving_average.model import MovingAverage


class DCRemoval(HW):
    def __init__(self, window_len):
        self.mavg = [MovingAverage(window_len) for _ in range(2)]
        self.delay = [Sfix()] * 2 #todo: this is incorrect, should include MAV group delay
        self.out = Sfix(0, 0, -17)

        self._delay = 3

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

        return x - tmp
