from pyha.common.hwsim import HW
from pyha.common.sfix import resize, Sfix
from pyhacores.moving_average.model import MovingAverage


class DCRemoval(HW):
    def __init__(self, window_len):
        self.mavg = [MovingAverage(window_len) for _ in range(2)]
        self.delay_x = Sfix(0, 0, -17)
        self.delay_x2 = Sfix(0, 0, -17)
        self.out = Sfix(0, 0, -17)

        self._delay = 3

    def main(self, x):

        # run sample trough all moving averages
        tmp = x
        for mav in self.mavg:
            tmp = mav.main(tmp)

        self.next.delay_x = x
        self.next.delay_x2 = self.delay_x
        self.next.out = self.delay_x2 - tmp
        return self.out

    def model_main(self, x):
        tmp = x
        for mav in self.mavg:
            tmp = mav.model_main(tmp)

        # for long moving average this could be equal to:
        # x - np.mean(x)

        return x - tmp
