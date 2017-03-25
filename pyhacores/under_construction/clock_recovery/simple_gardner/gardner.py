from pyha.common.hwsim import HW
from pyha.common.sfix import Sfix, fixed_truncate, fixed_wrap

from pyhacores.moving_average.model import MovingAverage


class SimpleGardnerTimingRecovery(HW):
    def __init__(self, sps):
        # sps must be divisible by 2 -> required by gardner
        assert not (sps & 1)
        # assert sps >= 8 #
        self.sps = sps

        self.counter = 0
        self.middle_delay = Sfix()
        self.e = Sfix(0.0, 0, -17, round_style=fixed_truncate, overflow_style=fixed_wrap)
        self.cp_diff = Sfix(0.0, 0, -17, round_style=fixed_truncate)
        self.mu = Sfix(0.0, 1, -17, round_style=fixed_truncate, overflow_style=fixed_wrap)
        self.sample_shr = [Sfix()] * self.sps

        self.avg = MovingAverage(4)
        # self._delay = 8

    def main(self, sample):
        avg = Sfix(0.0, 0, -17)
        valid = False
        self.next.sample_shr = [sample] + self.sample_shr[:-1]
        self.next.counter = self.counter + 1
        if self.counter == self.sps - 1:  # -1 because hardware delay already counts for 1 tick
            valid = True
            self.next.counter = 0
            previous = self.sample_shr[self.sps - 1]
            middle = self.sample_shr[self.sps // 2 - 1]
            current = sample

            # pipelined:
            # e = (current - previous) * middle
            # mu = mu + e
            self.next.middle_delay = middle
            self.next.cp_diff = current - previous
            self.next.e = self.cp_diff * self.middle_delay
            avg = self.avg.main(self.e)
            self.next.mu = self.mu + avg

            if self.next.mu > 1.0:
                self.next.mu = 0.0
                self.next.counter = 1
            elif self.next.mu < 0.0:
                self.next.mu = 1.0
                self.next.counter = -1

        return sample, avg, self.mu, valid

    def model_main(self, xlist):
        err_debug = []
        ret = []
        mu_debug = []

        counter = 0
        mu = 0.0

        average = [0.0] * 4
        delay = [0.0] * (self.sps + 1)
        for i, sample in enumerate(xlist):

            delay = [sample] + delay[:-1]
            counter += 1
            if counter == self.sps:
                counter = 0
                previous = delay[self.sps]
                middle = delay[self.sps // 2]
                current = sample

                e = (current - previous) * middle
                average = [e] + average[:-1]
                e = sum(average) / len(average)

                mu = mu + e

                if mu > 1.0:
                    mu = 0.0
                    counter = 1
                elif mu < 0.0:
                    mu = 1.0
                    counter = -1

                mu_debug.append(mu)
                err_debug.append(e)
                ret.append(current)

        return ret, err_debug, mu_debug
