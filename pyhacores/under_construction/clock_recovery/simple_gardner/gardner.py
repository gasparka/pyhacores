from pyha.common.hwsim import HW


class SimpleGardnerTimingRecovery(HW):
    def __init__(self, sps, test_inject_error=None):
        # sps must be divisible by 2 -> required by gardner
        assert not (sps & 1)
        self.test_inject_error = test_inject_error
        self.sps = sps

        self.counter = 0
        self.e = 0
        self.mu = 0
        self.sample_shr = [0.0] * (self.sps+1)

    def main(self, sample):
        sample = float(sample)

        valid = False
        self.next.sample_shr = [sample] + self.sample_shr[:-1]
        self.next.counter = self.counter + 1
        if self.next.counter == self.sps:
            valid = True
            self.next.counter = 0
            previous = self.next.sample_shr[self.sps]
            middle = self.next.sample_shr[self.sps // 2]
            current = self.next.sample_shr[0]

            self.next.e = (current - previous) * middle
            self.next.mu = self.next.mu + self.next.e

            if self.next.mu > 1.0:
                self.next.mu = 0.0
                self.next.counter = 1
            elif self.next.mu < 0.0:
                self.next.mu = 1.0
                self.next.counter = -1


        return self.next.sample_shr[0], self.next.e, self.next.mu, valid


    def model_main(self, xlist):
        err_debug = []
        ret = []
        mu_debug = []

        counter = 0
        mu = 0.0

        delay = [0.0] * (self.sps + 1)
        for sample in xlist:

            delay = [sample] + delay[:-1]
            counter += 1
            if counter == self.sps:
                counter = 0
                previous = delay[self.sps]
                middle = delay[self.sps // 2]
                current = delay[0]

                e = (current - previous) * middle
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
