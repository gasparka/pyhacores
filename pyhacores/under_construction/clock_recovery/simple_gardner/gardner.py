
class SimpleGardnerTimingRecovery:
    def __init__(self, sps, test_inject_error=None):
        # sps must be divisible by 2 -> required by gardner
        assert not (sps & 1)
        self.test_inject_error = test_inject_error
        self.sps = sps

    def model_main(self, xlist):
        err_debug = []
        ret = []
        mu_debug = []

        counter = 0
        mu = 0.0

        delay = [0.0] * (self.sps + 1)
        for sample in xlist:
            delay = [sample] + delay[:-1]
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

            counter += 1

        return ret, err_debug, mu_debug
