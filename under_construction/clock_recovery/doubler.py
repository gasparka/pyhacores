from pyha.common.hwsim import HW

from pyhacores.under_construction.interpolator.model import Interpolator


class GardnerTimingRecovery(HW):
    def __init__(self, sps, test_inject_error=None):
        assert sps == 4
        self.sps = sps
        self.interpolator = Interpolator()


        self.counter = 0
        self.mu = 0.0
        self.e = 0.0


    def model_main(self, xlist):
        err_debug = []
        ret = []
        mu_debug = []

        counter = 0
        mu = 0.0

        delay = [0.0] * (self.sps + 1)

        sample_shr = []
        intepolator0_shr = [0.0] * 3
        sample_now = False
        for sample in xlist:
            if counter == self.sps//2:
                intepolator0_shr = [self.interpolator.filter(sample_shr[0], mu)] + intepolator0_shr[:-1]

                delay = [sample] + delay[:-1]

                sample_now ^= 1
                counter = 0
                # counter += cdelay[-1]
                cdelay = [0] + cdelay[:-1]
                if sample_now:
                    previous = delay[2]
                    middle = delay[1]
                    current = delay[0]
                    # print(f'({current:.2f} - {previous:.2f}) * {middle:.2f}')

                    if skip_error_update:
                        skip_error_update = False
                    else:
                        e = (current - previous) * middle

                    if self.test_inject_error is not None:
                        mu = mu + self.test_inject_error
                    else:
                        mu = mu + e / 4

                    if mu > 1.0:
                        # skip_error_update = True
                        print('>')
                        # mu = 1.0
                        mu = mu % 1
                        counter += 1
                        cdelay[0] = 1
                        # sample_now ^= 1
                    elif mu < 0.0:
                        skip_error_update = True
                        print('<')
                        mu = mu % 1
                        counter -= 1

                    mu_debug.append(mu)
                    err_debug.append(e)
                    ret.append(current)
            counter += 1

        return ret, err_debug, mu_debug