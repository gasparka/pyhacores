from pyhacores.under_construction.interpolator.model import Interpolator


class GardnerTimingRecovery:
    def __init__(self, sps):
        self.mu = 1.0
        self.sps = sps
        self.interpolator = Interpolator()
        self.out_int = []
        self.sps_counter = 0

    def model_main(self, xlist):
        err_debug = []
        ret = []
        mu_debug = []
        d = 1


        for sample in xlist:
            i_sample = self.interpolator.filter(sample, self.mu)
            self.out_int.append(i_sample)

            self.sps_counter += 1
            if self.sps_counter >= self.sps and len(self.out_int) > 2 * self.sps:
                # if self.mu > 0.1:
                #     d = 4
                # else:
                #     d = 3
                self.sps_counter = 0
                e = (self.out_int[-1 - d] - self.out_int[-self.sps - d]) * self.out_int[-self.sps // 2 - d]
                self.mu -= e/2
                if self.mu < 0.0:
                    self.mu = 0.0
                    d += 1
                    print('d:', d)
                if self.mu > 1.0:
                    self.mu = 0.0
                    d += 1
                    print('d:', d)
                mu_debug.append(self.mu)
                err_debug.append(e)
                ret.append(self.out_int[-1 - d+1])

        return ret, err_debug, mu_debug