from math import floor

import pytest

from pyhacores.under_construction.interpolator.model import Interpolator


class GardnerTimingRecovery:
    def __init__(self, sps):
        # sps must be divisible by 2 -> required by gardner
        assert sps in [2, 4]
        self.d = sps - 1
        self.sps = sps
        self.interpolator = Interpolator()
        self.out_int = [0.0] * self.sps
        self.sps_counter = 0

    def model_main(self, xlist):
        err_debug = []
        mu_debug = []
        ret = []

        for sample in xlist:
            i_sample = self.interpolator.filter(sample, self.d % 1)
            self.out_int.append(i_sample)

            self.sps_counter += 1
            if self.sps_counter >= self.sps:
                self.sps_counter = 0

                intd = int(floor(self.d))
                e = (self.out_int[-1 - intd] - self.out_int[-self.sps - intd]) * self.out_int[-self.sps // 2 - intd]
                self.d += e / 4

                mu_debug.append(self.d)
                err_debug.append(e)
                ret.append(self.out_int[-1 - intd + 1])

        return ret, err_debug, mu_debug

        # def model_main(self, xlist):
        #     err_debug = []
        #     ret = []
        #     mu_debug = []
        #     d = self.sps - 1
        #
        #     for sample in xlist:
        #         i_sample = self.interpolator.filter(sample, self.mu)
        #         self.out_int.append(i_sample)
        #
        #         self.sps_counter += 1
        #         if self.sps_counter >= self.sps:
        #             self.sps_counter = 0
        #             e = (self.out_int[-1 - d] - self.out_int[-self.sps - d]) * self.out_int[-self.sps // 2 - d]
        #             self.mu += e / 4
        #             if self.mu < 0.0:
        #                 # pytest.xfail()
        #                 tmp = self.mu
        #                 # self.mu = self.mu + 1.0
        #                 self.mu = 0.5
        #                 d -= 1
        #                 print(f'<d:{d} mu_in:{tmp:.2f} mu{self.mu:.2f}')
        #             if self.mu > 1.0:
        #                 # self.mu = 0.0
        #                 tmp = self.mu
        #                 # self.mu = self.mu - 1.0
        #                 self.mu = 0.5
        #                 d += 1
        #                 print(f'>d:{d} mu_in:{tmp:.2f} mu{self.mu:.2f}')
        #
        #             mu_debug.append(self.mu)
        #             err_debug.append(e)
        #             ret.append(self.out_int[-1 - d + 1])
        #
        #
        #
        #     return ret, err_debug, mu_debug
