from math import floor
import numpy as np

import pytest
from scipy.interpolate import interp1d

from pyhacores.under_construction.interpolator.model import Interpolator
import matplotlib.pyplot as plt

class GardnerTimingRecovery:
    def __init__(self, sps):
        # sps must be divisible by 2 -> required by gardner
        assert sps in [2, 4, 8]

        self.error_divide = 4
        self.d = 0
        self.sps = sps

        # From the worst case sampling point, this + fractional delay shift to optimal sampling point
        self.max_int_delay = sps
        self.interpolator = Interpolator()
        self.out_int = [0.0] * self.sps*4
        self.sps_counter = 0

        self.lim = sps



    def model_main(self, xlist):
        err_debug = []
        mu_debug = []
        ret = []

        lastd = self.d
        ii = 0
        ll = 0
        f = interp1d(range(len(xlist)), xlist)

        dd = [0.0] * 3
        f = False
        cnt = 0
        last_err = 0.0
        ccc = 0
        # plt.stem(xlist)
        # plt.show()
        for sample in xlist:
            ii += 1
            i_sample = self.interpolator.filter(sample, self.d % 1)


            # new = f(ii+ (self.d % 1))
            # i_sample = float(new)
            # plt.plot(new)
            # plt.plot(match_filtered)
            # plt.show()

            # if lastd != int(floor(self.d)):
            #     lastd = int(floor(self.d))
            #     self.d = lastd
            # else:
            #     self.out_int.append(i_sample)
            # self.out_int.append(sample)
            self.out_int.append(i_sample)


            self.sps_counter += 1
            if self.sps_counter >= self.lim:
                ll += 1
                self.sps_counter = 0

                # intd = int(floor(self.d))

                # intd = int(floor(dd[-1]))
                intd = int(floor(self.d))

                # if lastd != intd or f:


                # if lastd != intd or f:
                #     f = True
                #     cnt += 1
                #     if cnt > 2 + ccc:
                #         ccc += 1
                #         cnt = 0
                #         f = False
                #     lastd = intd
                #     e = 0
                # else:
                # intd = intd % 2
                c = self.out_int[-1-intd]
                p = self.out_int[-self.sps-intd]
                m = self.out_int[-self.sps // 2-intd]
                print(f'{c:.2f} {m:.2f} {p:.2f}')
                # last_err = e
                e = (c - p) * m


                # c = self.out_int[-1-intd]
                # p = self.out_int[-self.sps-intd]
                # m = self.out_int[-self.sps // 2-intd]
                # print(f'{c:.2f} {m:.2f} {p:.2f}')
                # e = (c - p) * m
                # last_err = e


                self.d = (self.d + 0.05) % self.max_int_delay
                dd = [self.d] + dd[:-1]

                # intd = int(floor(dd[-1]))
                # intd = int(floor(self.d))
                intd = int(floor(self.d))
                if lastd != intd:
                    lastd = intd
                    e = 0
                    self.sps_counter = -self.sps * lastd

                # dd = [self.d % 1] + dd[:-1]
                # self.d = self.d % (self.sps+1+3)



                mu_debug.append(self.d)
                err_debug.append(e)
                ret.append(self.out_int[-1 - intd + 1])
                # plt.plot([p, m, c])
                # plt.plot(self.out_int)
                # plt.plot(err_debug)
                # plt.show()

        return ret, err_debug, mu_debug

    # def model_main(self, xlist):
    #     err_debug = []
    #     mu_debug = []
    #     ret = []
    #
    #     out = np.array([0.0] * 10000)
    #     oo = 0
    #     ni = len(xlist) - 8
    #     ii = 0
    #     while ii < ni:
    #         out[oo] = self.interpolator.filter(xlist[ii], self.d % 1)
    #
    #         e = (out[oo] - out[oo-self.sps]) * out[oo-self.sps // 2]
    #         self.d += e / 4
    #
    #         ii += int(np.floor(self.d)/2)
    #         oo += 1
    #         mu_debug.append(self.d)
    #         err_debug.append(e)
    #         ret.append(self.out_int[-1 + 1])
    #
    #     return ret, err_debug, mu_debug
    # def model_main(self, xlist):
    #     err_debug = []
    #     mu_debug = []
    #     ret = []
    #
    #     ii = 0
    #     ll = 0
    #     for sample in xlist:
    #         i_sample = self.interpolator.filter(sample, self.d % 1)
    #         self.out_int.append(i_sample)
    #
    #         self.sps_counter += 1
    #         if self.sps_counter >= self.lim:
    #             ii += 1
    #             self.sps_counter = 0
    #
    #             intd = int(floor(self.d))
    #             # if ii == 9:
    #             #     # 1, 4, 5 BEST
    #             #     b = 1
    #             #     e = (self.out_int[-1 - intd+b] - self.out_int[-self.sps - intd+b]) * self.out_int[-self.sps // 2 - intd+b]
    #             # else:
    #             e = (self.out_int[-1 - intd] - self.out_int[-self.sps - intd]) * self.out_int[-self.sps // 2 - intd]
    #             # self.d += e / self.error_divide
    #             self.d += 0.05
    #
    #             self.d = self.d % (self.sps+1+3)
    #             print(self.d)
    #
    #             mu_debug.append(self.d)
    #             err_debug.append(e)
    #             ret.append(self.out_int[-1 - intd + 1])
    #             # self.lim = int(self.d)
    #
    #     return ret, err_debug, mu_debug

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
