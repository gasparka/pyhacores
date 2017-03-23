from math import floor
import numpy as np

import pytest
from pyha.common.hwsim import HW
from pyha.common.sfix import Sfix
from scipy.interpolate import interp1d

from pyhacores.under_construction.interpolator.model import Interpolator
import matplotlib.pyplot as plt


class GardnerTimingRecovery(HW):
    def __init__(self, sps, test_inject_error=None):
        # sps must be divisible by 2 -> required by gardner
        assert not (sps & 1)
        self.test_inject_error = test_inject_error
        self.sps = sps
        self.interpolator = Interpolator()


        self.counter = 0
        self.mu = 0.0
        self.skip_error_update = False
        self.e = 0.0

        self.sample_shr = [0.0] * (self.sps + 1)


    def main(self, sample):
        sample = float(sample)

        # sample = self.interpolator.filter(sample, mu)
        self.next.sample_shr = [sample] + self.sample_shr[:-1]
        if self.counter == self.sps:
            self.next.counter = 1
            previous = self.sample_shr[self.sps]
            middle = self.sample_shr[self.sps // 2]
            current = self.sample_shr[0]
            # print(f'({current:.2f} - {previous:.2f}) * {middle:.2f}')

            if self.skip_error_update:
                self.next.skip_error_update = False
            else:
                self.next.e = (current - previous) * middle

            if self.test_inject_error is not None:
                self.next.mu = self.mu + self.test_inject_error
            else:
                self.next.mu = self.mu + self.e / 4

            if self.mu > 1.0:
                self.next.skip_error_update = True
                print('>')
                self.next.mu = self.mu % 1
                self.next.counter = 1
            elif self.mu  < 0.0:
                self.next.skip_error_update = True
                print('<')
                self.next.mu = self.mu % 1
                self.next.counter = -1
        else:
            self.next.counter = self.counter + 1

        return self.sample_shr[0], self.e, self.mu

    def model_main(self, xlist):
        err_debug = []
        ret = []
        mu_debug = []

        counter = 0
        mu = 0.0

        delay = [0.0] * (self.sps + 1)
        hw_delay = [0.0] * 100
        skip_error_update = False
        cdelay = [0] * 4

        sample_now = False
        for sample in xlist:
            counter += 1
            if counter == self.sps//2:

                # sample = self.interpolator.filter(sample, mu)
                # hw_delay = [sample] + hw_delay[:-1]
                # delay = [hw_delay[-1]] + delay[:-1]
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


        return ret, err_debug, mu_debug

        # def model_main(self, xlist):
        #     err_debug = []
        #     ret = []
        #     mu_debug = []
        #     d = 1
        #
        #     e = 0.0
        #     skip_next = False
        #     state = 0
        #     for sample in xlist:
        #         i_sample = self.interpolator.filter(sample, self.mu)
        #         self.out_int.append(i_sample)
        #
        #         self.sps_counter += 1
        #         if self.sps_counter >= self.sps:
        #             self.sps_counter = 0
        #             if skip_next:
        #                 #
        #                 skip_next = False
        #                 # continue
        #             else:
        #                 e = (self.out_int[-1 - d] - self.out_int[-self.sps - d]) * self.out_int[-self.sps // 2 - d]
        #                 if self.test_inject_error is not None:
        #                     # if d >= 1 and self.mu > 0.1:
        #                     #     state = 1
        #                     #
        #                     # if d <= 0 and self.mu < 0.9:
        #                     #     state = 0
        #                     #
        #                     # if not state:
        #                     #     self.mu = self.mu + self.test_inject_error
        #                     # else:
        #                     #     self.mu = self.mu - self.test_inject_error
        #
        #                     self.mu = self.mu - self.test_inject_error
        #
        #                 else:
        #                     self.mu = self.mu + e / 4
        #
        #
        #
        #             if self.mu < 0.0 - self.hysteresis:
        #                 self.sps_counter -= 3
        #                 if d == 0:
        #                     ret.append(self.out_int[-4])
        #                 else:
        #                     ret.append(self.out_int[-d])
        #                 skip_next = True
        #                 # self.mu = self.mu + 1.0
        #                 self.mu = 1.0
        #                 d = (d - 1) % self.sps
        #                 print('i: ', len(ret), ' <d:', d, ' mu: ', self.mu, ' o: ', ret[-1])
        #             elif self.mu > 1.0 + self.hysteresis:
        #                 self.sps_counter -= 3
        #                 if d == 0:
        #                     ret.append(self.out_int[-4])
        #                 else:
        #                     ret.append(self.out_int[-d])
        #                 skip_next = True
        #                 # self.mu = self.mu - 1.0
        #                 self.mu = 0.0
        #                 d = (d + 1) % self.sps
        #
        #                 print('i: ', len(ret), ' >d:', d, ' mu: ', self.mu, ' o: ', ret[-1])
        #
        #             else:
        #                 # no idea why this is needed...kind of scary
        #                 no = -1 - d + 1
        #                 if no == 0:
        #                     no = -4
        #                 ret.append(self.out_int[no])
        #
        #             mu_debug.append(d+self.mu)
        #             err_debug.append(e)
        #
        #     return ret, err_debug, mu_debug

    # class GardnerTimingRecovery:
    #     def __init__(self, sps, test_inject_error=None):
    #         # sps must be divisible by 2 -> required by gardner
    #         assert sps in [2, 4, 8]
    #
    #         self.test_inject_error = test_inject_error
    #         self.error_divide = 2
    #         self.d = 3
    #         self.sps = sps
    #
    #         # From the worst case sampling point, this + fractional delay shift to optimal sampling point
    #         self.max_int_delay = sps
    #         self.interpolator = Interpolator()
    #         self.out_int = [0.0] * self.sps * 4
    #         self.sps_counter = 0
    #
    #         self.lim = sps
    #
    #     def model_main(self, xlist):
    #         err_debug = []
    #         mu_debug = []
    #         ret = []
    #
    #         lastd = self.d
    #
    #         for sample in xlist:
    #             i_sample = self.interpolator.filter(sample, self.d % 1)
    #             self.out_int.append(i_sample)
    #
    #             self.sps_counter += 1
    #             if self.sps_counter >= self.lim:
    #                 self.sps_counter = 0
    #
    #                 intd = int(floor(self.d))
    #
    #                 if lastd != intd:
    #                     print('JO JO JO')
    #                     # due to the interpolator delay, we have to skip error update when ever integer delay changes!
    #                     lastd = intd
    #                 else:
    #                     c = self.out_int[-1 - intd]
    #                     p = self.out_int[-self.sps - intd]
    #                     m = self.out_int[-self.sps // 2 - intd]
    #                     e = (c - p) * m
    #
    #                 if self.test_inject_error is not None:
    #                     self.d = (self.d + self.test_inject_error) % self.max_int_delay
    #                 else:
    #                     self.d = (self.d - e / self.error_divide) % self.max_int_delay
    #
    #                 mu_debug.append(self.d)
    #                 err_debug.append(e)
    #                 ret.append(self.out_int[-1-intd-1])
    #
    #
    #         return ret, err_debug, mu_debug

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
