# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 2018/5/11 下午8:26


from basic.baseFunction.baseFunction import baseFunction

class f_genpowsg(baseFunction):
    def __init__(self, n_var, m_rx, Gpu=False):
        super(f_genpowsg, self).__init__(n_var, m_rx, False, Gpu)

    def _function(self, var_list):
        F = 0
        self.var_x = var_list
        for i in range(1, int(self.m/2) - 1):
            f1 = var_list[2*i - 1 - 1] + 10 * var_list[2*i - 1]
            f2 = var_list[2*i + 1 - 1] - var_list[2*i + 2 - 1]
            f3 = var_list[2*i - 1] - 2* var_list[2*i + 1 - 1]
            f4 = var_list[2*i - 1 - 1] - var_list[2*i + 2 - 1]
            F = F + f1.pow(2) + 5 * f2.pow(2) + f3.pow(4) + 10 * f4.pow(4)
        self.F_v = F
        return F