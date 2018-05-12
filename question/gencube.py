# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 2018/5/8 下午10:36

from basic.baseFunction.baseFunction import baseFunction

class f_gencube(baseFunction):
    def __init__(self, n_var, m_rx, Gpu=False):
        super(f_gencube, self).__init__(n_var, m_rx, False, Gpu)

    def _function(self, var_list):
        F = 0
        self.var_x = var_list
        for i in range(0, self.m):
            if i == 0:
                F = F + (var_list[i] - 1).pow(2)
                continue
            F = F + 100 * (var_list[i] - var_list[i-1].pow(3)).pow(2)
        self.F_v = F
        return F