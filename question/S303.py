# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 2018/5/8 下午10:36

from basic.baseFunction.baseFunction import baseFunction


class f_S303(baseFunction):
    def __init__(self, n_var, m_rx, Gpu=False):
        super(f_S303, self).__init__(n_var, m_rx, False, Gpu)

    def _function(self, var_list):
        F_var1 = 0
        F_var2 = 0
        self.var_x = var_list
        for i in range(0, self.m ):
            t = (i + 1) / 2
            F_var1 = F_var1 + var_list[i].pow(2)
            F_var2 = F_var2 + t * var_list[i].pow(2)
        F = F_var1 + F_var2.pow(2) + F_var2.pow(4)
        self.F_v = F
        return F