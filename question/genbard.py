# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 2018/5/11 下午1:48

from basic.baseFunction.baseFunction import baseFunction


class f_genbard(baseFunction):
    def __init__(self, n_var, m_rx, Gpu=False):
        super(f_genbard, self).__init__(n_var, m_rx, False, Gpu)

    def _function(self, var_list):
        yi = [0.14, 0.18, 0.22, 0.25, 0.29,
              0.32, 0.35, 0.39, 0.37, 0.58,
              0.73, 0.96, 1.34, 2.10, 4.39]
        F = 0
        self.var_x = var_list
        for j in range(0, self.n - 2):
            for i in range(0, 15):
                ui = i + 1
                vi = 16 - ui
                wi = min(ui, vi)
                f_tmp = yi[i] - var_list[j] - (ui) / (vi * var_list[j + 1] + wi * var_list[j + 2])
                F = F + f_tmp.pow(2)
            self.F_v = F
        return F
