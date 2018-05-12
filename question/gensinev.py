# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 2018/5/11 下午1:41
import torch
from basic.baseFunction.baseFunction import baseFunction

class f_gensinev(baseFunction):
    def __init__(self, n_var, m_rx, Gpu=False):
        super(f_gensinev, self).__init__(n_var, m_rx, False, Gpu)

    def _function(self, var_list):
        F = 0
        c1 = 1e-4
        c2 = 4
        self.var_x = var_list
        for i in range(0, self.m-1):
            F = F + (var_list[i+1] - torch.sin(var_list[i])).pow(2) / c1 \
                + var_list[i].pow(2) / c2
        self.F_v = F
        return F