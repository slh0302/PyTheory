# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 2018/5/7 下午10:36

import torch
import numpy as np
from basic.baseFunction import baseFunction


class f_B3d(baseFunction):
    def __init__(self, n_var, m_rx, need_hessian=True, Gpu=False):
        super(f_B3d, self).__init__(n_var, m_rx, need_hessian, Gpu)

    def _function(self, var_list):
        F = 0
        self.var_x = var_list
        for i in range(1, self.m + 1):
            t = 0.1 * i
            rx1 = torch.exp(-t * var_list[0]) - torch.exp(-t * var_list[1])
            rx2 = var_list[2] * (np.exp(-t) - np.exp(-10 * t))
            F = F + (rx1 - rx2) ** 2
        self.F_v = F
        return F
