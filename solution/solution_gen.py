# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 2018/5/8 下午5:36

import torch
import numpy as np
from question.gencube import f_gencube
from Optimizer.FR import FR
"""
 param
"""
n_var = 10000
m_rx = n_var
qs = f_gencube(n_var, m_rx)

"""
if name == "extract":
    return ELS(param, self.Function)
elif name == "wlf":
    return wolfe(param, self.Function)
elif name == "stwlf":
    return st_wolfe(param, self.Function)
else:
    return Armijo_Goldstein(param, self.Function)
"""

sx = [0.9 for _ in range(n_var)]
start_point = torch.Tensor(np.array(sx)).double()

"""
 start_point, eps, max_iter, JinTui_step=1e-5, amax=None, min_alph=1.0
"""
eps = 1e-8
max_iter = 15

"""
# GBB 方法

self.kexi = self.param[0]
self.theta1 = self.param[1]
self.theta2 = self.param[2]
self.gama = self.param[3]
self.M = self.param[4]
"""

opt = FR(qs, "stwlf", [0.95, 0.05], required_norm=True)
# opt.optimize(start_point, eps, max_iter, JinTui_step=1e-2)
opt.optimize(start_point, eps, max_iter, JinTui_step=1e-10, required_debug=False)



