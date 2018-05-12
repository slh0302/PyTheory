#coding:utf-8
#author:LiHui Su

import torch
import numpy as np
from question_1.B3d import f_B3d
from Optimizer.DampNewton import DampNewton

# n_var, m_rx, need_hessian=True, Gpu=False
n_var = 3
m_rx = 10000
b3d = f_B3d(n_var, m_rx)
# question, line_search=None, line_search_param=None
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
dn = DampNewton(b3d, "extract", [0.95, 0.05])

start_point = torch.Tensor(np.array([0, 10, 20])).double()

"""
 start_point, eps, max_iter, JinTui_step=1e-5, amax=None, min_alph=1.0
"""
eps = 1e-8
max_iter = 10
dn.optimize(start_point, eps, max_iter, JinTui_step=1e-3)




