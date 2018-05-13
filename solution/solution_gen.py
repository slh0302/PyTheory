# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 2018/5/8 下午5:36

import torch
import os
import numpy as np
from question.gencube import f_gencube
from Optimizer.FR import FR
from Optimizer.DY import DY
from Optimizer.EGCG import EGCG
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
"""
 param
"""
n_var = 10000
m_rx = n_var
Use_GPU = False
qs = f_gencube(n_var, m_rx, Gpu=Use_GPU)

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
if Use_GPU:
    start_point = torch.Tensor(np.array(sx)).double().cuda()
else:
    start_point = torch.Tensor(np.array(sx)).double()

"""
 start_point, eps, max_iter, JinTui_step=1e-5, amax=None, min_alph=1.0
"""
eps = 1e-8
max_iter = 10
tmp_param = [0.95, 0.05]
opt = EGCG(qs, "stwlf", tmp_param, App_Hessian=True)
JinTui_step = 1e-4
amax = None
# opt.optimize(start_point, eps, max_iter, JinTui_step=1e-2)
print("param:", n_var, eps, max_iter, JinTui_step, amax, tmp_param)
opt.optimize(start_point, eps, max_iter, JinTui_step=JinTui_step, amax=amax, required_debug=False)



