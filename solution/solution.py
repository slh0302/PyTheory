# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 2018/5/11 下午8:50


import torch
import numpy as np
from basic.utils import  InitFunction, InitOptimize

"""
问题和优化器选择
"""
question = "gencube"
n_var = 10
m_rx = n_var
optimzie = "GBB"
ls_method = "extract"
s_p = [0.9 for _ in range(n_var)]
start_point = torch.Tensor(np.array(s_p)).double()
eps = 1e-8
max_iter = 15

"""
线搜索默认参数
#   参数rho，sita 
"""
sita = 0.85
rho = 0.15
JinTui_step = 1e-2

"""
# GBB 方法参数
"""
kexi = 1e-10
theta1 = 0.1
theta2 = 0.5
gama = 1e-4 # 这个参数在GBB方法里和sita一样
M = 10
if optimzie == "GBB":
    param = [kexi, theta1, theta2, gama, M]
else:
    param = [rho, sita]

qs = InitFunction(question, n_var, m_rx)
opt = InitOptimize(qs, optimzie, param, ls_method, App_hess=True, sqrt_eta=4e-10)

"""
运行优化器
"""
amax = None
opt.optimize(start_point, eps, max_iter, JinTui_step=JinTui_step, amax=amax)
