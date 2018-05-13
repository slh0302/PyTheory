# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 2018/5/11 下午8:50


import torch
import numpy as np
from basic.utils import  InitFunction, InitOptimize

"""
问题和优化器选择
"""
question = "S303"   # gencube S303 genbard genpowsg
n_var = 10000          # 100 1000 10000
m_rx = n_var
optimzie = "GBB"         # FR PRP PRP+ DY EGCG GBB
ls_method = "stwlf"     # stwlf extract wlf armgld
eps = 1e-8
amax = 5
max_iter = 10

# JinTui
sita = 0.95
rho = 0.15
JinTui_step = None

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

qs, s_p = InitFunction(question, n_var, m_rx)
opt = InitOptimize(qs, optimzie, param, ls_method, App_hess=True, sqrt_eta=4e-10)

"""
运行优化器
"""
print("param:",optimzie, question, ls_method, n_var, eps, max_iter, JinTui_step, amax, param)
start_point = torch.Tensor(np.array(s_p)).double()
opt.optimize(start_point, eps, max_iter, JinTui_step=JinTui_step, amax=amax)
