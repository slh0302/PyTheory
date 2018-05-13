# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 2018/5/7 下午5:30
import torch
import numpy as np
from question.S303 import f_S303
from Optimizer.FR import FR
from Optimizer.PRP import PRP
from Optimizer.GBB import GBB

"""
参数设置
 eps, max_iter, JinTui_step=1e-5, amax=None, min_alph=1.0
"""
eps = 1e-8
max_iter = 10
JinTui_step = 1e-12

"""
函数初始化
"""
n_var = 10000
m_rx = n_var
qs = f_S303(n_var, m_rx)


"""
初始点设置
"""
sx = [0.1 for _ in range(n_var)]
start_point = torch.Tensor(np.array(sx)).double()

"""
线搜索设置：
    "extract":
        return ELS(param, self.Function)
    "wlf":
        return wolfe(param, self.Function)
    "stwlf":
        return st_wolfe(param, self.Function)
    else:
        return Armijo_Goldstein(param, self.Function)
"""
opt = FR(qs, "stwlf", [0.95, 0.05], required_norm=True)

opt.optimize(start_point, eps, max_iter, JinTui_step=JinTui_step)
