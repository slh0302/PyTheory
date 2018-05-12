# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 2018/5/6 下午9:30

import torch
from basic.baseOptimizer.baseOptimizer import baseOptimizer

class DampNewton(baseOptimizer):
    def __init__(self, question, line_search=None, line_search_param=None):
        super().__init__(question, line_search, line_search_param)
        self.name = "Damp-Newton"

    def _init_dk(self, xk, gk, required_norm=False):
        G = self.Function.G(xk)
        e, _ = torch.Tensor.symeig(G)
        if G.det() == 0:
            if required_norm:
                gk = gk / gk.norm()
            return - 1.0 * gk
        else:
            G_1 = torch.Tensor.inverse(G)
            res_dk = G_1.mm(gk.view(-1, 1)).view(-1)
            res_dk = res_dk / res_dk.norm()
            return - 1.0 * res_dk

    def _next_step(self, xk, yk, yk_1, gk, gk_1, dk_1, required_norm=False):
        G = self.Function.G(xk)
        e, _ = torch.Tensor.symeig(G)
        if G.det() == 0:
            if required_norm:
                gk = gk / gk.norm()
            return - 1.0 * gk
        else:
            G_1 = torch.Tensor.inverse(G)
            res_dk = G_1.mm(gk.view(-1, 1)).view(-1)
            res_dk = res_dk / res_dk.norm()
            return - 1.0 * res_dk




