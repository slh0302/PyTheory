# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 2018/5/11 下午12:25

import math
from basic.baseOptimizer.baseOptimizer import baseOptimizer

"""
Efficient Generalized Conjugate Gradient Algorithms
"""
class EGCG(baseOptimizer):
    def __init__(self, question, line_search=None, line_search_param=None, App_Hessian=False, sqrt_eta=4e-10):
        super().__init__(question, line_search, line_search_param, required_norm=False)
        self.name = "FR-CG"
        self.k_step = 1
        self.app_hess = App_Hessian
        self.sqrt_eta = sqrt_eta
        self.param_r = 1 / self.sqrt_eta

    def _init_dk(self, xk, gk, required_norm=False):
        self.k_step = 1
        if required_norm:
            gk = gk.norm()
        return -1.0 * gk

    def _next_step(self, xk, yk, yk_1, gk, gk_1, dk_1, required_norm=False):
        def f_value(point):
            return self.Function.g(point)

        if self.k_step > self.Function.n and self.Function.n > 2:
            dk = -1.0 * gk
            self.k_step = 1
            return dk

        sk_1 = dk_1
        delta = self.sqrt_eta / math.sqrt(sk_1.dot(sk_1))
        gama = self.sqrt_eta / math.sqrt(gk_1.dot(dk_1))
        if self.app_hess:
            Hk = f_value(xk + delta * sk_1) - gk
            Hk2 = f_value(xk + gama * sk_1) - gk
            tk = sk_1.dot(Hk) / delta
            uk = gk.dot(Hk) / delta
            vk = gk.dot(Hk2) / gama
        else:
            Hk = self.Function.G(xk)
            tmp = Hk.mm(sk_1.view(-1,1))
            tk = sk_1.mm(tmp)
            vk = gk.mm(Hk.mm(gk.view(-1,1)))
            uk = gk.mm(tmp)

        if tk > 0 and vk > 0 and (1 - uk.pow(2)/(tk * vk)) >= 1/(4 * self.param_r) \
                and (vk / gk.dot(gk)) / (tk / sk_1.dot(sk_1)) <= self.param_r:
            sk = (uk * gk.dot(sk_1) - tk * gk.dot(gk)) * gk + \
                 (uk * gk.dot(gk) - vk * gk.dot(sk_1)) * sk_1
            wk = tk * vk - uk.pow(2)
            sk = sk / wk
            self.k_step = self.k_step + 1
        else:
            sk = -1.0 * gk
            self.k_step = 1
            print( "Using -gk as direction of EGCG method" )

        dk = sk
        return dk
