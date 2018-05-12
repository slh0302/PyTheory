# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 2018/5/7 下午5:30
from basic.baseOptimizer.baseOptimizer import baseOptimizer

class  PRP(baseOptimizer):
    def __init__(self, question, line_search=None, line_search_param=None, required_norm=False):
        super().__init__(question, line_search, line_search_param, required_norm=required_norm)
        self.name = "PRP-CG"

    def _next_step(self, xk, yk, yk_1, gk, gk_1, dk_1, required_norm=False):
        betak_1 = gk.dot(gk - gk_1) / gk_1.dot(gk_1)
        dk = -gk + betak_1 * dk_1
        print("betak=: %.6f" % (betak_1))
        if required_norm:
            dk = dk / dk.norm()
        return dk