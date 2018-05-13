# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 2018/5/11 下午12:16

from basic.baseOptimizer.baseOptimizer import baseOptimizer

class  DY(baseOptimizer):
    def __init__(self, question, line_search=None, line_search_param=None):
        super().__init__(question, line_search, line_search_param)
        self.name = "PRP-CG"

    def _next_step(self, xk, yk, yk_1, gk, gk_1, dk_1, required_norm=False):
        betak_1 = gk.dot(gk) / dk_1.dot(gk - gk_1)
        dk = -gk + betak_1 * dk_1
        print("betak=: %.6f" % (betak_1))
        if required_norm:
            dk = dk / dk.norm()
        return dk