# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 2018/5/9 下午9:33

import torch
from basic.baseOptimizer.baseOptimizer import baseOptimizer



"""
论文：NMN-Randon
方法：GBB
"""
class GBB(baseOptimizer):

    def __init__(self, question, line_search_param=None):
        super().__init__(question, "", line_search_param, required_norm=False)
        self.name = "GBB"

    def _next_step(self, xk, yk, yk_1, gk, gk_1, dk_1, required_norm=False):
        return -1.0 * gk




