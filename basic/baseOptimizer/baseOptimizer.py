# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 2018/5/6 下午1点


import datetime
from LineSearch.lineSearch import *
from basic.baseFunction.baseFunction import baseFunction


class baseOptimizer:
    def __init__(self, question, line_search, line_search_param, required_norm=False):
        assert isinstance(question, baseFunction)
        self.Function = question
        self.line_search = line_search
        self.name = "base"
        self.required_norm = required_norm
        self.ls_method = self._init_line_search(line_search, line_search_param)


    def _init_line_search(self, name, param):
        if name == "extract":
            return ELS(param, self.Function)
        elif name == "wlf":
            return wolfe(param, self.Function)
        elif name == "stwlf":
            return st_wolfe(param, self.Function)
        elif name == "armgld":
            return Armijo_Goldstein(param, self.Function)
        else:
            # nonmonotone line search
            return nonmonotone(param, self.Function)

    def optimize(self, start_point, eps, max_iter, JinTui_step=1e-5, amax=None, min_alph=1.0, required_debug=False):
        # 初始化信息记录
        opt_info = [0,0]

        # 初始化
        begin_time = datetime.datetime.now()
        s_v = self.Function.F(start_point)
        s_g = self.Function.g(start_point)
        k = 0
        yk = 0.0
        yk_2 = None
        xk_1 = start_point
        yk_1 = s_v
        gk_1 = s_g
        dk_1 = self._init_dk(xk_1, gk_1, self.required_norm)
        norm_gk = gk_1.norm()
        while norm_gk > eps:
            if self.line_search == "extract":
                step_size, p_value, p_grad, info = \
                    self.ls_method._lineSearch(xk_1, yk_1, gk_1, dk_1, max_iter, JinTui_step)
            else:

                step_size, p_value, p_grad, info = \
                    self.ls_method._lineSearch(xk_1, yk_1, gk_1, dk_1, max_iter,
                                                JinTui_step, amax, yk_2, min_alph)
            print('In step %d:' % k)
            print('   alphk= %.12lf; ||gk||= %.12lf; yk= %.12lf' % (step_size, norm_gk, yk_1))
            if required_debug:
                print('   xk= ', xk_1)
                print()
                print('   gk= ', gk_1)

            xk = xk_1 + step_size * dk_1
            yk = self.Function.F(xk)
            gk = self.Function.g(xk)
            opt_info[0] = opt_info[0] + info['feval'] + 2
            opt_info[1] = opt_info[1] + info['iter'] + 1
            dk = self._next_step(xk, yk, yk_1, gk, gk_1, dk_1, self.required_norm)
            xk_1 = xk
            yk_2 = yk_1
            if abs(yk_1 - yk) < eps:
                break
            yk_1 = yk
            gk_1 = gk
            dk_1 = dk
            norm_gk = gk_1.norm()
            k = k + 1
        end_time = datetime.datetime.now()
        print('Flinal step:' )
        print('   alphk= %.12lf; ||gk||= %.12lf;' % (step_size, norm_gk))
        print("Final yk= %.12lf" % yk.data.numpy())
        print("Total Time: ", str(end_time - begin_time))
        print("Total Fevel= %d, Total Iter= %d" % (opt_info[0], opt_info[1]))
        return yk

    def _init_dk(self, xk, gk, required_norm=False):
        if required_norm:
            gk = gk / gk.norm()
        return -1.0 * gk

    def _next_step(self, xk, yk, yk_1, gk, gk_1, dk_1, required_norm=False):
        raise NotImplementedError
