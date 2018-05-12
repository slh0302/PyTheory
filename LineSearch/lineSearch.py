# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 2018/5/6 下午8:21

import torch
from basic.baseLineSearch.baseLineSearch import baselineSearch

class ELS(baselineSearch):
    def __init__(self, param, function):
        super().__init__(param, function, True)

    def _constraint(self, **kwargs):
        return True

    def _check(self):
        return True

    def ELS(self, start_point, s_v, s_g, dk, max_iter, JinTui_step=1e-5):
        step_size, p_value, p_grad, info = self._lineSearch(start_point, s_v, s_g, dk, max_iter,
                                                            JinTui_step, None, None)
        return step_size, p_value, p_grad, info

class wolfe(baselineSearch):
    def __init__(self, param, function):
        super().__init__(param, function, False)

    def _constraint(self, **kwargs):
        #  xk, xk_1, phixk_1, derphi0, alph
        derphi0 = kwargs['derphi0']
        derphi_aj = kwargs['derphi_aj']

        c1 = self.param[0]
        if derphi_aj >= c1 * derphi0:
            return True
        else:
            return False

    def _check(self):
        return True

    def wolfe(self, start_point, s_v, s_g, dk, max_iter, JinTui_step=None,
                    amax=None, pre_yk=None, min_alph=1.0):

        step_size, p_value, p_grad, info = self._lineSearch(start_point, s_v, s_g, dk, max_iter,
                                                            JinTui_step, amax, pre_yk, min_alph)
        return step_size, p_value, p_grad, info

class st_wolfe(baselineSearch):
    def __init__(self, param, function):
        super().__init__(param, function, False)

    def _constraint(self, **kwargs):
        derphi0 = kwargs['derphi0']
        derphi_aj = kwargs['derphi_aj']

        c1 = self.param[0]
        if abs(derphi_aj) <= -c1 * derphi0:
            return True
        else:
            return False

    def _check(self):
        return True

    def st_wolfe(self, start_point, s_v, s_g, dk, max_iter, JinTui_step=None,
                    amax=None, pre_yk=None, min_alph=1.0):
        step_size, p_value, p_grad, info = self._lineSearch(start_point, s_v, s_g, dk, max_iter,
                                                            JinTui_step, amax, pre_yk, min_alph)
        return step_size, p_value, p_grad, info

class Armijo_Goldstein(baselineSearch):
    def __init__(self, param, function):
        super().__init__(param, function, False)

    def _constraint(self, **kwargs):
        #  xk, xk_1, phixk_1, derphi0, alph
        xk = kwargs['phi0']
        xk_1 = kwargs['phi_aj']
        alph = kwargs['a_j']
        derphi0 = kwargs['derphi0']

        rho_gold = self.param[1] #rho是小得那个值
        if xk >= xk_1 + (1 - rho_gold) * alph * derphi0:
            return True
        else:
            return False

    def _check(self):
        return True


    def armgld(self, start_point, s_v, s_g, dk, max_iter, JinTui_step=None,
                    amax=None, pre_yk=None, min_alph=1.0):
        step_size, p_value, p_grad, info = self._lineSearch(start_point, s_v, s_g, dk, max_iter,
                                                            JinTui_step, amax, pre_yk, min_alph)
        return step_size, p_value, p_grad, info

class nonmonotone(baselineSearch):
    def __init__(self, param, function):
        super().__init__(param, function, False)
        # 参数设置
        self.kexi = self.param[0]
        self.theta1 = self.param[1]
        self.theta2 = self.param[2]
        self.gama = self.param[3]
        self.M = self.param[4]

        # 初始化一些值
        self.f_list = torch.Tensor([0.] * self.M).double()
        self.alph = torch.Tensor([1.]).double()
        self.step_k = 0

    def _constraint(self, **kwargs):
        return True

    def _check(self):
        if len(self.param ) < 5:
            return False
        return True

    def _choose_value(self, gk):
        norm_gk = gk.norm()
        if norm_gk > 1:
            beta = 1
        elif norm_gk <= 1 and norm_gk >= 1e-5:
            beta = 1 / norm_gk
        else:
            beta = 1e5
        return beta

    def _lineSearch(self, start_point, s_v, s_g, dk, max_iter, JinTui_step=None,
                    amax=None, pre_yk=None, min_alph=1.0):

        def f_value(point):
            return self.function.F(point)

        def der_value(point):
            return self.function.g(point)

        # 设置信息
        opt_info = [0,0]

        if self.step_k == 0:
            self.f_list[0] = s_v

        # 更改 alph
        if self.alph <= self.kexi or self.alph >= 1/self.kexi:
            alph = self._choose_value(s_g)
        else:
            alph = self.alph

        lamda = 1 / alph

        f_lamda = f_value(start_point - lamda * s_g)
        max_value = None
        for j in range(0, min(self.step_k, self.M) + 1):
            tmpk = (self.step_k - j) % self.M
            if j == 0:
                max_value = self.f_list[tmpk]
            else:
                max_value = torch.max(self.f_list[tmpk], max_value)

        lamda_k = lamda


        while not f_lamda <= max_value - self.gama * lamda_k * s_g.dot(s_g):
            fa = f_value(start_point - self.theta1 * lamda_k * s_g)
            ga = der_value(start_point - self.theta1 * lamda_k * s_g)
            # 重要导数信息计算
            ga = ga.dot(-1.0 * s_g * lamda_k)
            fb = f_value(start_point - self.theta2 * lamda_k * s_g)
            theta_min = self._quadmin(self.theta1, fa, ga, self.theta2, fb)
            lamda_k = lamda_k * theta_min
            f_lamda = f_value(start_point - lamda_k * s_g)

        new_lamda = lamda_k
        new_gk = der_value(start_point - new_lamda * s_g)
        self.alph = -1.0 * s_g.dot(new_gk - s_g) / (new_lamda * s_g.dot(s_g))
        self.step_k = self.step_k + 1
        loc = self.step_k % self.M
        self.f_list[loc] = f_lamda

        step_size = new_lamda
        p_value = f_lamda
        p_grad = new_gk
        info = {}
        info['feval'] = opt_info[0]
        info['iter'] = opt_info[1]
        return step_size, p_value, p_grad, info