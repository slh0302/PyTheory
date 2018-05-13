# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 2018/5/6 下午7:30

import torch
import math
import numpy.random as rd
import numpy as np
from basic.baseFunction.baseFunction import baseFunction

class baselineSearch:
    def __init__(self, param, function, extract):
        self.param = param # [cita, rho]
        assert isinstance(function, baseFunction)
        self.function = function
        self.extract = extract
        self.gold_error = 1e-3
        self.extract_max_iter = 15
        if not self._check():
            print("Wrong param")
            self.status = False
            exit(-1)
        else:
            self.status = True

    def _lineSearch(self, start_point, s_v, s_g, dk, max_iter, JinTui_step=None,
                    amax=None, pre_yk=None, min_alph=1.0):

        info = {}
        info['feval'] = 0
        info['iter'] = 0

        def phi(alpha):
            info['feval'] = info['feval'] + 1
            return self.function.F(start_point + alpha * dk)

        def derphi(alpha):
            info['feval'] = info['feval'] + 1
            return torch.dot(self.function.g(start_point + alpha * dk), dk)

        # First get JinTui
        if JinTui_step != None:
            # _JinTui(self, alpha, s_v, point, dk, step):
            interval, f_value, JT_info = self._JinTui(0, s_v, JinTui_step, phi)
            print("Jintui interval = ", interval)
            # info['feval'] = info['feval'] + JT_info['feval']
            info['iter'] = info['iter'] + JT_info['iter']
            print(info)
        else:
            if amax == None:
                print("Jin Tui or amax must choose one.")
                exit(-1)
            interval = [0, amax]
            f_value = [s_v, phi(amax)]
            # info['feval'] = info['feval'] + 2

        left, right = interval
        l_value, r_value = f_value
        # 精确线搜索
        if self.extract:
            step_size, f_phi, glod_info = self._gold(phi, left, right, self.gold_error, max_iter=max_iter)
            g_phi = derphi(step_size)
            alph = step_size
            # info['feval'] = info['feval'] + 1
            # info['feval'] = info['feval'] + glod_info['feval'] + 1
            info['iter'] = info['iter'] + glod_info['iter']
            print(info)
            return alph, f_phi, g_phi, info

        # 当前点的函数值和下一点的函数值
        # 参数设置
        phi0 = s_v
        derphi0 = s_g.dot(dk)
        alpha0 = left

        if JinTui_step != None:
            alpha1 = right
        else:
            if pre_yk is not None and derphi0 != 0:
                alpha1 = min(min_alph, 1.01 * 2 * (phi0 - pre_yk) / derphi0)
            else:
                alpha1 = 1.0

            if alpha1 < 0:
                alpha1 = min_alph

        phi_a1 = phi(alpha1)
        phi_a0 = l_value
        derphi_a0 = derphi(phi_a0)
        extra_condition = lambda alpha, phi: True
        c1 = self.param[1]  # rho
        c2 = self.param[0]  # cita
        zoom_info = {}
        zoom_info['feval'] = 0
        zoom_info['iter'] = 0

        # 根据amax寻找一个包含满足armijo准则的区间，如果点直接满足也可以
        if JinTui_step == None:
            for i in range(0, max_iter):
                if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or \
                        ((phi_a1 >= phi_a0) and (i > 1)):
                    alpha_star, phi_star, derphi_star, zoom_info = \
                        self._zoom(alpha0, alpha1, phi_a0, phi_a1, derphi_a0,
                                   phi, derphi, phi0, derphi0, c1, c2, extra_condition, max_iter=max_iter)
                    break

                derphi_a1 = derphi(alpha1)
                # 根据特定方法进行更改
                if self._constraint(phi0=phi0, phi_aj=phi_a1,
                                    a_j=alpha1, derphi0=derphi0, derphi_aj=derphi_a1):
                    # info['feval'] = info['feval'] + 1
                    info['iter'] = info['iter'] + 1
                    alpha_star = alpha1
                    phi_star = phi_a1
                    derphi_star = derphi_a1
                    break

                if (derphi_a1 >= 0):
                    alpha_star, phi_star, derphi_star, zoom_info = \
                        self._zoom(alpha1, alpha0, phi_a1,
                              phi_a0, derphi_a1, phi, derphi,
                              phi0, derphi0, c1, c2, extra_condition, max_iter=max_iter)
                    break

                scale = rd.random_integers(10, 20) / 10
                alpha2 = scale * alpha1  # increase by factor of two on each iteration
                if amax is not None:
                    alpha2 = min(alpha2, amax)
                alpha0 = alpha1
                alpha1 = alpha2
                phi_a0 = phi_a1
                phi_a1 = phi(alpha1) #phi(alpha1)
                derphi_a0 = derphi_a1
                # info['feval'] = info['feval'] + 2
                info['iter'] = info['iter'] + 1
                if i == max_iter - 1 :
                    # stopping test maxiter reached
                    alpha_star = alpha1
                    phi_star = phi_a1
                    derphi_star = None
                    print('The line search algorithm did not converge')
                    break
        else:
            # 直接使用进退法得到的单调区间
            alpha_star, phi_star, derphi_star, zoom_info = \
                self._zoom(alpha0, alpha1, phi_a0, phi_a1, derphi_a0,
                           phi, derphi, phi0, derphi0, c1, c2, extra_condition)
            if zoom_info['done'] ==  -1:
                alpha_star = (alpha0 + alpha1) / 2
                phi_star = phi(alpha_star)
                derphi_star = derphi(alpha_star)

        # info['feval'] = info['feval'] + zoom_info['feval'] + 1
        info['iter'] = info['iter'] + zoom_info['iter']
        step_size = alpha_star
        p_value = phi_star
        p_grad = derphi_star
        return step_size, p_value, p_grad, info

    def _constraint(self, **kwargs):
        raise NotImplementedError

    def _check(self):
        raise NotImplementedError

    def _JinTui(self, alpha, s_v, step, f_value):
        # default: alpha==0, step==0.0001
        info = {}
        info['feval'] = 0
        info['iter'] = 0
        ak = alpha
        k = 0
        cal_k = s_v
        value = {}
        while True:
            info['iter'] = info['iter'] + 1
            info['feval'] = info['feval'] + 1
            akp = ak + step
            cal_kp = f_value(akp)
            # print(point + akp * dk)
            if cal_kp < cal_k:
                step = 2 * step
                alpha = ak
                ak = akp
                cal_k = cal_kp
                k += 1
            else:
                if k == 0:
                    # 反向搜索
                    step = -step
                    alpha = akp
                    akp = ak
                    cal_kp = cal_k
                    k = 1
                else:
                    value[alpha] = cal_k
                    value[akp] = cal_kp
                    break
        left = min(alpha, akp)
        right = max(alpha, akp)
        return [left, right], [value[left], value[right]], info

    def _gold(self, f, lower, upper, merror, max_iter=None):
        seg = (math.sqrt(5)-1)/2
        error = abs(lower - upper)
        info = {}
        info['feval'] = 1
        info['iter'] = 1
        # vals = []
        # vals.append((lower+upper)/2)
        # objectf = []
        # objectf.append(f((lower+upper)/2))
        # you can customize your own condition of convergence, here we limit the error term
        # TODO: can be faster
        right_alph = upper
        while error >= merror * right_alph:
            temp1 = upper - seg * (upper-lower)
            temp2 = lower + seg * (upper-lower)
            ft1 = f(temp1)
            ft2 = f(temp2)
            info['feval'] += 2
            info['iter'] += 1
            if ft1 - ft2 < 0:
                upper = temp2
            else:
                lower = temp1
            error = abs(lower - upper)

            # vals.append((lower+upper)/2)
            # objectf.append(f((lower+upper)/2))
            if max_iter != None and info['iter'] > max_iter:
                print("Reach the max iter in 0.618 ELS.")
                print(info)
                break
        print(info)
        return (temp2+temp1)/2, f((temp2+temp1)/2), info

    def _zoom(self, a_lo, a_hi, phi_lo, phi_hi, derphi_lo, phi, derphi,
              phi0, derphi0, c1, c2, extra_condition, max_iter=10):
        """
        Part of the optimization algorithm.
        """
        info = {}
        info['feval'] = 0
        info['iter'] = 0
        info['done'] = 0

        maxiter = max_iter
        i = 0
        delta1 = 0.2  # cubic interpolant check
        delta2 = 0.1  # quadratic interpolant check
        phi_rec = phi0
        a_rec = 0
        while True:
            # interpolate to find a trial step length between a_lo and
            # a_hi Need to choose interpolation here.  Use cubic
            # interpolation and then if the result is within delta *
            # dalpha or outside of the interval bounded by a_lo or a_hi
            # then use quadratic interpolation, if the result is still too
            # close, then use bisection

            dalpha = a_hi - a_lo
            if dalpha < 0:
                a, b = a_hi, a_lo
            else:
                a, b = a_lo, a_hi

            # minimizer of cubic interpolant
            # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
            #
            # if the result is too close to the end points (or out of the
            # interval) then use quadratic interpolation with phi_lo,
            # derphi_lo and phi_hi if the result is stil too close to the
            # end points (or out of the interval) then use bisection

            if (i > 0):
                cchk = delta1 * dalpha
                a_j = self._cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi,
                                a_rec, phi_rec)
            if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
                qchk = delta2 * dalpha
                a_j = self._quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
                if (a_j is None) or (a_j > b - qchk) or (a_j < a + qchk):
                    a_j = a_lo + 0.5 * dalpha

            # Check new value of a_j

            phi_aj = phi(a_j)
            info['feval'] = info['feval'] + 1
            if (phi_aj > phi0 + c1 * a_j * derphi0) or (phi_aj >= phi_lo):
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_j
                phi_hi = phi_aj
            else:
                derphi_aj = derphi(a_j)
                # info['feval'] = info['feval'] + 1
                # if abs(derphi_aj) <= -c2 * derphi0 and extra_condition(a_j, phi_aj):
                if self._constraint(phi0=phi0, phi_aj=phi_aj,
                                    a_j=a_j, derphi0=derphi0, derphi_aj=derphi_aj):
                    a_star = a_j
                    val_star = phi_aj
                    valprime_star = derphi_aj
                    break

                # 重新更新区间
                if derphi_aj * (a_hi - a_lo) >= 0:
                    phi_rec = phi_hi
                    a_rec = a_hi
                    a_hi = a_lo
                    phi_hi = phi_lo
                else:
                    phi_rec = phi_lo
                    a_rec = a_lo
                a_lo = a_j
                phi_lo = phi_aj
                derphi_lo = derphi_aj
            info['iter'] = info['iter'] + 1
            i += 1
            if (i > maxiter):
                if phi_aj < phi0:
                    print("zoom reach max, using least value")
                    a_star = a_j
                    val_star = phi_aj
                    valprime_star = derphi(a_j)
                else:
                    # Failed to find a conforming step size
                    info['done'] = -1
                    a_star = None
                    val_star = None
                    valprime_star = None
                break
        return a_star, val_star, valprime_star, info

    def _quadmin(self, a, fa, fpa, b, fb):
        """
        Finds the minimizer for a quadratic polynomial that goes through
        the points (a,fa), (b,fb) with derivative at a of fpa,

        """
        # f(x) = B*(x-a)^2 + C*(x-a) + D
        try:
            D = fa
            C = fpa
            db = b - a * 1.0
            B = (fb - D - C * db) / (db * db)
            xmin = a - C / (2.0 * B)
        except ArithmeticError:
            print("quad Math cubic error, using mid value")
            return (a+b)/2

        try:
            tmp = xmin.data.numpy()
        except Exception:
            tmp = xmin.cpu().data.numpy()

        if not np.isfinite(tmp):
            print("quad Math infinite error, using mid value")
            return (a+b)/2
        return xmin

    def _cubicmin(self, a, fa, fpa, b, fb, c, fc):
        """
        Finds the minimizer for a cubic polynomial that goes through the
        points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.

        If no minimizer can be found return None

        """
        # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D

        try:
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = torch.zeros((2, 2)).double()
            d1[0, 0] = dc ** 2
            d1[0, 1] = -db ** 2
            d1[1, 0] = -dc ** 3
            d1[1, 1] = db ** 3
            tmp = torch.Tensor([fb - fa - C * db,
                                fc - fa - C * dc]).double()
            [A, B] = d1.mm(tmp.view(-1,1))
            A /= denom
            B /= denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + torch.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            print("Cube Math cubic error, using mid value")
            return None
            # return (a+b)/2
        try:
            tmp = xmin.data.numpy()
        except Exception:
            tmp = xmin.cpu().data.numpy()

        if not np.isfinite(tmp):
            print("Cube Math infinite error, using mid value")
            return None
            # return (a+b)/2
        return xmin