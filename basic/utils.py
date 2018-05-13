# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 2018/5/11 下午8:40

from Optimizer.FR import FR
from Optimizer.GBB import GBB
from Optimizer.DY import DY
from Optimizer.EGCG import EGCG
from Optimizer.PRP_plus import PRP_plus
from Optimizer.PRP import PRP

def InitOptimize(qs, f_name, param, line_search, **kwargs):
    if f_name == "DY":
        return DY(qs, line_search, param)
    elif f_name == "FR":
        return  FR(qs, line_search, param)
    elif f_name == "PRP":
        return PRP(qs, line_search, param)
    elif f_name == "PRP+":
        return PRP_plus(qs, line_search, param)
    elif f_name == "GBB":
        return GBB(qs, param)
    elif f_name == "EGCG":
        App_hess = kwargs['App_hess']
        if 'sqrt_eta' in kwargs.keys():
            sqt_eta = kwargs['sqrt_eta']
        else:
            sqt_eta = 4e-10
        return EGCG(qs, line_search, param, App_Hessian=App_hess, sqrt_eta=sqt_eta)
    else:
        raise NotImplementedError


from question.gencube import f_gencube
from question.genbard import f_genbard
from question.S303 import f_S303
from question.genpowsg import f_genpowsg
from question.gensinev import f_gensinev

def InitFunction(f_name, n_var, m_rx):
    # s_p = [0.9 for _ in range(n_var)]

    # s_p = [ -1 for _ in range(n_var) ]
    # for i in range(0, n_var, 2):
    #     s_p[i] = s_p[i] + 4
    # s_p[0] = 4.712389
    if f_name == "gencube":
        s_p = [0.9 for _ in range(n_var)]
        return f_gencube(n_var, m_rx), s_p
    elif f_name == "genpowsg":
        s_p = [-1 for _ in range(n_var)]
        for i in range(0, n_var, 2):
            s_p[i] = s_p[i] + 4
        return  f_genpowsg(n_var, m_rx), s_p
    elif f_name == "S303":
        s_p = [0.1 for _ in range(n_var)]
        return f_S303(n_var, m_rx), s_p
    elif f_name == "genbard":
        s_p = [ -1 for _ in range(n_var) ]
        s_p[0] = 4.712389
        return f_genbard(n_var, m_rx), s_p
    elif f_name == "gensinev":
        s_p = [-1 for _ in range(n_var)]
        return f_gensinev(n_var, m_rx), s_p
    else:
        raise NotImplementedError

