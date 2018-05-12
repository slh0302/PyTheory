# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 2018/5/6 下午7:30
import torch
from torch.autograd import grad, Variable

class baseFunction:
    def __init__(self, n_var, m_rx, need_hessian=True, Gpu=False):
        self.n = n_var
        self.m = m_rx
        self.hessian = need_hessian
        self.Gpu = Gpu
        # self.F_v = torch.tensor([0.0], dtype=torch.float64)
        # self.g_v = torch.tensor([0.0] * self.n, dtype=torch.float64)
        # self.var_x = torch.tensor([0.0] * self.n, dtype=torch.float64)
        self.F_v = torch.Tensor([0.0]).double()
        self.g_v = torch.Tensor([0.0] * self.n).double()
        self.var_x = torch.Tensor([0.0] * self.n).double()

    def F(self, var_x):
        return self._function(self._check_value(var_x))

    def g(self, var_x):
        return self._der_function(self._check_value(var_x))

    def G(self, var_x):
        return self._hessian_value(self._check_value(var_x))

    def _check_value(self, var_x):
        assert isinstance(var_x, torch.Tensor)
        if not var_x.requires_grad:
            if var_x.eq(self.var_x).all() == 1:
                return self.var_x
            torch_var_x = Variable(var_x.clone(), requires_grad=True)
        else:
            if var_x.eq(self.var_x).all() == 1:
                return self.var_x
            torch_var_x = var_x.clone()

        if self.Gpu and not torch_var_x.is_cuda():
            torch_var_x = torch_var_x.cuda()

        return torch_var_x

    def _function(self, var_x):
        """
        Must be design by real function
        :param var_x:
        :return:
        """
        raise NotImplementedError

    def _der_function(self, var_x):
        if var_x.eq(self.var_x).all() == 1:
            self.g_v = grad(self.F_v, var_x, create_graph=self.hessian, retain_graph=True)[0]
            return self.g_v.clone()
        else:
            self.g_v = grad(self._function(var_x), var_x, create_graph=self.hessian, retain_graph=True)[0]
            return self.g_v.clone()

    def _hessian_value(self, var_x):
        hessian_matrix = torch.zeros(self.n, self.n, dtype=torch.float64)
        if var_x.eq( self.var_x ).all() == 1:
            grad_value = self.g_v
        else:
            grad_value = self._der_function(var_x)

        for i in range(self.n):
            tmp_tensor = torch.zeros(self.n, dtype=torch.float64)
            tmp_tensor[i] = 1.0
            matrix_i = grad(grad_value, var_x, grad_outputs=tmp_tensor, retain_graph=True)
            hessian_matrix[i, :] = matrix_i[0]

        return hessian_matrix
