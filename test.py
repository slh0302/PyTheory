import torch
import numpy as np
from torch.autograd import Variable
from torch.autograd import grad
from torch.autograd import backward
from question_1.B3d import f_B3d

var_num = 3
min_least = 3
tensor_x = torch.from_numpy(np.array([0, 10, 20], dtype=np.float32))
print(type(tensor_x))
x = Variable(torch.ones(2), requires_grad=True)

var_x = Variable(tensor_x.double(), requires_grad=True)

y = f_B3d(3, 1)
yy = y.F(var_x)
yy.backward()

print(var_x.grad)
print()
# gy = grad(/, var_x, create_graph=True)
# # g2 = grad(gy, var_x, retain_graph=False)
# grad_norm = 0
# hessian_matrix = torch.zeros(3, 3)
#
# for item in gy:
#     print(item)
# for i in zip(range(3)):
#     tmp_tensor = torch.zeros(3)
#     tmp_tensor[i] = 1.0
#     matrix_i = grad(gy, var_x, grad_outputs=tmp_tensor, retain_graph=True)
#     hessian_matrix[i, :] = matrix_i[0]
#
# hessian_matrix1 = torch.zeros(3, 3)


# print(hessian_matrix)