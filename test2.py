import torch
from torch import Tensor
from torch.autograd import Variable
from torch.autograd import grad
from torch import nn

# some toy data
x = Variable(Tensor([4., 2.]), requires_grad=False)
y = Variable(Tensor([1.]), requires_grad=False)

# linear model and squared difference loss
model = nn.Linear(2, 1)
loss = torch.sum((y - model(x))**2)

# instead of using loss.backward(), use torch.autograd.grad() to compute gradients
loss_grads = grad(loss, model.parameters(), create_graph=True)

# compute the squared norm of the loss gradient
gn2 = sum([grd.norm()**2 for grd in loss_grads])

model.zero_grad()
gn2.backward()
print(model.parameters().grad)
