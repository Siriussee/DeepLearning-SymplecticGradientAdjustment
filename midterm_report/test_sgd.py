import torch
import torch.nn as nn
#import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import Function
import my_sgd

lr = 0.01

# Data
x = Variable(torch.tensor(1., requires_grad=True))
y = Variable(torch.tensor(1., requires_grad=True))

optimizer1 = my_sgd.my_SGD([x], lr=lr)
optimizer2 = my_sgd.my_SGD([y], lr=lr)

class loss1_fn(Function):

    def forward(self, x, y):
        self.save_for_backward(x, y)
        return 0.5*x**2 + 10*x*y

    def backward(self, grad_output):
        x, y = self.saved_tensors
        gard_x = x + 10*y
        gard_y = y - 10*x
        return (gard_x, gard_y)

class loss2_fn(Function):

    def forward(self, x, y):
        return 0.5*y**2 - 10*x*y

    def backward(self, x, y):
        return y - 10*x

# Run training
niter = 3
for _ in range(0, niter):
	optimizer1.zero_grad()
	loss1 = loss1_fn(x, y)
	loss1.backward()
	optimizer1.step()

	print("-" * 50)
	print("error1 = {}".format(loss1))
