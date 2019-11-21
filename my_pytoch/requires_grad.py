import torch
from builtins import print
x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)

print(y.grad_fn)

z = y * y * 3
out = z.mean()
print("out = ",out)
print(z, out)

print("out = ",out)
out.backward()
print("x.grad = ",x.grad)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print("a = ",a)
a.requires_grad_(True)
b = (a * a).sum()
print("b = ",b)
print(b.grad_fn)

