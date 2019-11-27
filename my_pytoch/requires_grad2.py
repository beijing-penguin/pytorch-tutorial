import torch
x=torch.tensor([[1.0,1.0],[1.0,1.0]], requires_grad=True)
w=torch.tensor([[2.0,2.0],[2.0,2.0]])
print("x=",x,sep="")
print("w=",w,sep="")

y=x*w
Y = torch.tensor([[4.0,4.0],[4.0,4.0]])
L = Y-y
print(torch.ones(2, 2, dtype=torch.float))
L.backward(torch.ones(2, 2, dtype=torch.float))
help(L.backward)
print("梯度=",x.grad,sep="")


target = torch.randn(10)  # 随机值作为样例
print(target)
target = target.view(1, -1)  # 使target和output的shape相同
print(target)