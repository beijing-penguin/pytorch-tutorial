import torch
import torch.nn as nn

lin=nn.Linear(5,3)#maps from R^5 to R^3
print(lin)
data=torch.randn(1,5)
print("weight=",lin.weight)
print('bias=:', lin.bias)

print("data=",data)
output=lin(data)

print('output.shape:', output.shape)

print(output)