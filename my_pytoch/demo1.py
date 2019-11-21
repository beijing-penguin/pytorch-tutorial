import torch
import torchvision
print(torchvision)
print(torch.__version__)
a = torch.rand(5,3);
print(a[0][0])

b = torch.empty(5, 3)
print(b)

c = torch.zeros(5, 3, dtype=torch.long)
print(c)

#直接构造一个张量
d = torch.tensor([5.5, 3])
print(d)

e = d.new_ones(5, 3, dtype=torch.double)
# new_* methods take in sizes
print(e)
f = torch.randn_like(e, dtype=torch.float)

# override dtype!
print(f)

# result has the same size

print(f.size())


g = torch.rand(5, 3)
print(f + g)
print(torch.add(f, g))


result = torch.empty(5, 3)
torch.add(f, g,out=result);
print(result)
print(f.add(g))
h=torch.rand(5,3)
print(h)
#倒数第二列
print(h[:, 1])
#如果你有一个元素 tensor ，使用 .item() 来获得这个 value 。
#randn标准正太分布、rand平均分布
j = torch.randn(1)
print(j)
print(j.item())

help(torch.randn)