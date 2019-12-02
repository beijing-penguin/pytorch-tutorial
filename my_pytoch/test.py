import torch

#输入通道数，输出通道数，卷积核大小
c2 = torch.nn.Conv2d(1,10,(5,5)) # 10, 24x24
#样本数，输入通道大小，图片高，图片宽
x1 = torch.rand([1, 1, 10, 5]);
print(x1)
print(c2(x1))
print(c2(x1).shape)