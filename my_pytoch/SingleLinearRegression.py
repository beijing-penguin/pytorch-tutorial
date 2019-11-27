import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.86], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

class SingleLinearRegression(nn.Module):
    def __init__(self):
        super(SingleLinearRegression, self).__init__()
        self.regression = nn.Linear(1, 1)

    def forward(self, x):
        out = self.regression(x)
        return out
model = SingleLinearRegression()

MSELoss = nn.MSELoss()
print(model.parameters())
print(dir(model))
optimizer = optim.SGD(model.parameters(), lr=1e-3)
epoch = 2000
for i in range(epoch):
   
    ## 6. 获取模型的输出值
    out = model(x_train)

    ## 7. 得到损失函数值
    loss = MSELoss(y_train, out)
    if i % 100 == 0:
        print('| Epoch[ {} / {} ], loss: {:.6f}'.format(i + 1, epoch, loss.item()))

    ## 8. 清空参数的所有梯度
    optimizer.zero_grad()

    ## 9. 计算梯度值
    loss.backward()

    ## 10. 跟新参数
    optimizer.step()

    
x_train = Variable(x_train)
model.eval()
predict = model(x_train)
predict = predict.data.numpy()

plt.plot(x_train.data.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(x_train.data.numpy(), predict, label='Fitting Line')
plt.show()