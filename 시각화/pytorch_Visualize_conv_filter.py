#출처: https://discuss.pytorch.org/t/visualize-feature-map/29597/2
#다른 방법으로 구현한 코드 : https://stackoverflow.com/questions/55594969/how-to-visualise-filters-in-a-cnn-with-pytorch
#마지막 필터를 시각화 해보면 3개의 필터가 나온다. 1개가 나올지 알았는데 생각해보니 당연한 이야기다. 1개가 나오는건 activation function 이다.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import math

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, 1, 1) #input 채널, output 채널..
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(3, 6, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2)

        self.conv_trans1 = nn.ConvTranspose2d(6, 3, 4, 2, 1)
        self.conv_trans2 = nn.ConvTranspose2d(3, 1, 4, 2, 1)

    def forward(self, x):
        x = F.relu(self.pool1(self.econv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.conv_trans1(x))
        x = self.conv_trans2(x)
        return x

model = torch.load('model.pt')

kernels = model.conv_trans2.weight.detach()

# # Visualize conv filter #가로로 시각화
# fig, axarr = plt.subplots(kernels.size(0))
# for idx in range(kernels.size(0)):
#     axarr[idx].imshow(kernels[idx].squeeze())
#


# Visualize conv filter #사각형으로 시각화
for idx in range(kernels.size(0)):
    plt.subplot(int(math.ceil(math.sqrt(kernels.size(0)))), int(math.ceil(math.sqrt(kernels.size(0)))), idx+1)  # 가로, 세로 나눠서 보기 좋게 배열
    a = kernels[idx]
    plt.imshow(kernels[idx].squeeze())

plt.show()