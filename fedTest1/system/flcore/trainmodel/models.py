import torch
import torch.nn.functional as F
from torch import nn

batch_size = 10


# split an original model into a base and a head
class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__() # 调用父类 (nn.Module) 的初始化方法，让 BaseHeadSplit 类继承父类的所有属性和方法。
        self.base = base
        self.head = head

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)
        return out


###########################################################

class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential( # 输入数据的维度形状为：(batch_size, in_features, height, width)
            nn.Conv2d(in_features, # 以输入通道=1为例
                        32, # 输出通道=32
                        kernel_size=5, # 卷积核
                        padding=0, # 填充
                        stride=1, # 步幅
                        bias=True), # 偏置开启
            # 以MNIST为例，输入图像大小：28 x 28，输入通道=1，输入数据的维度形状为：(batch_size, 1, 28, 28)
            # 输出尺寸 =（输入尺寸-卷积核+填充*2）/步幅+1 = （28-5+0）/1 + 1 = 24
            # 尺寸=24，输出通道=32，输出数据的维度形状为：(batch_size, 32, 24, 24)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)) # 池化窗口大小 = 2*2，步幅默认等于窗口大小
                                             # 输出尺寸 =（输入尺寸-池化窗口）/步幅+1 = （24-2）2+1 = 12
                                             # 输出数据的维度形状为：(batch_size, 32, 12, 12)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True), # (batch_size, 64, 8, 8)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)) # (batch_size, 64, 4, 4)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512), # (batch_size, 512)
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes) # (batch_size, 10)

    def forward(self, x): # (batch_size, 1, 28, 28)
        out = self.conv1(x) # (batch_size, 32, 12, 12)
        out = self.conv2(out) # (batch_size, 64, 4, 4)
        out = torch.flatten(out, 1) # (batch_size, 1024)
        out = self.fc1(out) # (batch_size, 512)
        out = self.fc(out) # (batch_size, 10)
        return out

# ====================================================================================================================

