# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file

# 设置种子，以保证random模块和numpy的np.random模块中的随机生成函数每次运行生成的随机数是一样的
random.seed(1)
np.random.seed(1)

# 联邦学习中客户端的数量
num_clients = 20
# 存储 MNIST 数据集的目录路径，也就是当前路径下的MNIST文件夹中
dir_path = "MNIST/"


# Allocate data to users
# generate_dataset 函数：
# dir_path：数据目录，如果不存在就创建一个
# num_clients：客户端数量
# balance：数据是否平均分配
# partition：分配策略
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    # 如果dir_path不存在，则创建该目录
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # 定义配置文件和数据存储路径
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    # 用check函数检查数据和配置是否已经存在，存在则直接返回，避免重复生成数据
    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # # FIX HTTP Error 403: Forbidden
    # from six.moves import urllib
    # opener = urllib.request.build_opener()
    # opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    # urllib.request.install_opener(opener)

    # transforms.Compose([])是一个组合变换函数，把多个图像变换步骤组合在一起;此处把ToTensor()和Normalize()组合起来
    # 1.transforms.ToTensor()变换会将图像从PIL.Image 或 numpy.ndarray 转换为 torch.Tensor;
    # MNIST中的图像是灰度图，像素值范围为 0 到 255，ToTensor() 会将这些像素值归一化为 0 到 1 之间的浮点数。
    # 2.transforms.Normalize([],[])的作用是对图像数据进行标准化处理，即调整张量中的数值到某个范围;normalized_value=(tensor_value−mean)/std_dev
    # 这样做的效果是将像素值（范围 0 到 1）转换为 -1 到 1 的范围。如果像素值是 1.0，标准化后是 (1.0 - 0.5) / 0.5 = 1。
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    # 使用 torchvision 下载并处理 MNIST 数据集，应用 transform 将图像转换为张量并进行标准化。
    trainset = torchvision.datasets.MNIST(
        # train=True表示加载的是训练数据；download=True如果本地没找到数据集，就从互联网上下载MNIST数据集
        # transform=transform使用变换对象对加载的数据进行预处理，最后会处理成[-1.1]的范围
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)

    # shuffle=False表示不对数据进行随机打乱
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    # 将数据加载到 trainset 和 testset 对象中。
    # enumerate(iterable, start)用于对一个可迭代对象进行遍历时获取每个元素的索引和对应的值：iterable 是要遍历的可迭代对象，start 是索引的起始值，默认是 0
    # 例如此处，trainloader 只有一个批次的数据，因此enumerate返回的是(0, (train_data, train_labels))，用_来接收索引，意为忽略索引
    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    # 将训练集和测试集的数据和标签合并，并转换为 NumPy 数组以便后续处理。
    dataset_image = []
    dataset_label = []
    # .cpu()将存储在 GPU 上的张量转移到 CPU 上，当需要进行数据处理或后处理和保存时比较方便和高效
    # .detach()将张量与计算图断开，意为不计算梯度
    # .numpy()把pytorch张量（tensor）转换成numpy数组
    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    # 将列表转换成numpy数组
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # 打印数据集中类别的数量（MNIST 数据集有 10 个类别）。
    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # 使用separate_data方法分配数据
    # X：一个列表，其中包含 num_clients 个子列表，每个子列表对应一个客户端的数据子集。每个子列表中的数据为图像数据（dataset_image）。
    # y：一个列表，与 X 对应，其中包含 num_clients 个子列表，每个子列表对应一个客户端的标签子集。每个子列表中的数据为标签数据（dataset_label）。
    # statistic：一个列表，与 X 和 y 对应，其中包含 num_clients 个子列表。每个子列表记录了每个客户端的标签分布统计信息。例如，统计信息包括每个标签的数量。
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=2)

    # 使用split_data方法把数据分割成训练集和测试集
    train_data, test_data = split_data(X, y)

    # 使用save_file方法将数据和配置保存到指定的路径
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)

# 主程序执行
# 从命令行参数中获取数据分配策略（niid,balance,partition)并调用generate_dataset函数生成数据
if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False # 如果是noniid 则为true 否则为false
    balance = True if sys.argv[2] == "balance" else False # 如果是balance则为true 否则为false
    partition = sys.argv[3] if sys.argv[3] != "-" else None # 如果不是- 则为输入的分配策略 否则为none

    generate_dataset(dir_path, num_clients, niid, balance, partition)