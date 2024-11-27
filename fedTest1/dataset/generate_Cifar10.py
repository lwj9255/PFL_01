# 导入所需的库和模块
import numpy as np  # 用于处理数组和数学运算
import os  # 用于与操作系统交互，处理文件和目录
import sys  # 用于访问命令行参数
import random  # 用于生成随机数
import torch  # 用于深度学习框架 PyTorch
import torchvision  # 用于计算机视觉相关功能
import torchvision.transforms as transforms  # 图像转换工具
from utils.dataset_utils import check, separate_data, split_data, save_file  # 从自定义的 dataset_utils 模块导入函数

# 设置随机种子，以保证实验的可复现性
random.seed(1)
np.random.seed(1)

# 设置一些超参数
num_clients = 20  # 设置客户端数量为 20
dir_path = "Cifar10/"  # 设置数据存储目录路径为 "Cifar10/"

# 数据分配的函数
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    # 如果数据存储目录不存在，则创建该目录
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 设置训练和测试数据的目录路径
    config_path = dir_path + "config.json"  # 配置文件路径
    train_path = dir_path + "train/"  # 训练数据存储路径
    test_path = dir_path + "test/"  # 测试数据存储路径

    # 检查是否已经生成过数据，如果已经生成，则跳过
    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # 设置数据转换操作，主要是将图像数据标准化并转为 tensor 格式
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # 下载并加载 CIFAR-10 数据集
    trainset = torchvision.datasets.CIFAR10(
        root=dir_path+"rawdata", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=dir_path+"rawdata", train=False, download=True, transform=transform
    )

    # 加载数据到 DataLoader 中，便于后续的批处理操作
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False
    )

    # 将训练集和测试集的数据及标签赋值给 trainset 和 testset 对象
    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    # 初始化数据列表，用于存储所有图像数据和对应的标签
    dataset_image = []
    dataset_label = []

    # 将训练集和测试集的图像数据和标签分别加入列表
    dataset_image.extend(trainset.data.cpu().detach().numpy())  # 添加训练集图像数据
    dataset_image.extend(testset.data.cpu().detach().numpy())  # 添加测试集图像数据
    dataset_label.extend(trainset.targets.cpu().detach().numpy())  # 添加训练集标签
    dataset_label.extend(testset.targets.cpu().detach().numpy())  # 添加测试集标签

    # 转换为 numpy 数组
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # 输出类别数
    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # 数据分割：将数据按照客户端数目进行分配
    X, y, statistic = separate_data(
        (dataset_image, dataset_label), num_clients, num_classes, niid, balance, partition, class_per_client=2
    )

    # 将数据按客户端分割为训练集和测试集
    train_data, test_data = split_data(X, y)

    # 保存生成的数据和配置文件
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, statistic, niid, balance, partition)

# 主程序入口
if __name__ == "__main__":
    # 从命令行参数获取 niid, balance 和 partition 设置
    niid = True if sys.argv[1] == "noniid" else False  # 根据第一个参数决定是否为非独立同分布（noniid）
    balance = True if sys.argv[2] == "balance" else False  # 根据第二个参数决定是否为平衡分配
    partition = sys.argv[3] if sys.argv[3] != "-" else None  # 根据第三个参数设置分区方式

    # 调用数据生成函数
    generate_dataset(dir_path, num_clients, niid, balance, partition)