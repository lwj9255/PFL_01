

import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data


class Client(object):
    """
    这是联邦学习中客户端的基类。
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)  # 设置随机种子为 0，确保实验可复现
        self.model = copy.deepcopy(args.model)  # 深拷贝传入的模型，避免修改原始模型
        self.algorithm = args.algorithm  # 获取使用的算法
        self.dataset = args.dataset  # 获取数据集名称
        self.device = args.device  # 获取设备信息（CPU 或 GPU）
        self.id = id  # 客户端的唯一标识符
        self.save_folder_name = args.save_folder_name  # 保存文件的文件夹名称

        self.num_classes = args.num_classes  # 类别数量
        self.train_samples = train_samples  # 训练样本数量
        self.test_samples = test_samples  # 测试样本数量
        self.batch_size = args.batch_size  # 批量大小
        self.learning_rate = args.local_learning_rate  # 学习率
        self.local_epochs = args.local_epochs  # 本地训练的轮数

        # 检查模型中是否有 BatchNorm 层
        self.has_BatchNorm = False  # 默认没有 BatchNorm 层
        for layer in self.model.children():  # 遍历模型的所有层
            if isinstance(layer, nn.BatchNorm2d):  # 如果某层是 BatchNorm2d
                self.has_BatchNorm = True  # 标记为有 BatchNorm
                break  # 找到后跳出循环

        # 客户端的额外配置
        self.train_slow = kwargs['train_slow']  # 是否慢训练
        self.send_slow = kwargs['send_slow']  # 是否慢发送
        self.train_time_cost = {'训练的轮次数': 0, '累计训练所花费的总时间': 0.0}  # 本地训练的时间成本
        self.send_time_cost = {'发送的轮次数': 0, '累计发送所花费的总时间': 0.0}  # 数据发送的时间成本

        self.loss = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)  # 使用 SGD 优化器
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR( # 学习率调度器，它通过指数衰减的方式来调整学习率
            optimizer=self.optimizer,  # 前面定义的优化器
            gamma=args.learning_rate_decay_gamma  # 学习率衰减比例，默认是0.99
        )
        self.learning_rate_decay = args.learning_rate_decay  # 学习率是否启用衰减，默认是false

    def load_train_data(self, batch_size=None):
        """
        加载训练数据
        """
        if batch_size == None:  # 如果没有传入批量大小，使用默认的批量大小
            batch_size = self.batch_size
        # 调用外部函数加载客户端训练数据
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size=batch_size, shuffle=True)  # 使用 DataLoader 封装训练数据
                                                                            # shuffle: 如果为 True，则在每个 epoch 开始时会将数据集打乱。

    def load_test_data(self, batch_size=None):
        """
        加载测试数据
        """
        if batch_size == None:  # 如果没有传入批量大小，使用默认的批量大小
            batch_size = self.batch_size
        # 调用外部函数加载客户端测试数据
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size=batch_size, shuffle=False)  # 使用 DataLoader 封装测试数据

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()): # 传入的模型是新的，自己原本的是旧的
            old_param.data = new_param.data.clone() # 新模型参数覆盖旧模型参数

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        # 把模型设置为评估模式
        self.model.eval()

        # 累积预测正确的样本数量
        test_acc = 0
        # 累积测试样本的总数
        test_num = 0
        # 用于存储模型对每个样本的预测概率分布
        y_prob = []
        # 用于存储每个样本的真实标签
        y_true = []

        # 评估过程不计算梯度
        with torch.no_grad():
            for x, y in testloaderfull:
                # 检查x是否是一个列表
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                # 统计正确的样本数量
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                # 统计测试集总样本数量
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())

                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "servers" + ".pt"))
