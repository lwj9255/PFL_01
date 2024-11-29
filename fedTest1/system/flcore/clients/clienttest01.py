from collections import OrderedDict

import numpy as np
import time
import torch
import torch.nn as nn
import copy
from flcore.clients.clientbase import Client
from torch.autograd import Variable

from flcore.trainmodel.models import BaseHeadSplit


class clientTest01(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args = args
        self.customized_model = copy.deepcopy(self.model)  # 定制的全局模型初始时等于全局模型
        self.k = args.kk_weight # 控制阈值的松紧程度的超参数k
        self.alpha = args.kk_alpha # 平滑因子

        trainloader = self.load_train_data()  # 获得DataLoader
        for x, y in trainloader:
            x = x.to(self.device)  # 将输入数据和标签移动到指定设备上（例如 GPU），以加速计算
            y = y.to(self.device)
            with torch.no_grad():  # 为了计算初始特征表示，不需要梯度计算
                rep = self.model.base(x).detach()  # 初始化时先通过模型的基础部分输入本地数据得到初始特征表示rep
                # .detach()用于将张量从计算图中分离出来，以确保不会记录用于反向传播的梯度
                # 这里的base就是去掉了fc层的FedAvgCNN模型，因此rep的维度为 (batch_size, 512)
            break  # break：只需要处理一个批次的数据来初始化，因此取到第一个批次后直接跳出循环

        # client_mean 客户端表示个性化器，将特征提取器输出的初始表示与表示个性化器相加得到个性化表示
        # torch.zeros_like(rep[0]) 与rep[0]的形状一样一样，是一个维度(512,)的全0向量
        # Variable() 用于包装张量，以便追踪其计算图并计算梯度
        # nn.Parameter() 用于创建可学习参数，意味着它会成为模型的一部分，并且会在训练过程中被更新。
        self.client_mean = nn.Parameter(Variable(torch.zeros_like(rep[0])))

        # opt_client_mean 是一个优化器，用于优化客户端的特征均值 client_mean
        # [self.client_mean]中的[]表示将 client_mean 作为需要更新的参数传递给优化器
        self.opt_client_mean = torch.optim.SGD([self.client_mean], lr=self.learning_rate)

        # 初始化梯度矩阵和参数敏感度矩阵
        self.update_grad = copy.deepcopy(args.model.state_dict())
        self.previous_grad = copy.deepcopy(args.model.state_dict())
        self.parameter_sensitivity = copy.deepcopy(args.model.state_dict())
        self.previous_parameter_sensitivity = copy.deepcopy(args.model.state_dict())
        # 初始化前轮模型参数
        self.previous_model_params = copy.deepcopy(args.model.state_dict())
    def train(self):
        # 保存上轮的梯度
        self.previous_grad = copy.deepcopy(self.update_grad)

        trainloader = self.load_train_data()  # 加载训练数据

        start_time = time.time() # 记录训练开始的时间

        self.model.train() #  将模型设置为训练模式，训练模式下开启Dropout 和 BatchNorm 的行为

        max_local_epochs = self.local_epochs  # 设定本地训练轮次

        # 本地训练过程
        for epoch in range(max_local_epochs):  # 遍历每个训练周期
            for i, (x, y) in enumerate(trainloader):  # 遍历训练数据
                           # enumerate 用于将一个可迭代对象组合为一个索引序列，同时返回元素的索引和值。
                x = x.to(self.device)  # 将数据移动到设备上
                y = y.to(self.device)  # 将标签移动到设备上
                # ====== 训练过程开始 ======
                # ====== 前向传播 ======
                rep = self.model.base(x)  # 使用基础部分计算特征表示
                output = self.model.head(rep + self.client_mean) # 将初始表示与客户端个性化表示结合，通过模型的头部得到最终输出
                loss = self.loss(output, y)  # 计算损失

                # ====== 清空梯度 ======
                self.opt_client_mean.zero_grad()  # 清零 opt_client_mean优化器 的梯度
                self.optimizer.zero_grad()  # 清空梯度

                # ====== 反向传播 ======
                loss.backward()  # 反向传播计算梯度

                # ====== 参数更新 ======
                self.optimizer.step()  # 更新模型参数
                self.opt_client_mean.step()  # 更新 client_mean 参数

        # ====== 在整个训练结束后，记录最后一轮的梯度到 updated_grad ======
        self.update_gradient()

        # 第一次不计算参数敏感度
        if self.train_time_cost['训练的轮次数'] > 0:
            self.calculate_sensitivity()

        self.train_time_cost['训练的轮次数'] += 1  # 训练回合数加1
        self.train_time_cost['累计训练所花费的总时间'] += time.time() - start_time  # 训练时间累加


    # 记录base和head的梯度
    def update_gradient(self):

        # 获取当前模型的参数字典
        current_model_params = self.model.state_dict()

        # 计算当前模型参数与之前模型参数之间的差值
        for param_name in current_model_params:
            self.update_grad[param_name] = current_model_params[param_name] - self.previous_model_params[param_name]

        self.previous_model_params = copy.deepcopy(current_model_params)

    # 计算参数敏感度
    def calculate_sensitivity(self):
        # 计算当前梯度和上一轮梯度的差异
        grad_diff = {}

        # 遍历更新的梯度和上一轮的梯度
        for name, current_grad in self.update_grad.items(): # 用items返回模型字典的键值对
            # 找到相应的上一轮的梯度
            prev_grad = self.previous_grad[name]

            # 确保梯度不相等
            if not torch.equal(current_grad, prev_grad):
                # 计算梯度差异
                grad_diff[name] = current_grad.data - prev_grad.data

                # 获取当前模型中对应参数的绝对值，用于计算敏感度
                current_param = self.model.state_dict()[name]

                current_param_abs = current_param.abs()

                epsilon = 1e-8  # 设置一个小常数，避免除以零

                # 如果有元素为零，用epsilon替代
                safe_current_param_abs = torch.where(current_param_abs == 0,
                                                     torch.tensor(epsilon, device=current_param.device),
                                                     current_param_abs)

                # 计算参数敏感度：梯度差异除以当前参数的绝对值
                self.parameter_sensitivity[name] = grad_diff[name] / safe_current_param_abs

                # 判断是否有上一轮的参数敏感度进行聚合
                if self.train_time_cost['训练的轮次数'] > 1:
                    # 滑动加权：alpha * 当前敏感度 + (1 - alpha) * 上一轮敏感度
                    self.parameter_sensitivity[name] = self.alpha * self.parameter_sensitivity[name] + \
                                                       (1 - self.alpha) * self.previous_parameter_sensitivity[name]

            # 将当前轮的参数敏感度保存为下一轮的上一轮敏感度
            self.previous_parameter_sensitivity = copy.deepcopy(self.parameter_sensitivity)
            self.previous_parameter_sensitivity = copy.deepcopy(self.parameter_sensitivity)

            # 归一化参数敏感度
            max_sensitivity = max([sensitivity.max().item() for sensitivity in self.parameter_sensitivity.values()])
            min_sensitivity = min([sensitivity.min().item() for sensitivity in self.parameter_sensitivity.values()])

            # 归一化到[0, 1]范围
            for name, sensitivity in self.parameter_sensitivity.items():
                if max_sensitivity - min_sensitivity != 0:
                    self.parameter_sensitivity[name] = (sensitivity - min_sensitivity) / (
                                max_sensitivity - min_sensitivity)

    # 设置模型参数
    def set_parameters(self, model):
        if self.train_time_cost['训练的轮次数'] > 1:  # 如果训练轮次>1，说明已经计算出了参数敏感度矩阵了
            index = 0  # 初始化索引，用于遍历每个模型参数
            # 使用zip将当前模型(self.model)、传入的模型(model)和定制的模型(self.customized_model)的参数配对
            for (
                    (name1, param1),
                    (name2, param2),
                    (name3, param3),
            ) in zip (
                    self.model.named_parameters(), # named_parameters()方法用于获取模型中所有的参数，返回的是一个生成器，每次迭代时返回的是一个二元组
                    model.named_parameters(), # 每次返回包含：name-参数的名称，通常是该参数所属层的名称，例如 conv1.weight 或 fc1.bias。
                    self.customized_model.named_parameters(), # parameter-参数本身，包含该层的权重或偏置，可以通过 .data 或 .grad 访问这些参数的数值和梯度。
            ):
                # 从parameter_sensitivity字典中获取对应参数名的敏感度
                if name1 in self.parameter_sensitivity:
                    sensitivity = self.parameter_sensitivity[name1]  # 获取对应的敏感度张量
                else:
                    raise ValueError(f"未找到敏感度矩阵对应的参数: {name1}")

                # 确保敏感度矩阵的形状与参数形状一致
                if sensitivity.shape != param1.shape:
                    raise ValueError(f"敏感度矩阵和参数形状不匹配: {sensitivity.shape} vs {param1.shape}")
                # 计算每个参数的值，结合敏感度矩阵调整参数
                param1.data = (sensitivity.data * param3.data  # 敏感度矩阵值 × 定制模型参数
                               + (1 - sensitivity.data) * param2.data)  # (1 - 敏感度矩阵值) × 全局模型参数
                # 更新索引，指向下一个参数
                index += 1

        else:
            super().set_parameters(model)

    # 评估关键参数
    def evaluate_critical_parameter(self, parameter_sensitivity: nn.Module, k: float):
        r"""
        Overview:
             实现关键参数选择
        """
        global_mask = []  # 全局掩码，用于标记非关键参数
        local_mask = []  # 本地掩码，用于标记关键参数
        critical_parameter = []  # 记录关键参数

        # 计算敏感度的均值和标准差
        all_sensitivities = torch.cat(
            [param.data.view(-1) for name, param in parameter_sensitivity.named_parameters() if
             'base' in name or 'head' in name])
        mean_sensitivity = all_sensitivities.mean().item()  # 将张量转换为标量
        std_sensitivity = all_sensitivities.std().item()  # 将张量转换为标量

        # 设置最小阈值
        min_threshold = 1e-5

        # 计算关键参数的阈值
        threshold = max(mean_sensitivity + k * std_sensitivity, min_threshold)

        # 遍历模型的每一层参数，并根据敏感度选择关键参数
        for name, sensitivity in parameter_sensitivity.named_parameters():
            if 'base' in name or 'head' in name:  # 仅对 base 和 head 部分进行关键参数选择
                c = sensitivity.data  # 获取参数的敏感度

                # 获取本地掩码和全局掩码
                mask = (c >= threshold).int().to('cpu')  # 参数敏感度矩阵 c 中，大于等于阈值的为1，小于的为0
                global_mask.append((c < threshold).int().to('cpu'))  # 标记非关键参数
                local_mask.append(mask)  # 保存本地掩码，也就是关键参数掩码
                critical_parameter.append(mask.view(-1))  # 展平并保存关键参数掩码

        critical_parameter = torch.cat(critical_parameter)  # 合并所有关键参数

        return critical_parameter, global_mask, local_mask  # 返回关键参数和掩码