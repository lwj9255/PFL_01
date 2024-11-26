import numpy as np
import time
import torch
import torch.nn as nn
import copy
from flcore.clients.clientbase import Client

class clientCAC(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args = args
        self.critical_parameter = None  # 记录关键参数位置
        self.customized_model = copy.deepcopy(self.model)  # 定制的全局模型初始时等于全局模型
        self.critical_parameter, self.global_mask, self.local_mask = None, None, None
                                    # 全局掩码           本地掩码

    def train(self):
        trainloader = self.load_train_data()  # 加载训练数据

        start_time = time.time() # 记录训练开始的时间

        # 记录模型在本地更新之前的状态，用于选择关键参数
        initial_model = copy.deepcopy(self.model)


        self.model.train() #  将模型设置为训练模式，训练模式下开启Dropout 和 BatchNorm 的行为

        max_local_epochs = self.local_epochs  # 最大本地训练轮次，这里根据文献设置为5

        if self.train_slow:  # 如果设置了train_slow，默认是没设置的
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)  # 训练轮次随机减半

        # 本地训练过程
        for epoch in range(max_local_epochs):  # 遍历每个训练周期
            for i, (x, y) in enumerate(trainloader):  # 遍历训练数据
                           # enumerate 用于将一个可迭代对象组合为一个索引序列，同时返回元素的索引和值。
                if type(x) == type([]):  # 如果x是一个列表
                    x[0] = x[0].to(self.device)  # 将数据移动到设备上
                else:
                    x = x.to(self.device)  # 将数据移动到设备上
                y = y.to(self.device)  # 将标签移动到设备上

                if self.train_slow:  # 如果启用慢训练，默认不启动
                    time.sleep(0.1 * np.abs(np.random.rand()))  # 延迟训练，模拟慢训练

                output = self.model(x)  # 前向传播
                loss = self.loss(output, y)  # 计算损失
                self.optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播计算梯度
                self.optimizer.step()  # 更新模型参数

        if self.learning_rate_decay:  # 如果启用了学习率衰减，默认没启动
            self.learning_rate_scheduler.step()  # 更新学习率

        # 选择关键参数
        self.critical_parameter, self.global_mask, self.local_mask = self.evaluate_critical_parameter(
            prevModel=initial_model, model=self.model, tau=self.args.tau
        )

        self.train_time_cost['num_rounds'] += 1  # 训练回合数加1
        self.train_time_cost['total_cost'] += time.time() - start_time  # 训练时间累加

    # 选择关键参数
    def evaluate_critical_parameter(self, prevModel: nn.Module, model: nn.Module, tau: float):
        r"""
        Overview:
             实现关键参数选择
        """
        global_mask = []  # 全局掩码，用于标记非关键参数
        local_mask = []  # 本地掩码，用于标记关键参数
        critical_parameter = []  # 记录关键参数

        # 遍历模型的每一层参数，并比较前一个模型和当前模型的变化
        for (name1, prevparam), (name2, param) in zip(prevModel.named_parameters(), model.named_parameters()):
            g = (param.data - prevparam.data)  # 计算参数的变化量
            v = param.data  # 获取当前参数的值
            c = torch.abs(g * v)  # 计算变化量和当前值的乘积的绝对值，作为参数敏感度

            metric = c.view(-1)  # 将参数敏感度拉平为一维
            num_params = metric.size(0)  # 获取参数总数，size(0)返回维度0的大小
            nz = int(tau * num_params)  # 选择前tau比例的参数敏感度
            top_values, _ = torch.topk(metric, nz)  # 选出前nz个参数敏感度
                                                    # torch.topk（m,k）用于从张量 m 中找到前 k 个最大值的方法
                                                    # 返回前 k 个最大值和它们在原始张量 m 中的索引(values, indices)
            thresh = top_values[-1] if len(top_values) > 0 else np.inf  # 设置关键参数阈值
                                                                        # 如果前 nz 个参数敏感度不为空，则选择前 nz 个里最小的作为关键参数阈值，
                                                                        # 如果前 nz 个参数敏感度为空，则选择np.inf 作为关键参数阈值，inf 表示正无穷大
            # 如果阈值等于0，选择非零的最小元素作为阈值
            if thresh <= 1e-10: # 1e-10 是一个很小的值，通常表示阈值几乎为零
                new_metric = metric[metric > 1e-20] # 过滤出所有大于 1e-20 的参数敏感度
                if len(new_metric) == 0:  # 如果所有度量值都为零
                    print(f'异常!!! 参数敏感度metric:{metric}')
                else:
                    thresh = new_metric.sort()[0][0] # new_metric 按升序排序，选择最小的敏感度作为关键参数阈值

            # 获取本地掩码和全局掩码
            mask = (c >= thresh).int().to('cpu')  # 参数敏感度矩阵 c 中，大于等于关键参数阈值 thresh 的为1，小的为0
            global_mask.append((c < thresh).int().to('cpu'))  # 标记非关键参数
            local_mask.append(mask)  # 保存本地掩码，也就是关键参数掩码
            critical_parameter.append(mask.view(-1))  # 展平并保存关键参数掩码

        model.zero_grad()  # 清空模型的梯度

        critical_parameter = torch.cat(critical_parameter)  # 合并所有关键参数

        return critical_parameter, global_mask, local_mask  # 返回关键参数和掩码

    # 设置模型参数
    def set_parameters(self, model):
        if self.local_mask != None:  # 如果本地掩码不为空，说明需要使用掩码调整模型参数
            index = 0  # 初始化索引，用于遍历每个模型参数
            # 使用zip将当前模型(self.model)、传入的模型(model)和定制的模型(self.customized_model)的参数配对
            for (
                    (name1, param1),
                    (name2, param2),
                    (name3, param3)
            ) in zip (
                    self.model.named_parameters(), # named_parameters()方法用于获取模型中所有的参数，返回的是一个生成器，每次迭代时返回的是一个二元组
                    model.named_parameters(), # 每次返回包含：name-参数的名称，通常是该参数所属层的名称，例如 conv1.weight 或 fc1.bias。
                    self.customized_model.named_parameters() # parameter-参数本身，包含该层的权重或偏置，可以通过 .data 或 .grad 访问这些参数的数值和梯度。
            ):
                # 计算每个参数的值，结合本地掩码和全局掩码调整参数
                param1.data = (self.local_mask[index].to(self.device).float()
                               *
                               param3.data # 本地掩码（关键参数掩码）×定制模型参数
                               +
                               self.global_mask[index].to(self.args.device).float()
                               *
                               param2.data) # 全局掩码（非关键参数掩码）×全局模型参数
                # 更新索引，指向下一个参数
                index += 1

        else:
            # 如果本地掩码为空，直接调用父类Client中的set_parameters方法
            super().set_parameters(model)
