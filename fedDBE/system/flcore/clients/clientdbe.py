

import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from torch.autograd import Variable


class clientDBE(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.klw = args.kl_weight # 正则化项MR的权重κ
        self.momentum = args.momentum # 动量超参数μ，控制当前批次对均值估计的贡献
        self.global_mean = None

        trainloader = self.load_train_data() # 获得DataLoader
        for x, y in trainloader:

            # if type(x) == type([]): 用mnist、cifar等数据集，x不会是一个列表，因此注释掉
            #     x[0] = x[0].to(self.device)
            # else:
            #     x = x.to(self.device)

            x = x.to(self.device) # 将输入数据和标签移动到指定设备上（例如 GPU），以加速计算
            y = y.to(self.device)
            with torch.no_grad(): # 为了计算初始特征表示，不需要梯度计算
                rep = self.model.base(x).detach() # 初始化时先通过模型的基础部分输入本地数据得到初始特征表示rep
                                                  # .detach()用于将张量从计算图中分离出来，以确保不会记录用于反向传播的梯度
                                                  # 这里的base就是去掉了fc层的FedAvgCNN模型，因此rep的维度为 (batch_size, 512)
            break # break：只需要处理一个批次的数据来初始化，因此取到第一个批次后直接跳出循环

        # running_mean 用于跟踪当前客户端的特征均值
        self.running_mean = torch.zeros_like(rep[0]) # rep[0] 表示从批次中选择第一个样本的初始特征表示，维度是(512,)
                                                     # 创建一个和 rep[0] 形状相同的全0向量

        # num_batches_tracked 用于跟踪已经处理的批次数量，张量，初始值为 0
        self.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=self.device)

        # client_mean 用于存储客户端特征均值的可训练参数
        # torch.zeros_like(rep[0]) 与 running_mean一样，是一个维度(512,)的全0向量
        # Variable() 用于包装张量，以便追踪其计算图并计算梯度
        # nn.Parameter() 用于创建可学习参数，意味着它会成为模型的一部分，并且会在训练过程中被更新。
        self.client_mean = nn.Parameter(Variable(torch.zeros_like(rep[0])))

        # opt_client_mean 是一个优化器，用于优化客户端的特征均值 client_mean
        # [self.client_mean]中的[]表示将 client_mean 作为需要更新的参数传递给优化器
        self.opt_client_mean = torch.optim.SGD([self.client_mean], lr=self.learning_rate)


    def train(self):
        trainloader = self.load_train_data() # 上面其实已经获取过了，是为了获取rep的形状来设置 running_mean 和 client_mean
                                             # 这里再获取一次，为了正式开始训练过程，说是分开管理比较好

        self.model.train() # 将模型设置为训练模式。这样可以确保在训练时启用 dropout 和 batch normalization

        start_time = time.time() # 记录训练开始时间

        max_local_epochs = self.local_epochs # 设定本地训练轮次

        # if self.train_slow: 是否慢训练，默认是关的
        #     max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        self.reset_running_stats() # 重置运行中的统计信息（清零running_mean特征均值和num_batches_tracked已经处理的批次数量）

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                # ====== 训练过程开始 ======
                rep = self.model.base(x)  # 使用基础部分计算特征表示
                running_mean = torch.mean(rep, dim=0)  # 计算当前批次特征的均值
                                                       # rep为(batch_size, 512)
                                                       # rep = [
                                                       #   [0.1, 0.3, 0.5, ..., 0.7],  # 第1个样本的特征向量
                                                       #   [0.2, 0.1, 0.6, ..., 0.8],  # 第2个样本的特征向量
                                                       #   。。。。。。。。。。。。。。。
                                                       #   [0.4, 0.5, 0.4, ..., 0.6]   # 第batch_size个样本的特征向量
                                                       # ]
                                                       # dim=0计算的是批次维度的均值，得到每个特征在当前batch_size个样本中的均值
                                                       # running_mean = [
                                                       #   mean(0.1, 0.2,..., 0.4),  # 第 1 个特征位置的均值
                                                       #   mean(0.3, 0.1, ...,0.5),  # 第 2 个特征位置的均值
                                                       #   mean(0.5, 0.6, ...,0.4),  # 第 3 个特征位置的均值
                                                       #   ...
                                                       #   mean(0.7, 0.8, ...,0.6)   # 第 512 个特征位置的均值
                                                       # ]

                # 更新已经处理的批次数量
                if self.num_batches_tracked is not None: # 确保这个变量已经被正确初始化
                    self.num_batches_tracked.add_(1) # 已经处理的批次数量+1

                # 用文献中的公式9更新表示均值z_g，z_g = (1-μ)*z_g_old + μ*z_g_new
                # running_mean是上面计算的新的表示均值 z_g_new
                # self.running_mean是上一轮的表示均值 z_g_old
                # self.momentum就是动量超参数 μ，初始化时就定义好了的，值在主函数里设置
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * running_mean

                if self.global_mean is not None:
                    # 计算正则化损失
                    reg_loss = torch.mean(0.5 * (self.running_mean - self.global_mean) ** 2)
                    # 将特征与客户端均值结合，通过模型的头部得到最终输出
                    output = self.model.head(rep + self.client_mean)
                    # 计算总损失
                    loss = self.loss(output, y) + reg_loss * self.klw
                else:
                    # 没有全局均值时，也就是初始化阶段，直接通过模型的头部计算输出
                    output = self.model.head(rep)
                    loss = self.loss(output, y)
                # ====== 训练过程结束 ======

                # 梯度清零与反向传播
                # opt_client_mean在上面定义的，负责的参数是client_mean
                self.opt_client_mean.zero_grad()  # 清零 opt_client_mean优化器 的梯度

                # self.optimizer在base类中定义好了，是一个SGD优化器，负责的参数是模型的基础部分以及头部部分
                self.optimizer.zero_grad()  # 清零 optimizer优化器 的梯度

                loss.backward()  # 反向传播计算梯度

                self.optimizer.step()  # 更新模型的基础部分以及头部部分参数
                self.opt_client_mean.step()  # 更新 client_mean 参数

                # 把running_mean从计算图中分离，作为一个统计张量，每轮计算完都从计算图中分离，避免计算图的膨胀
                self.detach_running()

        # if self.learning_rate_decay: # 是否开启学习率衰减，默认是不开启的
        #     self.learning_rate_scheduler.step()

        self.train_time_cost['训练的轮次数'] += 1
        self.train_time_cost['累计训练所花费的总时间'] += time.time() - start_time


    # 重置训练时的统计信息
    def reset_running_stats(self): # 。zero_()用于将张量的所有元素重置为零
        self.running_mean.zero_() # 重置均值统计
        self.num_batches_tracked.zero_() # 重新开始计数

    # 把 running_mean 从计算图中分离
    def detach_running(self):
        self.running_mean.detach_()

    def train_metrics(self):
        trainloader = self.load_train_data()
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
                rep = self.model.base(x)
                output = self.model.head(rep + self.client_mean)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        reps = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                output = self.model.head(rep + self.client_mean)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)
                reps.extend(rep.detach())

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc