import time
import torch
import copy

from flcore.clients.clientcac import clientCAC
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data

class FedCAC(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        args.beta = int(args.beta) # β：其他客户端对关键参数的帮助程度
        # 随机选择慢客户端（但现在没开）
        self.set_slow_clients() # 根据train_slow_rate标记哪些客户端在训练阶段是慢的，根据send_slow_rate标记哪些在发送阶段是慢的

        self.set_clients(clientCAC) # 设置客户端，传入 clientCAC 类

        print(f"\n参与率 / 总客户端数: {self.join_ratio} / {self.num_clients}")
        print("完成创建服务端和客户端")

        self.Budget = [] # 初始化空的 Budget 列表，用于存储时间开销

        # To be consistent with the existing pipeline interface. Maintaining an epoch counter.
        self.epoch = -1 # 设置epoch 为 -1，这和self.local_epoch不是同一个

    def train(self):
        for i in range(self.global_rounds+1):
            self.epoch = i # 更新当前的 epoch 值
            s_t = time.time() # 记录当前时间，计算训练时间
            self.selected_clients = self.select_clients() # 这里随机参与率默认为false，每轮参与比例为1，因此这里就是所有客户端的序列
            self.send_models() # 如果是第一轮，客户端把全局模型参数复制到本地
                               # 如果不是第一轮，

            if i%self.eval_gap == 0: # eval_gap：评估的轮次间隔，默认为1，也就是每轮都评估一次
                print(f"\n-------------轮次: {i}-------------")
                print("\n评估个性化模型")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, '本轮时间花费', '-'*25, self.Budget[-1])

            # auto_break默认是false
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\n最佳精度.")
        print(max(self.rs_test_acc))

        print("\n每回合的平均时间成本.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()

        if self.num_new_clients > 0: # num_new_clients：新加入的客户端数量，默认是0
            self.eval_new_clients = True
            self.set_new_clients(clientCAC)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def get_customized_global_models(self): # 为客户聚合定制的全局模型以协作关键参数

        # assert 后面跟着的条件必须是 True，如果条件为 False，程序会抛出一个 AssertionError 异常
        assert type(self.args.beta) == int and self.args.beta >= 1  # 确保 beta 参数是整数且大于等于 1

        overlap_buffer = [[] for i in range(self.args.num_clients)]  # 初始化一个列表，用于存储客户端之间关键参数位置的重叠率

        # 计算客户端 i 与客户端 j 之间的重叠率
        for i in range(self.args.num_clients):  # 遍历所有客户端
            for j in range(self.args.num_clients):  # 遍历所有客户端
                if i == j:  # 跳过客户端 i 和自己之间的比较
                    continue

                # 计算重叠率
                overlap_rate = (1
                                -
                                torch.sum(
                                        torch.abs( # 计算两个关键参数矩阵之间，对于每个相同位置的元素，计算它们的绝对差值。
                                            self.clients[i].critical_parameter # 客户端 i 的关键参数掩码
                                            .to(self.device) # 将参数传输到指定的device上进行计算，默认的是cuda
                                            -
                                            self.clients[j].critical_parameter # 客户端 j 的关键参数掩码
                                            .to(self.args.device))
                                )
                                /
                                float(
                                    torch.sum(
                                        self.clients[i].critical_parameter.to(self.args.device) # 客户端 i 的关键参数掩码
                                    )
                                    .cpu() * 2
                                )
                )
                overlap_buffer[i].append(overlap_rate)  # 将 i 和 j 之间的重叠率保存到 overlap_buffer[i]

        # 计算全局阈值
        overlap_buffer_tensor = torch.tensor(overlap_buffer)  # 将 overlap_buffer 转换为 PyTorch 张量，方便后续可以利用 PyTorch 提供的高效张量操作。
        overlap_sum = overlap_buffer_tensor.sum()  # 计算所有重叠率的总和
        overlap_avg = overlap_sum / ((self.args.num_clients - 1) * self.args.num_clients)  # 计算所有客户端之间的平均重叠率 O_arg
        overlap_max = overlap_buffer_tensor.max()  # 获取最大的重叠率 O_max
        threshold = overlap_avg + (self.epoch + 1) / self.args.beta * (overlap_max - overlap_avg)  # 计算阈值,按照fedcac中的公式

        # 为每个客户端计算定制化的全局模型
        for i in range(self.args.num_clients):  # 对每个客户端 i 进行操作
            w_customized_global = copy.deepcopy(self.clients[i].model.state_dict())  # 深拷贝客户端 i 的模型参数
                                                                                     # state_dict是模型的状态字典
            collaboration_clients = [i]  # 初始化协作的客户端列表，首先将客户端 i 加入
            index = 0  # 单独维护一个索引，因为在客户端i的重叠率矩阵overlap_buffer[i]中，从第i个开始，overlap_buffer[i][i]是i对客户端i+1的重叠率
                       # 当遍历到 i 时，跳过，这样index就不会+1，轮到i+1时，对应的重叠率索引是index=i，就是正确的
            # 查找与客户端 i 的关键参数位置相似的客户端
            for j in range(self.args.num_clients):  # 遍历所有客户端
                if i == j:  # 跳过与自己比较
                    continue
                if overlap_buffer[i][index] >= threshold:  # 如果客户端 i 和客户端 j 之间的重叠率大于阈值
                    collaboration_clients.append(j)  # 将客户端 j 加入到协作客户端列表
                index += 1  # 增加索引

            # 聚合来自协作客户端的模型
            for key in w_customized_global.keys():  # 遍历状态字典的每个键，键是对应的模型参数的名称（例如 layer1.weight、layer1.bias），值是对应的参数张量。
                for client in collaboration_clients:  # 遍历所有协作的客户端
                    if client == i:  # 跳过客户端 i 自己
                        continue
                    w_customized_global[key] += self.clients[client].model.state_dict()[key]  # 累加其他客户端的模型参数
                w_customized_global[key] = torch.div(w_customized_global[key],
                                                     float(len(collaboration_clients)))  # 对所有协作客户端的参数求平均

            # 将计算得到的定制化全局模型发送给客户端 i
            self.clients[i].customized_model.load_state_dict(w_customized_global)  # 将定制化的全局模型参数加载到客户端 i 的模型中
                                                                                   # load_state_dict用于加载参数字典（state_dict）到模型中

    # 发送模型给客户端
    def send_models(self):
        if self.epoch != 0:  # 如果当前不是第0轮（即训练不是刚开始时）
            self.get_customized_global_models()  # 获取定制化的全局模型

        super().send_models()  # 调用父类 Server 的 send_models 方法
