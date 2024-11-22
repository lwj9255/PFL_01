
import time
from flcore.clients.clientdbe import clientDBE
from flcore.servers.serverbase import Server
from threading import Thread


class FedDBE(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # 选择慢客户端，slow_rate默认是0，因此会返回一个客户端数量长度的全是false的列表
        self.set_slow_clients()

        # 初始化阶段
        self.set_clients(clientDBE) # 创建客户端对象并加入客户端列表clients[]中
        self.selected_clients = self.clients

        for client in self.selected_clients:
            client.train() # 进行初始化阶段，第一轮训练是无DBE的

        self.uploaded_ids = [] # 用于存储本轮中上传的客户端 ID
        self.uploaded_weights = [] # 用于存储每个上传的客户端的权重
        tot_samples = 0 # 记录样本总数

        for client in self.selected_clients:
            tot_samples += client.train_samples # 累计所有客户端的样本数，用于计算权重比例
            self.uploaded_ids.append(client.id) # 记录客户端ID
            self.uploaded_weights.append(client.train_samples) # 记录每个客户端的样本数
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples # 将每个客户端的样本数量除以总样本数，以计算该客户端的相对权重
            
        global_mean = 0 # 全局表示均值，初始值为0

        for cid, w in zip(self.uploaded_ids, self.uploaded_weights):
            global_mean += self.clients[cid].running_mean * w # 每个客户端的相对权重 * 它自己的局部表示均值 累加 等于全局表示均值
        print('>>>> 全局表示 <<<<', global_mean)

        for client in self.selected_clients:
            client.global_mean = global_mean.data.clone() # 使用 clone() 方法复制全局表示均值，以防止后续修改影响其他客户端，这样每个客户端都有自己独立的全局均值副本

        print(f"\n客户端参与率 / 客户端总数: {self.join_ratio} / {self.num_clients}")
        print("完成创建服务端和客户端")

        self.Budget = []

        print('客户端的特征均值 client_mean 的形状: ', self.clients[0].client_mean.shape) # clientDBE类中定义了client_mean是和running_mean的形状一样，(512,)
        print('特征均值张量的元素数目: ', self.clients[0].client_mean.numel()) # 打印的是元素数目，应该是512


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time() # 记录开始时间
            self.selected_clients = self.select_clients() # 选择客户端，但random_join_ratio为false，因此selected_clients就是全部客户端的列表
            self.send_models() # 把服务端的模型（body+head）克隆给所有客户端

            if i%self.eval_gap == 0: # eval_gap 评估间隔，默认是1，也就是每轮都评估
                print(f"\n-------------当前轮次: {i}-------------")
                print("\n评估模型")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()
