import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from utils.data_utils import read_client_data
from utils.dlg import DLG


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break # 自动停止，默认为false

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

    def set_clients(self, clientObj):
        # zip()函数将传入的多个可迭代对象按对应位置进行组合，返回一个元组迭代器
        # range(self.num_clients)：生成从 0 到 self.num_clients - 1 的整数序列，每个整数代表客户端的索引编号
        # 这里元组的元素分别是：客户端编号num_clients，是否训练时慢和是否发送时慢
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            # train_data和test_data分别通过调用read_client_data() 函数获取每个客户端的训练数据和测试数据
            # is_train=true表示读取训练数据
            train_data = read_client_data(self.dataset, i, is_train=True)
            # is_train=false表示读取测试数据
            test_data = read_client_data(self.dataset, i, is_train=False)
            # train_data和test_data只是为了计算样本数量的临时变量，当实例需要加载数据，再通过load_train_data或load_test_data方法动态加载
            client = clientObj(self.args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               train_slow=train_slow,
                               send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    # 根据给定的慢客户端的比例，随机选择哪些客户端是慢的，并返回一个标记列表
    def select_slow_clients(self, slow_rate):
        # 列表推导式 创建一个长度为num_clients，值全是false的列表
        slow_clients = [False for i in range(self.num_clients)]
        # 列表推导式 创建一个长度为num_clients，值是索引的列表
        idx = [i for i in range(self.num_clients)]
        # 使用了 numpy 库的 np.random.choice 函数，从列表idx中随机选择slow_rate * self.num_clients个索引
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        # 标记哪些客户端在训练阶段是慢的
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        # 标记哪些客户端在发送阶段是慢的
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        # random_join_ratio表示是否随机选择参与的客户端，默认为false
        # join_ratio表示每轮参与训练的客户端比例，默认为1
        # num_join_clients = num_clients * self.join_ratio 表示每轮参与训练的客户端数量
        # current_num_join_clients = num_join_clients 实时参与的客户端数量，如果设置了随机，则会改动

        # 如果为true，则客户端数量在num_join_clients和num_clients之间随机选择
        if self.random_join_ratio:
            # \是换行符，避免一行代码太长导致可读性差
            # np.random.choice 是 NumPy 库中的一个函数，作用是从指定的序列中随机选择元素
            # np.random.choice（序列，随机选择元素的个数，replace=false表示不允许重复选择）
            # 虽然只返回一个元素，但np.random.choice返回的是一个数组，因此需要[0]来取出这个元素
            self.current_num_join_clients = \
            np.random.choice(range(self.num_join_clients, self.num_clients + 1), 1, replace=False)[0]

        # 否则，直接使用num_join_clients作为客户端参与的数量
        else:
            self.current_num_join_clients = self.num_join_clients

        # 从客户端序列中随机选择实时参与客户端数量的客户端
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        # assert语句检查条件是否为真，如果为假，则抛出一个异常终止执行
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        # 上传客户端的ID
        self.uploaded_ids = []
        # 训练样本数量
        self.uploaded_weights = []
        self.uploaded_models = []
        # 总样本数量
        tot_samples = 0

        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                # 累加客户端的训练样本数量
                tot_samples += client.train_samples
                # 上传的客户端id放入列表
                self.uploaded_ids.append(client.id)
                # 往权重列表中添加训练样本数量
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            # 计算出了模型聚合时的权重占比
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)
        # 为了简化初始化，创建第一个客户端模型的副本作为全局模型的基础结构
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        # 将全局模型中所有参数的值设为0
        for param in self.global_model.parameters():
            param.data.zero_()
        # zip将权重占比和模型进行配对
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        # eval_new_clients是一个bool值，表示是否需要评估新加入的客户端
        if self.eval_new_clients and self.num_new_clients > 0:
            # 如果有新客户端并且需要评估，则对新客户端进行特殊处理
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        # 每个客户端的样本数量
        num_samples = []
        # 每个客户端预测正确的样本数量
        tot_correct = []
        # 每个客户端的AUC值
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]

        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("训练集平均损失: {:.4f}".format(train_loss))
        print("测试集平均准确率: {:.4f}".format(test_acc))
        print("测试集平均AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("每个客户端测试集准确率的标准差: {:.4f}".format(np.std(accs)))
        print("每个客户端测试集AUC的标准差: {:.4f}".format(np.std(aucs)))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))


    """
    判断训练过程是否已经满足停止条件
    acc_lss：一个列表，其中每个元素是一个记录训练进展的列表，通常是模型的精度或损失。
    top_cnt：一个整数，用于定义检查最近几轮数据的范围。
    div_value：一个浮点数，用于定义精度或损失波动的阈值。
    """
    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1

            # items.append((client_model, origin_grad, target_inputs))

        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               train_slow=False,
                               send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
