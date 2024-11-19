import os  # 导入os模块，用于处理文件路径和目录
import ujson  # 导入ujson（UltraJSON），用于快速JSON解析
import numpy as np  # 导入numpy，用于数组操作和数学计算
import gc  # 导入垃圾回收模块，用于管理内存
from sklearn.model_selection import train_test_split  # 从sklearn库导入train_test_split，用于数据集划分

batch_size = 100  # 默认批次大小
train_ratio = 0.75  # 训练集与测试集的划分比例
alpha = 0.1  # Dirichlet分布的超参数（用于非独立同分布数据生成）
# alpha较小时，数据分布非均匀，增强non-IID，推荐0.1是dir分配
# alpha较大时，数据分布更均匀，推荐100是exdir分配



# check方法：通过检查 config_path 是否存在来判断数据集配置是否已经生成。
# 如果配置文件存在，且配置文件中的参数（如客户端数量、非IID、平衡性、分区等）与当前输入的参数匹配，则认为数据集已经生成过，并返回 True。
# 如果不存在或不满足条件，就返回 False，并创建数据存储目录。
# 参数：
# config_path：数据集配置文件的路径。
# train_path 和 test_path：分别是训练数据和测试数据的路径。
# num_clients：客户端的数量。
# niid：是否是非独立同分布（non-IID）数据。
# balance：是否需要平衡数据。
# partition：分配数据的策略（例如：按类分配）。
def check(config_path, train_path, test_path, num_clients, niid=False, balance=True, partition=None):
    # 如果配置文件已存在，加载并验证配置是否与当前设置一致
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)  # 加载现有的配置文件
        # 检查配置是否与当前的参数匹配
        if config['num_clients'] == num_clients and \
                config['non_iid'] == niid and \
                config['balance'] == balance and \
                config['partition'] == partition and \
                config['alpha'] == alpha and \
                config['batch_size'] == batch_size:
            print("\n数据集已生成。\n")
            return True  # 如果数据集已经生成且配置一致，返回True

    # 确保训练集和测试集的路径所在目录存在，如果不存在则创建
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False  # 如果数据集未生成，返回False



# separate_data 方法：用于将整个数据集按照指定的策略分配给多个客户端。
# 参数：
# data：数据集，包括内容和标签。
# num_clients：客户端数量。
# num_classes：数据集中的类别数。
# niid：是否非IID（默认是否）
# balance：是否平衡（默认是否）
# partition：分配策略
# partition = 'pat' （默认）时，按类来分配样本；
# partition = 'dir' 时，使用 Dirichlet 分布来分配样本；
# partition = 'exdir' 是基于文献中的策略进行数据分配。
# class_per_client：每个客户端所拥有的类别数量
def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=None):
    X = [[] for _ in range(num_clients)]  # 列表生成式 [A for B in range(C)]:A表示生成的新列表中的元素，B表示依次取出range（C）中的每个值
    y = [[] for _ in range(num_clients)]  # 例：[ a*2 for b in range(c)] c=5 → [0,2,4,6,8]
    statistic = [[] for _ in range(num_clients)] # 创建三个空列表分别存储每个客户端的数据、每个客户端的标签、每个客户端的标签统计

    dataset_content, dataset_label = data # 数据和标签

    # 保证每个客户端至少有一批用于测试的数据。
    least_samples = int(min(batch_size / (1 - train_ratio), (len(dataset_label) / num_clients / 2)))

    dataidx_map = {} # 用于存储每个客户端的数据索引

    if not niid:
        partition = 'pat' # 如果不是非独立同分布数据，使用“pat”分配策略
        class_per_client = num_classes # 每个客户端分配的类别数=样本类别数

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))  # 长度等于数据集样本数量的数组，元素从0到样本数量-1，即获取样本的索引
        idx_for_each_class = []  # 一个空列表，存储每个类别的样本的索引
        for i in range(num_classes):
            idx_for_each_class.append(
                idxs[dataset_label == i])  # dataset_label是样本标签的数组，dataset_label==i会把数组内标签等于i的变成True,不等于i的变成False
            # idxs[每个样本是否属于类别i的bool数组]会返回所有索引为True的索引
            # 例：dataset_lable = [0,2,1,0,1,1,2,0] -> [dataset_label==0] = [T,F,F,T,F,F,F,T] -> idxs[dataset_label == 0] = [0,3,7]
            # idx_for_each_class=[ [0,3,7] , [2,4,5] , [1,6] ]表示样本中每个类别的索引
        class_num_per_client = [class_per_client for _ in range(num_clients)]  # class_num_per_client作为客户端分配类别列表
        # 每个客户端会分配到的类别数量 = class_per_client

        for i in range(num_classes):
            selected_clients = [] # 存储被选中的客户端
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client) # 选择有空余类别的客户端
            if len(selected_clients) == 0: # 没有客户端还有剩余类别才退出循环
                break

            # num_clients/num_classes表示每个类别平均可以被分配各多少个客户端
            # 乘每个客户端可以被分配到的类别数量，等于每个客户端平均分配到的类别数量
            # np.ceil(...)向上取整，返回浮点数，int()得到向上取整的整数
            # 例如：40个客户端，10个类别，每个客户端有2个类别，那么每个类别可以被分给40/10*2 = 8个客户端
            selected_clients = selected_clients[:int(np.ceil((num_clients / num_classes) * class_per_client))]

            num_all_samples = len(idx_for_each_class[i])  # 表示当前类别i的样本数量
            num_selected_clients = len(selected_clients)  # 当前被选中的客户端的数量
            num_per = num_all_samples / num_selected_clients  # 当前类别i的样本平均分配到每个选中的客户端
            if balance:  # 如果需要平衡，分配相等的样本数
                num_samples = [int(num_per) for _ in range(num_selected_clients - 1)] # 表示每个客户端可以分配到多少个样本
            else:
                num_samples = np.random.randint(max(num_per / 10, least_samples / num_classes), num_per,num_selected_clients - 1).tolist()
                # np.random.randint(最小值, 最大值, 生成的随机数数量)
                # least_samples/num_classes表示每个类别的最少样本数
                # num_selected_clients-1：-1是因为万一样本数量除不尽，最后一个客户端就拿剩余的
                # tolist把numpy数组转换成python列表

            num_samples.append(num_all_samples - sum(num_samples))  # 最后一个客户端拿剩余数量的样本

            idx = 0 # 初始化索引，用于遍历当前类别的所有样本
            for client, num_sample in zip(selected_clients, num_samples): # 遍历被选中的客户端和它们需要的样本数量
                # 如果当前客户端还没有分配任何数据，则为它分配当前类别的样本
                if client not in dataidx_map.keys():
                    # 通过索引，将当前类别的部分样本分配给该客户端
                    dataidx_map[client] = idx_for_each_class[i][idx: idx + num_sample]
                else:
                    # 如果当前客户端已经分配过数据，则将新的样本添加到该客户端的已有数据中
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx: idx + num_sample],
                                                    axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples:
            if try_cnt > 1:
                print(f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    
    elif partition == 'exdir':
        r'''This strategy comes from https://arxiv.org/abs/2311.03154
        See details in https://github.com/TsingZ0/PFLlib/issues/139

        This version in PFLlib is slightly different from the original version 
        Some changes are as follows:
        n_nets -> num_clients, n_class -> num_classes
        '''
        C = class_per_client
        
        '''The first level: allocate labels to clients
        clientidx_map (dict, {label: clientidx}), e.g., C=2, num_clients=5, num_classes=10
            {0: [0, 1], 1: [1, 2], 2: [2, 3], 3: [3, 4], 4: [4, 5], 5: [5, 6], 6: [6, 7], 7: [7, 8], 8: [8, 9], 9: [9, 0]}
        '''
        min_size_per_label = 0
        # You can adjust the `min_require_size_per_label` to meet you requirements
        min_require_size_per_label = max(C * num_clients // num_classes // 2, 1)
        if min_require_size_per_label < 1:
            raise ValueError
        clientidx_map = {}
        while min_size_per_label < min_require_size_per_label:
            # initialize
            for k in range(num_classes):
                clientidx_map[k] = []
            # allocate
            for i in range(num_clients):
                labelidx = np.random.choice(range(num_classes), C, replace=False)
                for k in labelidx:
                    clientidx_map[k].append(i)
            min_size_per_label = min([len(clientidx_map[k]) for k in range(num_classes)])
        
        '''The second level: allocate data idx'''
        dataidx_map = {}
        y_train = dataset_label
        min_size = 0
        min_require_size = 10
        K = num_classes
        N = len(y_train)
        print("\n*****clientidx_map*****")
        print(clientidx_map)
        print("\n*****Number of clients per label*****")
        print([len(clientidx_map[i]) for i in range(len(clientidx_map))])

        # ensure per client' sampling size >= min_require_size (is set to 10 originally in [3])
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_clients)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                # Balance
                # Case 1 (original case in Dir): Balance the number of sample per client
                proportions = np.array([p * (len(idx_j) < N / num_clients and j in clientidx_map[k]) for j, (p, idx_j) in enumerate(zip(proportions, idx_batch))])
                # Case 2: Don't balance
                #proportions = np.array([p * (j in label_netidx_map[k]) for j, (p, idx_j) in enumerate(zip(proportions, idx_batch))])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # process the remainder samples
                '''Note: Process the remainder data samples (yipeng, 2023-11-14).
                There are some cases that the samples of class k are not allocated completely, i.e., proportions[-1] < len(idx_k)
                In these cases, the remainder data samples are assigned to the last client in `clientidx_map[k]`.
                '''
                if proportions[-1] != len(idx_k):
                    for w in range(clientidx_map[k][-1], num_clients-1):
                        proportions[w] = len(idx_k)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))] 
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            dataidx_map[j] = idx_batch[j]
    
    else:
        raise NotImplementedError

    # 数据分配
    for client in range(num_clients):
        idxs = dataidx_map[client] # dataidx_map存储每个客户端分配到的数据索引，key是客户端序号，value是数据索引列表
        X[client] = dataset_content[idxs] # 将对应的数据样本赋值给当前客户端
        y[client] = dataset_label[idxs] # 将对应的标签赋值给当前客户端

        # 统计当前客户端标签的分布情况
        for i in np.unique(y[client]): # 遍历客户端分配到的标签的唯一值
            # 统计每个标签的样本数，并将结果添加到统计列表中
            statistic[client].append((int(i), int(sum(y[client]==i))))

    # 删除临时的数据，释放内存
    del data
    # gc.collect() # 可选，手动触发垃圾回收，释放内存

    # 打印每个客户端的数据统计
    for client in range(num_clients):
        print(f"客户端 {client}\t 数据大小: {len(X[client])}\t 标签种类: ", np.unique(y[client]))
        print(f"\t\t 样本标签统计: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic



# split_data 方法：该方法用于将每个客户端的数据集分为训练集和测试集，并返回分割后的数据。
#X：每个客户端的数据内容。
#y：每个客户端的标签。
def split_data(X, y):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train':[], 'test':[]}

    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_ratio, shuffle=True)

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("样本总数:", sum(num_samples['train'] + num_samples['test']))
    print("训练样本数量:", num_samples['train'])
    print("测试样本数量:", num_samples['test'])
    print()
    del X, y
    # gc.collect()

    return train_data, test_data


# save_file 方法：用于将分配好的数据集以及配置信息保存到指定路径下。
# config_path：配置文件路径。
# train_path 和 test_path：训练数据和测试数据的保存路径。
# train_data 和 test_data：已经分配并分割好的训练和测试数据。
# num_clients：客户端数量。
# num_classes：类别数量。
# statistic：每个客户端分配的数据统计信息。
# niid、balance、partition：是否为iID，是否平衡，分配策略
def save_file(config_path, train_path, test_path, train_data, test_data, num_clients,
                num_classes, statistic, niid=False, balance=True, partition=None):
    config = {
        'num_clients': num_clients, 
        'num_classes': num_classes, 
        'non_iid': niid, 
        'balance': balance, 
        'partition': partition, 
        'Size of samples for labels in clients': statistic, 
        'alpha': alpha, 
        'batch_size': batch_size, 
    }

    # gc.collect()
    print("保存到磁盘.\n")

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("完成创建数据集.\n")
