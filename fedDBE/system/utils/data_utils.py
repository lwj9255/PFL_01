import numpy as np
import os
import torch


def read_data(dataset, idx, is_train=True):
    if is_train:
        # 如果是训练
        # os.path.join 函数将不同路径拼接成一个完整的路径
        train_data_dir = os.path.join('../dataset', dataset, 'train/')
        # 拼接文件名，路径+客户端编号+.npz后缀
        train_file = train_data_dir + str(idx) + '.npz'
        # with open(train_file, 'rb') as f：以只读的二进制模式打开文件。
        with open(train_file, 'rb') as f:
            # 使用 NumPy 的 load 函数读取 .npz 文件，并且 allow_pickle=True 允许读取包含 Python 对象的数据（如列表、字典等）
            # ['data']：从 .npz 文件中提取键为 'data' 的内容
            # .tolist()：将 NumPy 数组转换为 Python 列表格式，以便后续处理更方便
            # MNIST中的npz文件中data对应的值是字典，字典中包含两个键x,y
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        # 测试
        test_data_dir = os.path.join('../dataset', dataset, 'test/')
        # 拼接文件名
        test_file = test_data_dir + str(idx) + '.npz'
        # 只读打开文件
        with open(test_file, 'rb') as f:
            # 先load读取，提取键为data的内容，并转换为列表
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset, idx, is_train=True):
    # idx是客户端索引，表明需要读取哪个客户端的数据
    # is_train为true表明读取训练数据，为false表明读取测试数据
    if "News" in dataset:
        return read_client_data_text(dataset, idx, is_train)
    elif "Shakespeare" in dataset:
        return read_client_data_Shakespeare(dataset, idx)

    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        # 返回包含(x,y)元组的列表
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def read_client_data_text(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)

        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data


def read_client_data_Shakespeare(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data

