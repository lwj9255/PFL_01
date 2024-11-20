import os
import sys

# sys.path.append(os.path.join(os.getcwd(), ''))

import logging
import warnings
import torch
import time
import copy
import numpy as np
import os
import argparse
import torchvision

from fedCAC.system.flcore.servers.servercac import FedCAC
from flcore.servers.serverdbe import FedDBE
from flcore.trainmodel.resnet8 import ResNet8

from flcore.trainmodel.transformer import *
from flcore.trainmodel.models import *

from utils.mem_utils import MemReporter
from utils.result_utils import average_data
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader

# 记录日志，只处理错误级别及其更高级别的信息
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

# 忽略警告
warnings.simplefilter("ignore")
# 用于设置 PyTorch 随机数生成器的种子，确保结果的可重复性
torch.manual_seed(0)

def run(args):
    # 用来记录每次实验的运行时间
    time_list = []
    # 创建一个 MemReporter 实例，用于报告内存使用情况
    reporter = MemReporter()
    # 从 args 中获取模型类型（如 "cnn"）
    model_str = args.model

    # 从之前的实验次数prev(默认是0）运行到实验次数times，times默认是1
    for i in range(args.prev, args.times):
        print(f"\n============= Running time实验次数: {i}th =============")
        print("创建客户端和服务端 ...")
        start = time.time()

        # 生成模型
        if model_str == "resnet8":
            args.model = ResNet8(num_classes=args.num_classes).to(args.device)
        elif model_str == "cnn":
            if "MNIST" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        print(f"模型：")
        print(args.model)

        # 选择算法
        if args.algorithm == "FedCAC":
            server = FedCAC(args, i)

        elif args.algorithm == "FedDBE":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedDBE(args, i)

        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time() - start) # 记录

    print(f"\n平均实验耗时: {round(np.average(time_list), 2)}s.")

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("完成!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()  # 记录脚本开始运行的时间。

    parser = argparse.ArgumentParser()  # 使用argparse 模块创建一个 命令行参数解析器
    # 通用参数
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="实验的目标")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["nv", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0",
                        help="当有多个GPU时，使用哪个")
    parser.add_argument('-data', "--dataset", type=str, default="Cifar10",
                        help="使用的数据集名称")
    parser.add_argument('-nb', "--num_classes", type=int, default=10,
                        help="数据集中类别的数量")
    parser.add_argument('-m', "--model", type=str, default="resnet8",
                        help="使用的模型架构，例如 'cnn'（卷积神经网络）或其他模型")
    parser.add_argument('-lbs', "--batch_size", type=int, default=100,
                        help="本地训练时的批处理大小，即每次训练中处理的数据样本数量")
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.1,
                        help="本地训练的学习率")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False,
                        help="是否启用学习率衰减")
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99,
                        help="学习率衰减的比例。如果启用了学习率衰减，则使用该值")
    parser.add_argument('-gr', "--global_rounds", type=int, default=500,
                        help="全局训练轮数，即所有客户端同步全局模型的总轮数")
    parser.add_argument('-ls', "--local_epochs", type=int, default=5,
                        help="每个客户端在本地训练中进行的轮次")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedCAC",
                        help="使用的算法，默认是'FedAvg'")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="每轮中参与训练的客户端比例")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="是否随机选择每轮参与训练的客户端比例")
    parser.add_argument('-nc', "--num_clients", type=int, default=40,
                        help="总的客户端数量")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="之前的运行次数")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="实验的运行次数")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="评估(evaluation)的轮次间隔")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items',
                        help="保存结果的文件夹名称")
    parser.add_argument('-ab', "--auto_break", type=bool, default=False,
                        help="是否启用自动停止机制，即根据某种条件（如收敛）自动结束训练")
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False,
                        help="是否启用 DLG 攻击评估")
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100,
                        help="DLG 评估的轮次间隔")
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2,
                        help="每个客户端的批次数量")
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0,
                        help="新加入的客户端数量")
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0,
                        help="新加入客户端的微调轮次")
    parser.add_argument('-fd', "--feature_dim", type=int, default=512,
                        help="特征的维度，例如模型的输出特征维度")
    parser.add_argument('-vs', "--vocab_size", type=int, default=98635,
                        help="词汇表的大小。适用于文本任务，不同的任务可能需要不同的词汇表大小")
    parser.add_argument('-ml', "--max_len", type=int, default=200,
                        help="文本的最大长度，通常用于处理文本数据时的截断或填充")

    # 模拟真实参数
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="训练过程中掉线的客户端比例")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="训练时慢客户端的比例，表示客户端训练的慢速情况")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="发送全局模型时慢客户端的比例")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="是否根据时间成本对客户端进行分组和选择")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="慢客户端的时间阈值，当客户端训练时间超过该值时，会被认为是慢客户端并可能被丢弃。")

    # FedCAC
    parser.add_argument('-bt', "--beta", type=float, default=100) # β：其他客户端对关键参数的帮助程度，β越大，对关键参数的帮助就越大
    parser.add_argument('-tau', "--tau", type=float, default=0.5) # 关键参数比例
    # FedDBE
    parser.add_argument('-mo', "--momentum", type=float, default=0.1)
    parser.add_argument('-klw', "--kl_weight", type=float, default=0.0)


    # 存储解析后的命令行参数
    args = parser.parse_args()

    # os.environ 是一个字典类型的对象，它允许你访问和修改环境变量
    # CUDA_VISIBLE_DEVICES 是一个环境变量，用于指定哪些 GPU 设备是可见的。设置这个环境变量可以控制深度学习框架使用哪些 GPU。
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    # 检查是否可以使用CUDA GPU
    # 如果设备为cuda且cuda设备不可用，就要切换成cpu
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorith算法: {}".format(args.algorithm))
    print("Local batch size本地批处理大小: {}".format(args.batch_size))  # 每次训练中所处理的数据样本数量
    print("Local epochs本地训练轮次: {}".format(args.local_epochs))  # 每个客户端本地模型的训练轮数
    print("Local learing rate本地学习率: {}".format(args.local_learning_rate))
    print("Local learing rate decay是否开启学习率衰减: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma学习率衰减的比例: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients总客户端数量: {}".format(args.num_clients))
    print("Clients join in each round每轮中参与训练的客户端比例: {}".format(args.join_ratio))
    print("Clients randomly join是否随机选择客户端来参与每轮训练: {}".format(args.random_join_ratio))
    print("Client drop rate客户端掉线率: {}".format(args.client_drop_rate))  # 参与训练但中途掉线的客户端比例
    print("Client select regarding time是否根据时间选择客户端: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold掉线时间阈值: {}".format(args.time_threthold))  # 超过这个时间的客户端可能被视为掉线
    print("Running times实验的运行次数: {}".format(args.times))
    print("Dataset数据集名称: {}".format(args.dataset))
    print("Number of classes数据集中的类别数量: {}".format(args.num_classes))
    print("Backbone模型架构: {}".format(args.model))
    print("Using device设备信息: {}".format(args.device))
    print("Auto break是否启用了自动停止机制: {}".format(args.auto_break))  # 即根据某种条件（如收敛等）自动结束训练。如果未启用自动停止，则继续训练到预定轮数。
    if not args.auto_break:
        print("Global rounds全局训练的轮数: {}".format(args.global_rounds))  # 如果未启用自动停止，输出全局训练的轮数，即所有客户端的全局模型同步的次数。
    if args.device == "cuda":  # 如果使用的是 GPU（cuda），输出所使用的 GPU 设备 ID。
        print("Cuda device id使用的GPU 设备 ID: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("DLG attack是否启用了 DLG攻击评估: {}".format(args.dlg_eval))  # DLG 是一种数据泄露攻击，测试模型是否泄露训练数据。
    if args.dlg_eval:  # 如果启用了 DLG 攻击评估，输出每隔多少轮进行一次 DLG 攻击评估
        print("DLG attack round gap DLG攻击评估间隔的轮次: {}".format(args.dlg_gap))
    print("Total number of new clients新增的客户端数量: {}".format(args.num_new_clients))  # 新客户端在后续轮次中加入训练
    print("Fine tuning epoches on new clients新客户端上微调模型的 epoch 数量: {}".format(args.fine_tuning_epoch_new))

    print("=" * 50)

    run(args)