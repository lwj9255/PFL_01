import h5py
import numpy as np
import os


def average_data(algorithm="", dataset="", goal="", times=10):
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times)

    max_accurancy = []
    # 记录每次的最佳准确率
    for i in range(times):
        max_accurancy.append(test_acc[i].max())

    print("最佳准确率的标准差:", np.std(max_accurancy))
    print("最佳准确率的平均值:", np.mean(max_accurancy))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        test_acc.append(np.array(read_data_then_delete(file_name, delete=False)))

    return test_acc


def read_data_then_delete(file_name, delete=False):
    file_path = "../results/" + file_name + ".h5"

    # 用h5py.File打开指定路径下的.h5文件，模式为只读
    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_path)
    # 打印读取到的数据长度
    print("长度: ", len(rs_test_acc))

    return rs_test_acc