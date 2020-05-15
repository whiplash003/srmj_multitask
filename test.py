
# -----------------------------------test 1----------------------------------------
import os

# print("获取当前文件路径——" + os.path.realpath(__file__))  # 获取当前文件路径
#
# parent = os.path.dirname(os.path.realpath(__file__))
# print("获取其父目录——" + parent)  # 从当前文件路径中获取目录
#
# garder = os.path.dirname(parent)
# print("获取父目录的父目录——" + garder)
# print("获取文件名" + os.path.basename(os.path.realpath(__file__)))  # 获取文件名
#
# # 当前文件的路径
# pwd = os.getcwd()
# print("当前运行文件路径" + pwd)
#
# # 当前文件的父路径
# father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
# print("运行文件父路径" + father_path)
#
# # 当前文件的前两级目录
# grader_father = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
# print("运行文件父路径的父路径" + grader_father)

# -----------------------------------test 2----------------------------------------
# # 查看最小label文件的长度
# import numpy as np
# path = os.getcwd()
# min_length = 100
# for i in range(10000):
#     file = path + '/data/labels/effect_tile' + str(i) + '.npy'
#     length = len(np.load(file))
#     print('第{}个label文件：len is {}'.format(i, length))
#     if length < min_length:
#         min_length = length
# print('min_length is {}'.format(min_length))

# -----------------------------------test 3----------------------------------------
# # 查看data和label文件的长度是否一致
# import numpy  as np
# path_data = os.getcwd()+'/data/data/'
# path_label = os.getcwd()+'/data/labels/'
# d1 = np.load(path_data+'train0.npy')
# l1 = np.load(path_label+'effect_tile0.npy')
# for i in range(10000):
#     data_file = np.load(path_data+'train'+str(i)+'.npy')
#     data_label = np.load(path_label+'effect_tile'+str(i)+'.npy')
#     if data_file.shape[0] != data_label.shape[0]:
#         print(i, ' ', data_file.shape, ' ', data_label.shape)
#         # np.save(path_data+'train'+str(i)+'.npy',d1)
#         # np.save(path_data+'effect_tile'+str(i)+'.npy',l1)

# -----------------------------------test 4----------------------------------------
# 测试visdom是否有用
from visdom import Visdom
import numpy as np
import time

# # 将窗口类实例化
# viz = Visdom()
#
# # 创建窗口并初始化
# viz.line([0.], [0], win='train_loss', opts=dict(title='train_loss'))
#
# for global_steps in range(10):
#     # 随机获取loss值
#     loss = 0.2 * np.random.randn() + 1
#     # 更新窗口图像
#     viz.line([loss], [global_steps], win='train_loss', update='append')
#     time.sleep(0.5)