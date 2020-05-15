
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
import math
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
# from PIL import Image
from random import randrange
import torch.nn.functional as F
from sklearn.metrics import f1_score
from prefetch_generator import BackgroundGenerator
# import pandas as pd
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

# 定义多任务模型的class
class multi_output_model(torch.nn.Module):
    def __init__(self, model_core, dd):
        super(multi_output_model, self).__init__()

        self.resnet_model = model_core

        self.x1 = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x1.weight)

        self.bn1 = nn.BatchNorm1d(256, eps=1e-2)
        # self.x2 =  nn.Linear(128,64)
        # nn.init.xavier_normal_(self.x2.weight)
        # self.x3 =  nn.Linear(64,32)
        # nn.init.xavier_normal_(self.x3.weight)
        # comp head 1

        # heads
        self.y1o = nn.Linear(256, 34)
        nn.init.xavier_normal_(self.y1o.weight)  #
        self.y2o = nn.Linear(256, 34)
        nn.init.xavier_normal_(self.y2o.weight)
        self.y3o = nn.Linear(256, 34)
        nn.init.xavier_normal_(self.y3o.weight)

        # self.d_out = nn.Dropout(dd)

    def forward(self, x):
        x = self.resnet_model(x)
        # x1 = F.relu(self.x1(x))
        x1 = self.bn1(F.relu(self.x1(x)))
        # x = F.relu(self.x2(x))
        # x1 = F.relu(self.x3(x))

        # heads
        y1o = torch.sigmoid(self.y1o(x1))  # should be sigmoid
        y2o = torch.sigmoid(self.y2o(x1))  # should be sigmoid
        y3o = torch.sigmoid(self.y3o(x1))  # should be sigmoid
        # y1o = self.y1o(x1)
        # y2o = self.y2o(x1)
        # y3o = self.y3o(x1)
        # y4o = self.y4o(x1)
        # y5o = self.y5o(x1) #should be sigmoid|

        return y1o, y2o, y3o


# data labels path
path = os.getcwd()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 使用resnet50预训练模型
model_ft = models.resnet101(pretrained=True)
# 修改输入层的通道数为8
w = model_ft.conv1.weight.clone()
model_ft.conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model_ft.conv1.weight = torch.nn.Parameter(torch.cat((w, torch.zeros(64, 5, 7, 7)), dim=1))
model_ft.avgpool = nn.AdaptiveAvgPool2d(1)

# print(model_ft)
# num_ftrs = model_ft.classifier[6].in_features

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 512)


# 构建有效牌的多任务模型
dd = .4
model_1 = multi_output_model(model_ft, dd)
model_1 = model_1.to(device)
model_1.load_state_dict(torch.load('model/resnet_split_lr_1-0001.pth'))
model_1.eval()

# 取data_name文件的第i条记录
def data_loader(data_name, i):
    """load numpy, returns cuda tensor"""
    data = np.load(os.getcwd()+'/data/data/' + data_name)
    data = torch.from_numpy(data)
    print(data.shape)
    data = data[0].unsqueeze(0)
    data = Variable(data.float(), requires_grad=False)
    return data.cuda()  # assumes that you're using GPU


# train0.npy effect_tile0.npy | 第0条记录
data = data_loader('train0.npy',0)
y_gt = np.load(path + '/data/labels/effect_tile0.npy')
y_gt = y_gt[0]

y_pred = model_1(data)


print(y_gt)
print(y_pred)
