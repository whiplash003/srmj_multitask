
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
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
import pandas as pd
from sklearn.model_selection import train_test_split
from visdom import Visdom
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


# num_data = 40000 # 麻将的局数*4（只取后4手，包含train和test）
batch_size = 256

# # 记录loss和acc
# train_loss = [[], [], [], []]
# val_loss = [[], [], [], []]
# train_acc = [[], [], []]
# val_acc = [[], [], []]

# 初始化loss和acc
viz = Visdom(env='srmj')
x, y = 0, 0
win1 = viz.line(X=np.array([x]), Y=np.array([[y,y,y,y]]),
                opts=dict(title='train_Loss',legend=['epoch_loss','opp1_loss','opp2_loss','opp3_loss']))
win2 = viz.line(X=np.array([x]), Y=np.array([[y,y,y,y]]),
                opts=dict(title='val_Loss',legend=['epoch_loss','opp1_loss','opp2_loss','opp3_loss']))
win3 = viz.line(X=np.array([x]), Y=np.array([[y,y,y]]),
                opts=dict(title='train_Acc',legend=['opp1_acc','opp2_acc','opp3_acc']))
win4 = viz.line(X=np.array([x]), Y=np.array([[y,y,y]]),
                opts=dict(title='val_Acc',legend=['opp1_acc','opp2_acc','opp3_acc']))

# 查看是否用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# path
data_path = os.getcwd() + '/data/data/'
labels_path = os.getcwd() + '/data/labels/'

# 自定义Dataset
class srmj_dataset(Dataset):
    def __init__(self, king_of_lists, transform=None):

        self.king_of_lists = king_of_lists
        self.transform = transform

    def __getitem__(self, index):

        # 只取了每局的后四手
        # x_numpy = np.load(
        #     data_path+'train' + str(math.floor(index / 4)) + '.npy')
        # x_numpy = x_numpy[-(index % 4 + 1)]
        #
        # y_label = np.load(labels_path+'effect_tile'+str(math.floor(index / 4)) + '.npy')
        # opp1_waiting = y_label[-(index % 4 + 1)][0]  # opp1_waiting
        # opp2_waiting = y_label[-(index % 4 + 1)][1]   # opp2_waiting
        # opp3_waiting = y_label[-(index % 4 + 1)][2]   # opp3_waiting

        # 取每局的每一手
        x_numpy = torch.from_numpy(self.king_of_lists[0][index])

        if self.transform is not None:
            x_numpy = self.transform(x_numpy)

        # list_of_labels = [torch.from_numpy(np.array(opp1_waiting)),
        #                   torch.from_numpy(np.array(opp2_waiting)),
        #                   torch.from_numpy(np.array(opp3_waiting))]
        list_of_labels = [torch.from_numpy(self.king_of_lists[1][index][0]),
                          torch.from_numpy(self.king_of_lists[1][index][1]),
                          torch.from_numpy(self.king_of_lists[1][index][2])]

        # list_of_labels = torch.FloatTensor(list_of_labels)
        # print(list_of_labels)

        return x_numpy, list_of_labels[0], list_of_labels[1], list_of_labels[2]

    def __len__(self):
        return len(self.king_of_lists[0])


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# 定义多任务模型的class
class multi_output_model(torch.nn.Module):
    def __init__(self, model_core, dd):
        super(multi_output_model, self).__init__()

        self.resnet_model = model_core

        self.x1 = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x1.weight)

        self.bn1 = nn.BatchNorm1d(256, eps=1e-2)

        self.x2 =  nn.Linear(256,128)
        nn.init.xavier_normal_(self.x2.weight)

        self.bn2 = nn.BatchNorm1d(128, eps=1e-2)

        self.x3 =  nn.Linear(128,64)
        nn.init.xavier_normal_(self.x3.weight)

        self.bn3 = nn.BatchNorm1d(64, eps=1e-2)

        # self.x3 =  nn.Linear(64,32)
        # nn.init.xavier_normal_(self.x3.weight)
        # comp head 1

        # heads
        self.y1o = nn.Linear(64, 34)
        nn.init.xavier_normal_(self.y1o.weight)  #
        self.y2o = nn.Linear(64, 34)
        nn.init.xavier_normal_(self.y2o.weight)
        self.y3o = nn.Linear(64, 34)
        nn.init.xavier_normal_(self.y3o.weight)

        # self.d_out = nn.Dropout(dd)

    def forward(self, x):
        x = self.resnet_model(x)
        # x1 = F.relu(self.x1(x))
        x1 = self.bn1(F.relu(self.x1(x)))
        # x = F.relu(self.x2(x))
        # x1 = F.relu(self.x3(x))
        x2 = self.bn2(F.relu(self.x2(x1)))
        x3 = self.bn3(F.relu(self.x3(x2)))

        # heads
        y1o = torch.sigmoid(self.y1o(x3))  # should be sigmoid
        y2o = torch.sigmoid(self.y2o(x3))  # should be sigmoid
        y3o = torch.sigmoid(self.y3o(x3))  # should be sigmoid

        # y1o = self.y1o(x1)
        # y2o = self.y2o(x1)
        # y3o = self.y3o(x1)
        # y4o = self.y4o(x1)
        # y5o = self.y5o(x1) #should be sigmoid|

        return y1o, y2o, y3o

# 训练模型函数
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 100

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_loss0 = 0.0
            running_loss1 = 0.0
            running_loss2 = 0.0

            running_corrects = 0
            opp1_corrects = []
            opp2_corrects = []
            opp3_corrects = []
            total_opp1 = []
            total_opp2 = []
            total_opp3 = []

            # Iterate over data.
            for inputs, opp1_waiting, opp2_waiting, opp3_waiting in dataloaders_dict[phase]:
                inputs = torch.tensor(inputs, dtype=torch.float32)
                inputs = inputs.to(device)
                # print(inputs.size())
                opp1_waiting = opp1_waiting.to(device)
                opp2_waiting = opp2_waiting.to(device)
                opp3_waiting = opp3_waiting.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # print(inputs)
                    outputs = model(inputs)

                    loss0 = criterion[0](outputs[0], opp1_waiting.float())
                    loss1 = criterion[1](outputs[1], opp2_waiting.float())
                    loss2 = criterion[2](outputs[2], opp3_waiting.float())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss = loss0 + loss1 + loss2
                        # print(loss, loss0,loss1, loss2, loss3,loss4)
                        loss.backward()
                        optimizer.step()

                # statisticsutputs[2]
                running_loss += loss.item() * inputs.size(0)
                running_loss0 += loss0.item() * inputs.size(0)
                running_loss1 += loss1.item() * inputs.size(0)
                running_loss2 += loss2.item() * inputs.size(0)

                opp1_corrects.append(
                    float((np.rint(outputs[0].cpu().detach().numpy()) == opp1_waiting.cpu().detach().numpy()).sum()))
                total_opp1.append(float((opp1_waiting.size()[0] * opp1_waiting.size(1))))
                opp2_corrects.append(
                    float((np.rint(outputs[1].cpu().detach().numpy()) == opp2_waiting.cpu().detach().numpy()).sum()))
                total_opp2.append(float((opp2_waiting.size()[0] * opp2_waiting.size(1))))
                opp3_corrects.append(
                    float((np.rint(outputs[2].cpu().detach().numpy()) == opp3_waiting.cpu().detach().numpy()).sum()))
                total_opp3.append(float((opp3_waiting.size()[0] * opp3_waiting.size(1))))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_loss0 = running_loss0 / dataset_sizes[phase]
            epoch_loss1 = running_loss1 / dataset_sizes[phase]
            epoch_loss2 = running_loss2 / dataset_sizes[phase]

            opp1_acc = float(sum(opp1_corrects)) / sum(total_opp1)
            opp2_acc = float(sum(opp2_corrects)) / sum(total_opp2)
            opp3_acc = float(sum(opp3_corrects)) / sum(total_opp3)

            # opp1_corrects_array = np.rint(outputs[0].cpu().detach().numpy())
            # opp1_acc = f1_score(opp1_waiting.cpu().float(), opp1_corrects_array,
            #                                  average='macro')
            # opp2_corrects_array = np.rint(outputs[1].cpu().detach().numpy())
            # opp2_acc = f1_score(opp2_waiting.cpu().float(), opp2_corrects_array,
            #                                  average='macro')
            # opp3_corrects_array = np.rint(outputs[2].cpu().detach().numpy())
            # opp3_acc = f1_score(opp3_waiting.cpu().float(), opp3_corrects_array,
            #                                   average='macro')

            print('{} epoch loss: {:.4f} opp1_waiting loss: {:.4f} '
                  'opp2_waiting loss: {:.4f} opp3_waiting loss: {:.4f} '.format(
                phase, epoch_loss, epoch_loss0,epoch_loss1, epoch_loss2,))
            print('{} opp1_corrects: {:.4f} '
                  'opp2_corrects: {:.4f}  opp3_corrects: {:.4f} '.format(
                phase, opp1_acc, opp2_acc, opp3_acc))

            # 添加loss和acc到数组中
            if phase == 'train':
                # train_loss[0].append(loss)
                # train_loss[1].append(loss0)
                # train_loss[2].append(loss1)
                # train_loss[3].append(loss2)
                # train_acc[0].append(opp1_waiting_corrects)
                # train_acc[1].append(opp2_waiting_corrects)
                # train_acc[2].append(opp3_waiting_corrects)

                # 更新loss acc曲线
                viz.line(X=np.array([epoch]), Y=np.array([[epoch_loss,epoch_loss0,epoch_loss1,epoch_loss2]]),
                         win=win1,  update='append')
                viz.line(X=np.array([epoch]), Y=np.array([[opp1_acc,opp2_acc,opp3_acc]]),
                         win=win3, update='append')
                # time.sleep(0.5)

            if phase == 'val':
                # val_loss[0].append(loss)
                # val_loss[1].append(loss0)
                # val_loss[2].append(loss1)
                # val_loss[3].append(loss2)
                # val_acc[0].append(opp1_waiting_corrects)
                # val_acc[1].append(opp2_waiting_corrects)
                # val_acc[2].append(opp3_waiting_corrects)

                # 更新loss acc曲线
                viz.line(X=np.array([epoch]), Y=np.array([[epoch_loss, epoch_loss0, epoch_loss1, epoch_loss2]]),
                         win=win2, update='append')
                viz.line(X=np.array([epoch]), Y=np.array([[opp1_acc, opp2_acc, opp3_acc]]),
                         win=win4, update='append')
            # time.sleep(0.5)

            # deep copy the model
            if phase == 'val' and epoch_loss < best_acc:
                print('saving with loss of {}'.format(epoch_loss),
                      'improved over previous {}'.format(best_acc))
                best_acc = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(float(best_acc)))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# 这里我是按顺序分割的train和test
# X = [i for i in range(num_data)]
# y = [i for i in range(num_data)]
# X_train, X_test = X[:math.floor(num_data*0.8)],X[math.floor(num_data*0.8):]
# y_train, y_test = y[:math.floor(num_data*0.8)],y[math.floor(num_data*0.8):]
# train_lists = [X_train, y_train]
# test_lists = [X_test, y_test]

# 按比例随机分割train核test
X = np.load(os.getcwd()+'/data/data_sum.npy')[:100000]
y = np.load(os.getcwd()+'/data/label_sum.npy')[:100000]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
train_lists = [X_train, y_train]
test_lists = [X_test, y_test]

# 构造好了train数据集和test数据集
training_dataset = srmj_dataset(king_of_lists = train_lists)
test_dataset = srmj_dataset(king_of_lists = test_lists )

print(len(X_train))
# 数据装载
dataloaders_dict = {'train': DataLoaderX(training_dataset, batch_size=batch_size, shuffle=True, num_workers=8,pin_memory=True),
                   'val':DataLoaderX(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8,pin_memory=True)
                   }
dataset_sizes = {'train':len(train_lists[0]),
                'val':len(test_lists[0])}


# 使用resnet50预训练模型
model_ft = models.resnet101(pretrained=True)
# 修改输入层的通道数为8
w = model_ft.conv1.weight.clone()
model_ft.conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model_ft.conv1.weight = torch.nn.Parameter(torch.cat((w, torch.zeros(64, 5, 7, 7)), dim=1))
model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
# for param in model_ft.parameters():
#     param.requires_grad = False
# print(model_ft)
# num_ftrs = model_ft.classifier[6].in_features
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 512)


# 构建有效牌的多任务模型
dd = .1
model_1 = multi_output_model(model_ft, dd)
model_1 = model_1.to(device)
# print(model_1)
# print(model_1.parameters())

# 设置损失函数
criterion = [nn.BCELoss(), nn.BCELoss(), nn.BCELoss()]


# 设置学习率
lrlast = .001
lrmain = .0001
optim = optim.Adam(
    [
        {"params": model_1.resnet_model.parameters()},
        {"params": model_1.x1.parameters(), "lr": lrlast},
        {"params": model_1.y1o.parameters(), "lr": lrlast},
        {"params": model_1.y2o.parameters(), "lr": lrlast},
        {"params": model_1.y3o.parameters(), "lr": lrlast},

    ],
    lr=lrmain)
# optim = optim.Adam(model_1.parameters(),lr=lrmain)#, momentum=.9)
# Observe that all parameters are being optimized
optimizer_ft = optim
# Decay LR by a factor of 0.1 every 5 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)


# 开始训练
model_ft1 = train_model(model_1, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=200)


#将loss acc保存
def array2File(name,array,type):
    output = open(os.getcwd()+'/result/V2_epoch200/' + name + '.txt', 'w')
    for i in range(len(array)):
        for j in range(len(array[i])):
            if type == 'loss':
                output.write(str(array[i][j].item()))
            elif type == 'acc':
                output.write(str(array[i][j]))
            output.write(' ')
        output.write('\n')
    output.close()

# 暂且先不用将loss acc记录到文本的方法
# array2File('train_loss',train_loss,'loss')
# array2File('train_acc',train_acc,'acc')
# array2File('val_loss',val_loss,'loss')
# array2File('val_acc',val_acc,'acc')

# 将模型保存
torch.save(model_ft1.state_dict(), os.getcwd()+'/model/V2/resnet_split_lr_1-0001.pth')
