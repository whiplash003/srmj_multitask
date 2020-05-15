import numpy as np
import os
# 拼接分布生成的数据集
def stackNpy(npy_path, num):
    data = ['train' + str(i) for i in range(num)]
    label = ['effect_tile' + str(i) for i in range(num)]
    tmp_t = []
    tmp_l = []
    for i in range(num-10000, num):
        if len(tmp_t) is 0:
            tmp_t = np.load(npy_path+'/data/' + data[i] + '.npy')
            tmp_l = np.load(npy_path+'/labels/' + label[i] + '.npy')
        else:
            print(i)
            tmp_t = np.vstack((tmp_t, np.load(npy_path+'/data/' + data[i] + '.npy')))
            tmp_l = np.vstack((tmp_l, np.load(npy_path+'/labels/' + label[i] + '.npy')))

    print(data)
    print(label)
    np.save(npy_path + '/data_sum.npy', tmp_t)
    np.save(npy_path + '/label_sum.npy', tmp_l)

data_path = os.getcwd()+'/data'
stackNpy(data_path, 10000)