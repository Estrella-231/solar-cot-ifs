import os
import random

import numpy as np
import torch.utils.data as td
from torch.utils.data import DataLoader
import pandas as pd
import time
import glob as gb
import operator
import torch

class MyDataset(td.Dataset):

    def __init__(self, data_dir: str, _train):
        self.data_dir = data_dir
        self.trainState = _train

        if self.trainState:
            self.sample_count = 20
            print('process is train, nums is ', self.sample_count)
        else:
            self.sample_count = 2
            print('process is valid, nums is ', self.sample_count)

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index):
        path = gb.glob('/run/media/qz/Data9/home/HX/2016/*.npy')
        ntrain = 20
        ntest = 2
        train_path = []
        valid_path = []
        for i in range(ntrain):
            train_path.append(sorted(path)[i])
        for j in range(ntest):
            valid_path.append(sorted(path)[ntrain+j+100])
        # print('TrainSet nums: ', len(train_path))
        # print('TestSet nums: ', len(valid_path))
        if self.trainState:
            path_data = train_path[index]  # index从0开始
        else:
            path_data = valid_path[index]
        data_xy = np.load(path_data)
        xx = data_xy[0:6, 200:712, 200:712, 3]  # input [6,1024,1024]
        xx = xx.transpose(1, 2, 0)  # (H,W,T) [1024,1024,6]
        yy = data_xy[6:12, 200:712, 200:712, 3]  # output
        yy = yy.transpose(1, 2, 0)  # (H,W,T)
        yy = torch.from_numpy(yy).long()
        yy = torch.nn.functional.one_hot(yy, num_classes=3)
        # xx = xx/np.nanmax(xx)
        # yy = yy/np.nanmax(yy)
        # xx[np.isnan(xx)] = -999
        # yy[np.isnan(yy)] = -999

        return xx, yy


if __name__ == "__main__":
    print('Begin read data')
    path = gb.glob('/public/MARS/datasets/cloud_properties/dataset_1024/*.npy')
    train_path = []
    valid_path = []
    ntrain = 100
    ntest = 10
    for i in range(ntrain):
        train_path.append(path[i])
    for j in range(ntest):
        valid_path.append(path[ntrain + j])
    train_data = MyDataset(train_path, _train=True)  # 构建Dataset对象
    print('Process MyDataset_train is done')
    print('Train nums: ', train_data.__len__())
    valid_data = MyDataset(valid_path, _train=False)
    print('Process MyDataset_valid is done')
    print('Valid nums: ', valid_data.__len__())
    batch_size = 2
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)  # 通过DataLoader来构建迭代对象
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)

    x1 = train_data.__getitem__(0)
    y1 = valid_data.__getitem__(0)  # 支持数据集索引的函数
    print(np.shape(x1))  # 检查是否可以正常取数;Tuple的shape需要用np.shape
    print(np.shape(y1))
