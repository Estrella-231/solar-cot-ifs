import torch
import torch.fft
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utilities3 import *
import glob as gb

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from Adam import Adam
from torch.utils.tensorboard import SummaryWriter
from makedata_fordataloader import MyDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


torch.manual_seed(0)
np.random.seed(0)

# Tensorboard 可视化
log_writer = SummaryWriter()

################################################################
# 3d fourier layers
################################################################

# Fourier Layer
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        # print(input.shape)  # [1, width, modes, modes, modes]
        # print(weights.shape)  # [width, width, modes, modes, modes]
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


# 构建了4层Fourier Layers
class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(9, self.width)
        # input channel is 13: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        x = self.fc1(x)  # 128
        # x = F.gelu(x)
        x = self.fc2(x)  # 128——>3
        # x = F.softmax(x, dim=-1)
        # print(x)
        # x = torch.argmax(x, dim=-1)
        x = x.to(torch.float32)
        # print('shape: ', x.shape)
        # print(' type: ', x.dtype)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
# ----------------------------------------------------------------------------------------------- #

# crossEntropy loss
def custom_cross_entropy(logits, targets):
    """
    自定义交叉熵损失函数，用于处理浮点型的one-hot编码标签。
    logits: 模型输出的原始logits（未经softmax处理）。
    targets: one-hot编码的浮点型标签。
    """
    # 计算softmax
    softmax_logits = F.softmax(logits, dim=1)
    # 计算交叉熵损失
    loss = -targets * torch.log(softmax_logits + 1e-6)  # 加上小常数防止数值不稳定
    # 计算平均损失
    return torch.mean(torch.sum(loss, dim=1))
# ----------------------------------------------------------------------------------------------- #


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# --------parameters set--------------- #
ntrain = 3000
ntest = 1200
print('train nums:', ntrain, 'valid nums:', ntest)

modes = 6
width = 24

batch_size = 4

epochs = 300
learning_rate = 0.00001
scheduler_step = 100
scheduler_gamma = 0.1
min_loss = 100

print('epoch:', epochs, 'lr:', learning_rate, 'scheduler_step:', scheduler_step, 'scheduler_gamma:',
      scheduler_gamma)
print('batch_size:', batch_size, 'modes:', modes, 'width:', width)

path = str(ntrain) + '_6h_LR0.00001_CP_512' + '_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
path_model = './model_3D/' + path

runtime = np.zeros(2, )
t1 = default_timer()

sub = 1  # sub是缩小的倍数，1则输入输出不变
S = 512 // sub
T_in = 6
T = 6
################################################################
# load data
################################################################
print('Begin read data')
path = gb.glob('/home/SSD4090/hongx/CP/*.npy')
train_path = []
valid_path = []
for i in range(ntrain):
    train_path.append(sorted(path)[i])
for j in range(ntest):
    valid_path.append(sorted(path)[ntrain+j])

train = []
test = []
for i in range(len(train_path)):
    train_data = np.load(train_path[i])
    train.append(train_data)
for j in range(len(valid_path)):
    test_data = np.load(valid_path[j])
    test.append(test_data)
train = np.array(train)
test = np.array(test)
train[np.isnan(train)] = 0
test[np.isnan(test)] = 0
print('train: ', train.min(), train.max())
print('valid: ', test.min(), test.max())

train_a = train[:ntrain, ::sub, ::sub, :T_in]  # train input
train_u = np.zeros((ntrain, S, S, T))  # train output label
train_u = train[:ntrain, ::sub, ::sub, T_in:T+T_in]

test_a = test[:ntest, ::sub, ::sub, :T_in]  # test input
test_u = np.zeros((ntest, S, S, T))  # test output label
test_u = test[:ntest, ::sub, ::sub, T_in:T+T_in]

train_a = torch.from_numpy(train_a)
train_a = train_a.float()
test_a = torch.from_numpy(test_a)
test_a = test_a.float()
train_u = torch.from_numpy(train_u).long()
train_u = torch.nn.functional.one_hot(train_u, num_classes=3)
train_u = train_u.float()
test_u = torch.from_numpy(test_u).long()
test_u = torch.nn.functional.one_hot(test_u, num_classes=3)
test_u = test_u.float()

print(train_a.shape, test_a.shape)
print(train_u.shape, test_u.shape)
assert (S == train_a.shape[-2])
assert (T == train_a.shape[-1])

# a_normalizer = UnitGaussianNormalizer(train_a)
# train_a = a_normalizer.encode(train_a)
# a_normalizer2 = UnitGaussianNormalizer(test_a)
# test_a = a_normalizer2.encode(test_a)

# y_normalizer = UnitGaussianNormalizer(train_u)
# train_u = y_normalizer.encode(train_u)  # encode就是进行归一化,decode就是还原
# y_normalizer2 = UnitGaussianNormalizer(test_u)
# test_u = y_normalizer2.encode(test_u)

train_a = train_a.reshape(ntrain, S, S, 1, T_in).repeat([1, 1, 1, T, 1])
test_a = test_a.reshape(ntest, S, S, 1, T_in).repeat([1, 1, 1, T, 1])
print(train_a.shape, test_a.shape)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size,
                                          shuffle=False)

t2 = default_timer()

print('preprocessing finished, time used:', t2 - t1)
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
# device = 'cuda'
# device = torch.cuda.set_device(1)  # 指定显卡

################################################################
# training and evaluation
################################################################
# model = FNO3d(modes, modes, modes, width)
# model = torch.nn.DataParallel(model, device_ids=device_ids)
model = torch.load('./model_3D/2000_CrossEntropy_CP_512_ep300_m6_w24')
model = model.cuda(device=device_ids[6])

print('model params: ', count_params(model))
# StepLR学习率策略
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
# 余弦退火学习率策略
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0)

myloss = nn.CrossEntropyLoss().to(device=device, dtype=torch.float32)
# Mse = MSEUsingWeight().to(device, dtype=torch.float)
# y_normalizer.cuda()

train_loss = []
test_loss = []
mse_loss = []
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    test_l2 = 0
    for x, y in train_loader:
        x = x.to(device=device_ids[6], dtype=torch.float32)
        y = y.to(device=device_ids[6], dtype=torch.float32)
        # print(x.shape, y.shape)  # x:[batch_size,512,512,6,6];y:[batch_size,512,512,6,3]
        # print(x.max(), x.min())  # [0、1、2]
        # print(y.max(), y.min())  # [0、1]
        optimizer.zero_grad()
        out = model(x).view(batch_size, S, S, T, 3)  # torch中的view相当于reshape
        # out = torch.autograd.Variable(out, requires_grad=True)
        # y = torch.autograd.Variable(y, requires_grad=True)
        # mse = F.mse_loss(out, y, reduction='mean')
        # mse.backward()
        # predicted_classes = torch.argmax(out, dim=-1)
        # out_one_hot = torch.eye(out.shape[-1], device=device_ids[0])[predicted_classes]
        l2 = custom_cross_entropy(out.view(-1, 3), y.view(-1, 3))  # reshape成[batch_size, S*S*T]
        l2.backward()
        optimizer.step()
        # train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in test_loader:
            x = x.to(device=device_ids[6], dtype=torch.float32)
            y = y.to(device=device_ids[6], dtype=torch.float32)
            out = model(x).view(batch_size, S, S, T, 3)
            # print('out shape: ', out.shape)
            # print('out min: ', out.min())
            # print('out max: ', out.max())
            # y = y_normalizer2.decode(y)
            # out = y_normalizer2.decode(out)
            # out[out < 0] = 0
            # out[out > 18] = 18
            predicted_classes = torch.argmax(out, dim=-1)  # [batch_size,6,512,512] values 0/1/2
            label_classes = torch.argmax(y, dim=-1)
            # out_one_hot = torch.eye(out.shape[-1], device=device_ids[0])[predicted_classes]
            # print('out_one_hot shape: ',  out_one_hot.shape)
            # print('out_one_hot max:', out_one_hot.max())
            l2 = custom_cross_entropy(out.view(-1, 3), y.view(-1, 3))
            test_l2 += l2.item()
            correct += (predicted_classes == label_classes).sum().item()
            # print('correct: ', correct)
            total += y.numel()  # 获取y tensor的元素个数
            # print('total: ', total)

    # train_mse /= len(train_loader)
    train_l2 = train_l2/ntrain*batch_size
    log_writer.add_scalar('Loss/train', float(train_l2), ep + 1)
    test_l2 = test_l2/ntest*batch_size
    # test_l2 = test_l2/ntrain*batch_size
    log_writer.add_scalar('Loss/validation', float(test_l2), ep + 1)
    log_writer.add_scalar('Accuracy/validation', float(correct / total), ep + 1)

    t2 = default_timer()
    train_loss.append(train_l2)
    test_loss.append(test_l2)
    # mse_loss.append(train_mse)
    print(f"Epoch : {ep + 1} - loss : {train_l2:.4f} - val_loss : {test_l2:.4f}\n")
    acc = 100 * correct / total
    print(f"Accuracy : {acc:.4f}")
    print('运行时长为:', t2 - t1)
    if test_l2 <= min_loss:
        min_loss = test_l2
        print('begin save model | epoch is %d' % (ep+1))
        torch.save(model, path_model)
        print("*** save state to file %s" % path_model)
        print('end save model')


train_loss = np.array(train_loss)
test_loss = np.array(test_loss)
# mse_loss = np.array(mse_loss)
np.save('./CP_6h_LP_512_train_loss.npy', train_loss)
np.save('./CP_6h_LP_512_test_loss.npy', test_loss)
# np.save('/home/hongx/out_predction/FNO/model_3D/mse_loss_0407.npy', mse_loss)
# scipy.io.savemat('pred/' + path + '.mat', mdict={'pred': pred.cpu().numpy(),
#                                                  'label': label.cpu().numpy()})
