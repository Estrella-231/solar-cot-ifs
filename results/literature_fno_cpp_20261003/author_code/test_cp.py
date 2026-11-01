import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utilities3 import *
import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io
import glob as gb

from skimage.metrics import structural_similarity as compare_ssim
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import os
import scipy.io as scio
import numpy as np
import math

torch.manual_seed(0)
np.random.seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"
################################################################
# 3d fourier layers
################################################################

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
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)
        # return torch.mm(input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        # x_ft = torch.rfft(x, 3, normalized=True, onesided=True)
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
        # x = torch.irfft(out_ft.float(), 3, normalized=True, onesided=True, signal_sizes=(x.size(-3), x.size(-2), x.size(-1)))
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
        self.fc2 = nn.Linear(128, 3)  # n_classes=3

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
        x = F.softmax(x, dim=-1)
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


################################################################
# configs
################################################################
t1 = default_timer()

ntest = 64
data = np.load('/home/LVM_date2/hongx/2017_test_plot/007.npy')
test = data[:, :, :, :, 3]  # [18,36,512,512]
test[np.isnan(test)] = 0
# test = (test+80)/110  # 归一化
test = test.transpose(1, 2, 3, 0)   # [36,512,512,18]

sub = 1
S = 512
T_in = 6
T = 12
batch_size = 1

# .npy load data

test_a = test[:ntest, ::sub, ::sub, :T_in]
test_u = test[:ntest, ::sub, ::sub, T_in:]

test_a = torch.from_numpy(test_a)
test_a = test_a.float()
test_u = torch.from_numpy(test_u)
test_u = test_u.float()
print(test_a.shape, test_u.shape)
assert (S == test_u.shape[-2])
assert (T == test_u.shape[-1])

a_normalizer = UnitGaussianNormalizer(test_a)
test_a = a_normalizer.encode(test_a)
y_normalizer = UnitGaussianNormalizer(test_u[:, :, :, :6])
y_normalizer2 = UnitGaussianNormalizer(test_u[:, :, :, 6:])

test_a = test_a.reshape(ntest, S, S, 1, T_in).repeat([1, 1, 1, 6, 1])
print('test_a:', test_a.shape)  # [ntest, S, S, T, T_in]

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
device = 'cuda'
# device = torch.cuda.set_device(2)  # 指定显卡

# load model
model = torch.load('./model_3D/3000_CrossEntropy_CP_512_ep300_m6_w24', map_location='cuda:1')
y_normalizer.cuda()
y_normalizer2.cuda()
# model = model.to('cpu')

# test
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size,
                                          shuffle=False)
# myloss = LpLoss(size_average=False)
# Mse = MSEUsingWeight().to(device, dtype=torch.float)
pred = torch.zeros((batch_size, S, S, T))
res = torch.zeros(test_u.shape)
index = 0
test_l2 = 0
t1 = default_timer()
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.cuda(device=device_ids[1]), y.cuda(device=device_ids[1])
        print('x max: ', x.max())
        print('y max: ', y.max())
        out = model(x).view(batch_size, S, S, 6, 3)
        first_out = torch.argmax(out, dim=-1)
        # print('out:', out.shape)
        # print('first_out: ', first_out.shape)
        out_input = first_out.reshape(batch_size, S, S, 1, 6).repeat([1, 1, 1, 6, 1])
        out2 = model(out_input).view(batch_size, S, S, 6, 3)
        second_out = torch.argmax(out2, dim=-1)
        predicted_classes = torch.cat((out, out2), axis=-2)
        # predicted_classes = torch.argmax(out, dim=-1)
        label_classes = torch.argmax(y, dim=-1)
        pred = predicted_classes
        index = index + 1
        if index == 1:
            res = pred
        else:
            res = torch.cat((res, pred), 0)
t2 = default_timer()
print('预测所用时间:', t2-t1)
print('pred:', res.shape, 'min:', res.min(), 'max:', res.max())
print('label:', test_u.shape, 'min:', test_u.min(), 'max:', test_u.max())

pred = res.cpu().numpy()
label = test_u.cpu().numpy()
np.savez('/home/LVM_date2/hongx/2017_test_plot/007_cp.npz', pred=pred, label=label)
