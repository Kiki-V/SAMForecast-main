import torch
import torch.nn as nn
import numpy as np
import math

import torch.fft as fft
from einops import rearrange, reduce, repeat


class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts

        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # print(gates)
        # [[0.0000, 0.6078, 0.3922, 0.0000],
        # [0.0000, 0.5594, 0.4406, 0.0000],
        # [0.0000, 0.0000, 0.2874, 0.7126],
        # [0.4415, 0.5585, 0.0000, 0.0000]]
        # print(torch.nonzero(gates)) #gates中非零元素的位置
        # [[0, 1],
        # [0, 2],
        # [1, 1],
        # [1, 2],
        # [2, 2],
        # [2, 3],
        # [3, 0],
        # [3, 1]]
        # print(sorted_experts) 
        # [[0, 0],
        # [0, 1],
        # [1, 1],
        # [1, 1],
        # [2, 2],
        # [2, 2],
        # [3, 2],
        # [3, 3]]
        # print(index_sorted_experts)
        # [[1, 6],
        # [0, 0],
        # [2, 2],
        # [3, 7],
        # [4, 4],
        # [5, 3],
        # [7, 1],
        # [6, 5]]
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # print(index_sorted_experts[:, 1]) [6, 0, 2, 7, 4, 3, 1, 5]
        # print(self._batch_index) [3, 0, 1, 3, 2, 1, 0, 2]
        self._part_sizes = (gates > 0).sum(0).tolist()
        # print(gates>0)
        # [[False,  True,  True, False],
        # [False,  True,  True, False],
        # [False, False,  True,  True],
        # [ True,  True, False, False]]
        # print((gates > 0).sum(0)) [1, 3, 3, 1]
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes
        
        # print(inp.shape) torch.Size([4, 96, 862, 16])
        # _batch_index:这是一个张量，存储了 gates 张量中每个非零门的批次索引。
        # inp_exp: 这行代码根据 _batch_index 从输入张量 inp 中提取相关样本。它本质上挑选了与 gates 张量中的非零门对应的样本。
        inp_exp = inp[self._batch_index].squeeze(1)
        # squeeze(1): 此操作会从张量中删除任何大小为 1 的维度。由于我们在索引后处理单个样本，因此这将删除索引后可能引入的额外维度。
        # print(inp[self._batch_index].shape) torch.Size([8, 96, 862, 16])
        # print(inp_exp.shape) torch.Size([8, 96, 862, 16])

        # _part_sizes: 这是一个列表，存储了分配给每个专家的样本数量。它也来自初始化。
        # print( self._part_sizes) [1, 3, 3, 1]
        return torch.split(inp_exp, self._part_sizes, dim=0)
        # print(len(expert_inputs)) 4
        # 1+3+3+1 = 8(dim=0)
        # print(expert_inputs[0].shape) torch.Size([1, 96, 862, 16])
        # print(expert_inputs[1].shape) torch.Size([3, 96, 862, 16])
        # print(expert_inputs[2].shape) torch.Size([3, 96, 862, 16])
        # print(expert_inputs[3].shape) torch.Size([1, 96, 862, 16])

    def combine(self, expert_out, multiply_by_gates=True):
        # combine 方法负责将来自不同专家输出的结果进行合并
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()
        # print(stitched.shape)
        # print( self._nonzero_gates.shape)
        # torch.Size([32, 96, 862])
        # torch.Size([32, 1])
        if multiply_by_gates:
            # 将连接后的输出 (stitched) 与非零门的张量 (self._nonzero_gates) 相乘。这相当于仅对通过非零门的专家输出进行加权。
            stitched = torch.einsum("ijk,ik -> ijk", stitched, self._nonzero_gates)

        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2), 
                            requires_grad=True, device=stitched.device)
        # print(zeros.shape) torch.Size([4, 96, 862, 16])
        # print(stitched.shape)torch.Size([8, 96, 862, 16])
        # combine samples that have been processed by the same k experts
        # index_add把stitched加到zeros 
        # self._batch_index的要求：长度为stitched的长度，里面的数字为0到len(zeros)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # print(combined.shape) torch.Size([4, 96, 862, 16])
        
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()
    def expert_to_gates(self):
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc = nn.Conv2d(in_channels=input_size,
                             out_channels=output_size,
                             kernel_size=(1, 1),
                             bias=True)

    def forward(self, x):
        out = self.fc(x)
        return out



class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    kernel_size=[4, 8, 12]
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    kernel_size=[4, 8, 12]
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        # print(x.shape) torch.Size([16, 96, 862])
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            # print(moving_avg.shape) torch.Size([16, 96, 862])
            temp = moving_avg.unsqueeze(-1)
            # print(temp.shape)  torch.Size([16, 96, 862, 1])
            moving_mean.append(temp)
        moving_mean = torch.cat(moving_mean, dim=-1)
        # print(moving_mean.shape) torch.Size([16, 96, 862])
        moving_mean = torch.sum(moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
        res = x - moving_mean
        return res, moving_mean


class FourierLayer(nn.Module):
    # pred_len=0, k=3
    def __init__(self, pred_len, k=None, low_freq=1, output_attention=False):
        super().__init__()
        # self.d_model = d_model
        self.pred_len = pred_len
        self.k = k
        self.low_freq = low_freq
        self.output_attention = output_attention

    def forward(self, x):
        """x: (b, t, d)"""
        # print(x.shape) torch.Size([16, 96, 862])
        if self.output_attention:
            return self.dft_forward(x)

        b, t, d = x.shape
        x_freq = fft.rfft(x, dim=1)
        # print(x_freq.shape) torch.Size([16, 49, 862])

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            # print(x_freq.shape) torch.Size([16, 47, 862])
            # 得到从最低频率到最高频率（不包括直流分量和最高频率的负值）的所有正频率分量
            f = fft.rfftfreq(t)[self.low_freq:-1]
            # print(f.shape) torch.Size([47])
            # fft.rfftfreq(t) 生成一个长度为 t 的频率向量，表示 FFT 后每个频率分量对应的频率值。
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2))
        f = f.to(x_freq.device)
        # 选择与 x_freq 中选定的 k 个最大频率分量相对应的频率值
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)

        return self.extrapolate(x_freq, f, t), None

    # extrapolate 方法：用于生成预测的时间序列数据。  
    #     将频率域的数据与其共轭合并，并将频率值也相应地合并。
    #     计算时间步长 t_val。
    #     计算幅度 amp 和相位 phase。
    #     使用幅度、相位和时间步长计算预测的时间序列数据。
    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        # print(t) 96
        # print(self.pred_len) 0
        t_val = rearrange(torch.arange(t + self.pred_len, dtype=torch.float),
                          't -> () () t ()').to(x_freq.device)
        # print(t_val.shape) torch.Size([1, 1, 96, 1])
        amp = rearrange(x_freq.abs() / t, 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        # 傅里叶逆变化
        x_time = amp * torch.cos(2 * math.pi * f * t_val + phase)
        # print(x_time.shape) torch.Size([16, 6, 96, 862])
        temp = reduce(x_time, 'b f t d -> b t d', 'sum')
        # print(temp.shape) torch.Size([16, 96, 862])
        return temp

    # 选择 k 个最大的频率分量
    def topk_freq(self, x_freq):
        # 包含 k 个最大绝对值的张量 values 和对应的索引张量 indices
        values, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        # print(indices.shape) torch.Size([16, 3, 862])
        # 一个网格 mesh_a，它包含了所有批次的索引，和一个网格 mesh_b，它包含了所有特征的索引
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)))
        # print(mesh_a.shape) torch.Size([16, 862])
        # print(mesh_b.shape) torch.Size([16, 862])
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]

        return x_freq, index_tuple

    def dft_forward(self, x):
        T = x.size(1)

        dft_mat = fft.fft(torch.eye(T))
        i, j = torch.meshgrid(torch.arange(self.pred_len + T), torch.arange(T))
        omega = np.exp(2 * math.pi * 1j / T)
        idft_mat = (np.power(omega, i * j) / T).cfloat()

        x_freq = torch.einsum('ft,btd->bfd', [dft_mat, x.cfloat()])

        if T % 2 == 0:
            x_freq = x_freq[:, self.low_freq:T // 2]
        else:
            x_freq = x_freq[:, self.low_freq:T // 2 + 1]

        _, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        indices = indices + self.low_freq
        indices = torch.cat([indices, -indices], dim=1)

        dft_mat = repeat(dft_mat, 'f t -> b f t d', b=x.shape[0], d=x.shape[-1])
        idft_mat = repeat(idft_mat, 't f -> b t f d', b=x.shape[0], d=x.shape[-1])

        mesh_a, mesh_b = torch.meshgrid(torch.arange(x.size(0)), torch.arange(x.size(2)))

        dft_mask = torch.zeros_like(dft_mat)
        dft_mask[mesh_a, indices, :, mesh_b] = 1
        dft_mat = dft_mat * dft_mask

        idft_mask = torch.zeros_like(idft_mat)
        idft_mask[mesh_a, :, indices, mesh_b] = 1
        idft_mat = idft_mat * idft_mask

        attn = torch.einsum('bofd,bftd->botd', [idft_mat, dft_mat]).real
        return torch.einsum('botd,btd->bod', [attn, x]), rearrange(attn, 'b o t d -> b d o t')