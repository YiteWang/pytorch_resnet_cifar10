import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn

# This function generates uniform orthogonal matrix (actually orthonormal by torch.qr)
# reference: https://stackoverflow.com/questions/38426349/how-to-create-random-orthonormal-matrix-in-python-numpy
# reference: https://github.com/thwgithub/CNN/blob/master/Orthogonal%20kernel%20initialization%20for%202DConv.py
def get_orthogonal_mat(dim):
    rand_mat = torch.zeros((dim, dim)).normal_(0,1)
    q, r = torch.qr(rand_mat)
    d = torch.diag(r, 0).sign()
    d_size = d.size(0)
    d_exp = d.view(1, d_size).expand(d_size, d_size)
    q.mul_(d_exp)
    return q


# This initialization method is based on Algorithm 2 of paper 
# of original paper: https://arxiv.org/abs/1806.05393
# reference: https://github.com/tensorflow/tensorflow/blob/v1.9.0-rc0/tensorflow/python/ops/init_ops.py
def DeltaOrthogonal_init(weights, gain=1):
    out_channels = weights.size(0)
    in_channels = weights.size(1)
    if in_channels > out_channels:
        raise ValueError("In_channels should not be larger than Out_Channels")
    weights.data.fill_(0)
    q = get_orthogonal_mat(out_channels) # size out_channels x out_channels
    q = q[:, :in_channels]
    q *= np.sqrt(gain) 
    beta = weights.size(2) // 2
    beta_prime = weights.size(3) // 2
    with torch.no_grad():
        weights[:, :, beta, beta_prime] = q
