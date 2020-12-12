import torch.nn as nn
import numpy as np
import torch
import numpy.linalg as la
import math
import os

# use reshape weight matrix
# def get_sv(net, size_hook):
#     # Here, iter_sv stores singular values for different layers
#     iter_sv = []
#     for layer in net.modules():
#         if isinstance(layer, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
#             if hasattr(layer, 'weight_q'):
#                 weight = layer.weight_q
#             else:
#                 weight = layer.weight
#             sv_result = np.zeros(20,)
#             s,v,d = torch.svd(weight.view(weight.size(0),-1), compute_uv=False)
#             top10_sv = v[:10].detach().cpu().numpy()
#             bot10_sv = v[-10:].detach().cpu().numpy()
#             sv_result[:len(top10_sv)] = top10_sv
#             sv_result[-len(bot10_sv):] = bot10_sv
#             iter_sv.append(sv_result.copy())
#     return np.array(iter_sv)

# Calculate accurate eigenvalues distribution
def get_sv(net, size_hook):
    # Here, iter_sv stores singular values for different layers
    iter_sv = []
    iter_std = [] # normalized standard deviation by normalizing using the largest SV.
    iter_avg = []
    iter_svmax = [] # maximum singular value
    iter_sv50 = [] # 50% singular value
    iter_sv80 = [] # 80% singular value
    iter_kclip12 = [] # singular values larger than 1e-12
    iter_sv50p = [] # 50% non-zero singular value
    iter_sv80p = [] # 80% non-zero singular value
    iter_kavg = [] # max condition number/average condition number

    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
            if isinstance(layer, nn.Linear):
                if hasattr(layer, 'weight_q'):
                    weight = layer.weight_q
                else:
                    weight = layer.weight
                sv_result = np.zeros(20,)
                s,v,d = torch.svd(weight.view(weight.size(0),-1), compute_uv=False)
                sorted_sv = v.detach().cpu().numpy()
                sorted_sv_pos = np.array([i for i in sorted_sv if i>0])
                sorted_sv_clip = np.array([i for i in sorted_sv if i>1e-12])
                top10_sv = sorted_sv[:10]
                bot10_sv = sorted_sv[-10:]
                sv_result[:len(top10_sv)] = top10_sv
                sv_result[-len(bot10_sv):] = bot10_sv
            elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                # Notice that layer.weight has shape (C_out, C_in, H, W) and we want to transform it to
                # (H, W, C_in, C_out)
                # Notice that size_hook returns the input size of the layer, which is (Batch, in_channel, H, W)
                # We only want (H, W)
                sv_result = np.zeros(20,)
                if hasattr(layer, 'weight_q'):
                    weight = layer.weight_q
                else:
                    weight = layer.weight
                sv = SVD_Conv_Tensor_NP(weight.detach().cpu().permute(2,3,1,0), size_hook[layer].input_shape[2:])
                sorted_sv = np.flip(np.sort(sv.flatten()),0)
                sorted_sv_pos = np.array([i for i in sorted_sv if i>0])
                sorted_sv_clip = np.array([i for i in sorted_sv if i>1e-12])
                top10_sv = sorted_sv[:10]
                bot10_sv = sorted_sv[-10:]
                sv_result[:len(top10_sv)] = top10_sv
                sv_result[-len(bot10_sv):] = bot10_sv
                # iter_sv.append(sv_result.copy())
                # iter_avg.append(np.mean(sorted_sv))
                # iter_std.append(np.std(sorted_sv))
            iter_sv.append(sv_result.copy())
            iter_std.append(np.std(sorted_sv/sorted_sv.max()))
            iter_avg.append(np.mean(sorted_sv))
            iter_svmax.append(sorted_sv.max())
            iter_sv50.append(sorted_sv[int(len(sorted_sv)*0.5)])
            iter_sv80.append(sorted_sv[int(len(sorted_sv)*0.8)])
            iter_kclip12.append(sorted_sv_clip[0]/sorted_sv_clip[-1])
            iter_sv50p.append(sorted_sv_pos[int(len(sorted_sv_pos)*0.5)])
            iter_sv80p.append(sorted_sv_pos[int(len(sorted_sv_pos)*0.8)])
            iter_kavg.append(sorted_sv.max()/sorted_sv.mean())
    return (np.array(iter_sv), np.array(iter_avg), np.array(iter_std), np.array(iter_svmax), 
        np.array(iter_sv50), np.array(iter_sv80), np.array(iter_kclip12),
        np.array(iter_sv50p), np.array(iter_sv80p), np.array(iter_kavg)) 

def Project_weight(net, size_hook, clip_to=0.01):
    with torch.no_grad():
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                layer.weight.data = Clip_OperatorNorm_NP(layer.weight.detach().cpu(), size_hook[layer].input_shape[2:], clip_to)

def Clip_OperatorNorm_NP(filter, inp_shape, clip_to=0.01):
    # compute the singular values using FFT
    # first compute the transforms for each pair of input and output channels
    filter = filter.permute(2,3,1,0)
    transform_coeff = np.fft.fft2(filter, inp_shape, axes=[0, 1])

    # now, for each transform coefficient, compute the singular values of the
    # matrix obtained by selecting that coefficient for
    # input-channel/output-channel pairs
    U, D, V = np.linalg.svd(transform_coeff, compute_uv=True, full_matrices=False)
    D_clipped = np.maximum(D, clip_to)
    if filter.shape[2] > filter.shape[3]:
        clipped_transform_coeff = np.matmul(U, D_clipped[..., None] * V)
    else:
        clipped_transform_coeff = np.matmul(U * D_clipped[..., None, :], V)
    clipped_filter = np.fft.ifft2(clipped_transform_coeff, axes=[0, 1]).real
    args = [range(d) for d in filter.shape]
    return torch.Tensor(clipped_filter[np.ix_(*args)]).permute(3,2,0,1).cuda()

class Hook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input_shape = np.array(input[0].shape)

    def close(self):
        # print(self.input_shape)
        self.hook.remove()


def get_hook(net, layer_types):
    hook_forward = {layer:Hook(layer) for layer in net.modules() if isinstance(layer,layer_types)}
    return hook_forward

def detach_hook(handle_lists):
    for handle_list in handle_lists:
        for hanle in handle_list.values():
            hanle.close()

# Used code from https://github.com/brain-research/conv-sv/blob/master/conv2d_singular_values.py
def SVD_Conv_Tensor_NP(filter, inp_size):
  # compute the singular values using FFT
  # first compute the transforms for each pair of input and output channels
  transform_coeff = np.fft.fft2(filter, inp_size, axes=[0, 1])

  # now, for each transform coefficient, compute the singular values of the
  # matrix obtained by selecting that coefficient for
  # input-channel/output-channel pairs
  return la.svd(transform_coeff, compute_uv=False)

def run_once(loader, model):
    with torch.no_grad():
        data_iter = iter(loader)
        output = model(data_iter.next()[0].cuda())

def kaiming_initialize(network):
    for m in network.modules():
        if isinstance(m, nn.Conv2d):
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    print('Kaiming initialization done!')

def save_sparsity(network, save_path):
    sparsity = []
    for m in network.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            sparsity.append((m.weight_mask.sum()/m.weight_mask.numel()).item())
    np.save(os.path.join(save_path, 'saved_sparsity.npy'), sparsity)

def check_layer_sparsity(network):
    sparsity = []
    for m in network.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            sparsity.append((m.weight_mask.sum()/m.weight_mask.numel()).item())
    return np.array(sparsity)


def check_sparsity(network):
    param_remain = 0
    param_total = 0
    for m in network.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            param_remain += m.weight_mask.sum()
            param_total += m.weight_mask.numel()
    return param_remain/param_total

def get_apply_layer(network):
    num_layers = 0
    for m in network.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            num_layers += 1
    return num_layers