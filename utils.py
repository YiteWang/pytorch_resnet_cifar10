import torch.nn as nn
import numpy as np
import torch
import numpy.linalg as la
import math

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
    iter_std = []
    iter_avg = []
    for layer in net.modules():
        if isinstance(layer, nn.Linear):
            if hasattr(layer, 'weight_q'):
                weight = layer.weight_q
            else:
                weight = layer.weight
            sv_result = np.zeros(20,)
            s,v,d = torch.svd(weight.view(weight.size(0),-1), compute_uv=False)
            top10_sv = v[:10].detach().cpu().numpy()
            bot10_sv = v[-10:].detach().cpu().numpy()
            sv_result[:len(top10_sv)] = top10_sv
            sv_result[-len(bot10_sv):] = bot10_sv
            iter_sv.append(sv_result.copy())
            iter_avg.append(np.mean(v.detach().cpu().numpy()))
            iter_std.append(np.std(v.detach().cpu().numpy()))
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
            top10_sv = sorted_sv_pos[:10]
            bot10_sv = sorted_sv_pos[-10:]
            sv_result[:len(top10_sv)] = top10_sv
            sv_result[-len(bot10_sv):] = bot10_sv
            iter_sv.append(sv_result.copy())
            iter_avg.append(np.mean(sorted_sv))
            iter_std.append(np.std(sorted_sv))
    return np.array(iter_sv), np.array(iter_avg), np.array(iter_std)

class Hook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input_shape = np.array(input[0].shape)

    def close(self):
        print(self.input_shape)
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