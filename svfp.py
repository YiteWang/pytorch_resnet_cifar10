# This pruning technique tries to prune weights that affects spectrum of layer weight

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import snip
import math
import torch.optim as opt
import numpy as np

NUM_REINIT = 10

def deconv_orth_dist(conv, stride = 2):
    kernel = conv.weight
    padding = int(np.floor((kernel.shape[2]-1)/conv.stride[0])*conv.stride[0])
    [o_c, i_c, w, h] = kernel.shape
    output = torch.conv2d(kernel*conv.weight_mask, kernel*conv.weight_mask, stride=stride, padding=padding)
    target = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1])).cuda()
    ct = int(np.floor(output.shape[-1]/2))
    target[:,:,ct,ct] = torch.eye(o_c).cuda()
    return torch.norm( output - target )

def svip_reinit(net):
    optimizer = opt.Adam(net.parameters(), lr=0.01)
    for i in range(NUM_REINIT):
        optimizer.zero_grad()
        loss = 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                loss += deconv_orth_dist(layer, layer.stride[0])
            elif isinstance(layer, nn.Linear):
                loss += torch.norm(torch.svd(layer.weight*layer.weight_mask)[1]-torch.ones(min(layer.weight.shape)).cuda())
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()


def get_svip_loss(net):
    loss = 0
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
            loss += deconv_orth_dist(layer, layer.stride[0])
        elif isinstance(layer, nn.Linear):
            loss += torch.norm(torch.svd(layer.weight*layer.weight_mask)[1]-torch.ones(min(layer.weight.shape)).cuda())
    return loss

def compute_layer_cond(W):
    sv = torch.svd(W)[1]
#     try:
#         condition_number = sv[0]/sv[-1] + sv[1]/sv[-2] + + sv[2]/sv[-3]
#     except:
#         try:
#             condition_number = sv[0]/sv[-1] + sv[1]/sv[-2]
#         except:
#             condition_number = sv[0]/sv[-1]
    condition_number = sv.sum()
#     condition_number = torch.log(sv+1e-5).sum()
    return condition_number


# Apply SVIP pruning methods
# TODO: support spectral norm
def apply_svip(args, nets):
    
    # first add masks to each layer of nets
    for net in nets:
        net.train()
        net.zero_grad()
        for layer in net.modules():
            snip.add_mask_ones(layer)

    model = nets[0]

    # if args.iter_prune:
    #     num_iter = round(math.log(args.sparse_lvl, 0.8))
    #     for i in range(num_iter):
    #         loss = get_svip_loss(model)
    #         loss.backward()
            
    #         # prune the network using CS
    #         for net in nets:
    #             net_prune_svip(net, 0.8**(i+1))

    if args.iter_prune:
        num_iter = 100
        for i in range(num_iter):
            loss = get_svip_loss(model)
            loss.backward()
            
            # prune the network using CS
            for net in nets:
                net_prune_svip(net, args.sparse_lvl**((i+1)/num_iter))
                # svip_reinit(net)

            if i%10 == 0:
                print('Prune ' + str(i) + ' iterations.')

    else:
        loss = get_svip_loss(model)
        loss.backward()
        
        # prune the network using CS
        for net in nets:
            net_prune_svip(net, args.sparse_lvl)

    deactivate_mask_update(net)
    print('[*] SVIP pruning done!')


def net_prune_svip(net, sparse_lvl):
    grad_mask = {}
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
            grad_mask[layer]=torch.abs(layer.weight_mask.grad)
            # grad_mask[layer]=layer.weight_mask.grad

    # find top sparse_lvl number of elements
    grad_mask_flattened = torch.cat([torch.flatten(a) for a in grad_mask.values()])
    grad_mask_sum = torch.abs(torch.sum(grad_mask_flattened))
    grad_mask_flattened /= grad_mask_sum

    left_params_num = int (len(grad_mask_flattened) * sparse_lvl)
    grad_mask_topk, _ = torch.topk(grad_mask_flattened, left_params_num)
    threshold = grad_mask_topk[-1]

    modified_mask = {}
    for layer, mask in grad_mask.items():
        modified_mask[layer] = ((mask/grad_mask_sum)>=threshold).float()
        a = modified_mask[layer]
    #     print(((a!=0).float().sum()/a.numel()))
    # print('-'*20)

    with torch.no_grad():
        for layer in net.modules():          
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
                # Stop calculating gradients of masks
                layer.weight_mask.data = modified_mask[layer]
                # layer.weight_mask.requires_grad = False
                layer.weight_mask.grad = None

                # Set those pruned weight as 0 as well, Here needs to address spectral_norm layer
                if hasattr(layer, 'weight_orig'): 
                    layer.weight_orig *= layer.weight_mask
                else:
                    layer.weight *= layer.weight_mask

def deactivate_mask_update(net):
    with torch.no_grad():
        for layer in net.modules():          
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
                layer.weight_mask.requires_grad = False