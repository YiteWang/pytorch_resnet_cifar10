import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import svfp
from snip import *
import snip
import numpy as np

# Apply feature pruning methods
def apply_fpt(args, nets, data_loader, num_classes, samples_per_class = 10):
    # apply mask
    for net in nets:
        net.eval()
        net.zero_grad()
        for layer in net.modules():
            add_mask_ones(layer)
    
    model = nets[0]

    # handles = []

    # # add handles to calculate gradients/importance
    # for module in model.modules():
    #     if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
    #         handles.append(module.register_forward_hook(ftp_forward()))
    
    if args.iter_prune:
        num_iter = 100
    else:
        num_iter = 1

    for i in range(num_iter):
        # using same data for 
        data_iter = iter(data_loader)
        for sample in range(samples_per_class):
            (input, target) = GraSP_fetch_data(data_iter, num_classes, 1)
            # (input, target) = data_iter.next()
            target = target.cuda()
            input_var = input.cuda()
            target_var = target
            if args.half:
                input_var = input_var.half()
            # compute output
            output = model(input_var)

            # this is only for the last layer
            (torch.norm(output.view(output.shape[0], -1), dim=1)/output.numel()*output.shape[0]).sum().backward()

        snip.net_iterative_prune(model, args.sparse_lvl**((i+1)/num_iter))
        if i % 10 ==0:
            print('Prune ' + str(i) + ' iterations.')

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            weight_check = module.weight
            print(((weight_check!=0).float().sum()/weight_check.numel()))
    
    print('-'*20)

    # # remove hooks
    # for handle in handles:
    #     handle.remove()

    # zero out gradients of weights
    for net in nets:
        net.zero_grad()
        net.train()


def ftp_forward():
    def hook(model, input, output):
        (torch.norm(output.view(output.shape[0], -1), dim=1)/output.numel()*output.shape[0]).sum().backward(retain_graph=True)
    return hook

def apply_specprune(nets, sparsity):
    applied_layer = 0
    for net in nets:
        for layer in net.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                if sparsity[applied_layer] != 1:
                    # add_mask_rand_basedonchannel(layer, sparsity[applied_layer], ratio, True, structured)
                    specprune(layer, sparsity[applied_layer])
                    print('[*] Layer '+str(applied_layer)+' pruned!')
                else:
                    snip.add_mask_ones(layer)
                applied_layer += 1
        deactivate_mask_update(net)

def specprune(layer, sparsity):
    remain = int(sparsity * layer.weight.numel())
    layer.mask_param = nn.Parameter(torch.randn(layer.weight.shape, device=layer.weight.device, requires_grad=True))
    optimizer = torch.optim.Adam([layer.mask_param], lr=0.1)
    # find Lagrangian:
    inner = 100
    outer = 30
    alpha = 1
    loss = []
    ck = 1e-14
    for epoch in range(outer):
        for in_epoch in range(inner):
            ortho_loss = deconv_orth_dist_mod(layer, layer.stride)
            L = ortho_loss  + ck*(layer.weight_mask.sum()-remain)**2  # + alpha * (layer.weight_mask.sum()-remain) 
            L.backward()
            optimizer.step()
            optimizer.zero_grad()
        ck *= 3
    with torch.no_grad():
        layer.weight_mask.fill_(0)
        criterion = layer.mask_param.view(-1).topk(remain)[0][-1]
        layer.weight_mask = (layer.mask_param >= criterion).float().detach()
        layer.weight *= layer.weight_mask
    layer.mask_param = None
    snip.modify_mask_forward(layer)

def deconv_orth_dist_mod(layer, stride = 2):
    kernel = layer.weight
    padding = int(np.floor((kernel.shape[2]-1)/layer.stride[0])*layer.stride[0])
    [o_c, i_c, w, h] = kernel.shape
    layer.weight_mask = torch.sigmoid(layer.mask_param)
    output = torch.conv2d(kernel.detach()*layer.weight_mask, kernel.detach()*layer.weight_mask, stride=stride, padding=padding)
    target = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1])).cuda()
    ct = int(np.floor(output.shape[-1]/2))
    target[:,:,ct,ct] = torch.eye(o_c).cuda()
    return torch.norm( output - target )
