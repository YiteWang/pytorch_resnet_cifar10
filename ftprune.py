import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from snip import *

# Apply feature pruning methods
def apply_fpt(args, nets, data_loader, num_classes, samples_per_class = 10):
    # apply mask
    for net in nets:
        net.train()
        net.zero_grad()
        for layer in net.modules():
            add_mask_ones(layer)
    
    model = nets[0]

    handles = []

    # add handles to calculate gradients/importance
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            handles.append(module.register_forward_hook(ftp_forward()))

    data_iter = iter(data_loader)

    # using same data for 
    for i in range(samples_per_class):
        (input, target) = GraSP_fetch_data(data_iter, num_classes, 1)
        # (input, target) = data_iter.next()
        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()
        # compute output
        output = model(input_var)

    # remove hooks
    for handle in handles:
        handle.remove()
    
    net_prune_fpt(model, args.sparse_lvl)

    # zero out gradients of weights
    for net in nets:
        net.zero_grad()


def ftp_forward():
    def hook(model, input, output):
        (torch.norm(output.view(output.shape[0], -1), dim=1)/output.numel()*output.shape[0]).sum().backward(retain_graph=True)
    return hook

def net_prune_fpt(net, sparse_lvl):
    grad_mask = {}
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
            grad_mask[layer]=torch.abs(layer.weight.grad)

    # find top sparse_lvl number of elements
    grad_mask_flattened = torch.cat([torch.flatten(a) for a in grad_mask.values()])
    grad_mask_sum = torch.sum(grad_mask_flattened)
    grad_mask_flattened /= grad_mask_sum

    left_params_num = int (len(grad_mask_flattened) * sparse_lvl)
    grad_mask_topk, _ = torch.topk(grad_mask_flattened, left_params_num)
    threshold = grad_mask_topk[-1]

    modified_mask = {}
    for layer, mask in grad_mask.items():
        modified_mask[layer] = ((mask/grad_mask_sum)>=threshold).float()
        a = modified_mask[layer]
        print(((a!=0).float().sum()/a.numel()))
    print('-'*20)

    with torch.no_grad():
        for layer in net.modules():          
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
                # Stop calculating gradients of masks
                layer.weight_mask.data = modified_mask[layer]
                layer.weight_mask.requires_grad = False

                # Set those pruned weight as 0 as well, Here needs to address spectral_norm layer
                if hasattr(layer, 'weight_orig'): 
                    layer.weight_orig *= layer.weight_mask
                else:
                    layer.weight *= layer.weight_mask
