import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import snip
import numpy as np

# Apply feature pruning methods
def apply_synflow(args, nets, data_loader):

    @torch.no_grad()
    def linearize(model):
        signs = {}
        for name, param in model.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(model, signs):
        for name, param in model.state_dict().items():
            param.mul_(signs[name])

    print('[*] Synflow pruner starts.')
    for net in nets:
        # To use Synflow, we have to set Batchnorm to eval mode
        net.eval()
        net.zero_grad()
        for layer in net.modules():
            snip.add_mask_ones(layer)

    # Reset batchnorm statistics
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            layer.weight_mask.grad = None
        if isinstance(layer, nn.BatchNorm2d):
            layer.running_mean.fill_(0)
            layer.running_var.fill_(1)
    
    model = nets[0]
    data_iter = iter(data_loader)
    (data, _) = next(data_iter)
    input_dim = list(data[0,:].shape)
    input = torch.ones([1] + input_dim).to(device)#, dtype=torch.float64).to(device)
    
    if args.iter_prune:
        num_iter = 100
    else:
        num_iter = 1

    signs = linearize(model)

    for i in range(num_iter):
        # Zero out gradients for weight_mask so to start a new round of iterative pruning
        model.zero_grad()

        output = model(input)
        torch.sum(output).backward()

        # snip.prune_net_increaseloss(model, args.sparse_lvl**((i+1)/num_iter))
        # snip.net_iterative_prune(model, args.sparse_lvl**((i+1)/num_iter))
        snip.net_prune_grasp(model, args.sparse_lvl**((i+1)/num_iter))
        if i % 5 ==0:
            print('Prune ' + str(i) + ' iterations.')

    snip.deactivate_mask_update(model)
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            weight_check = module.weight_mask
            print(((weight_check!=0).float().sum()/weight_check.numel()))
    
    print('-'*20)

    # # remove hooks
    # for handle in handles:
    #     handle.remove()

    # zero out gradients of weights
    for net in nets:
        net.zero_grad()
        net.train()
        # net.reinit()
    nonlinearize(model)
