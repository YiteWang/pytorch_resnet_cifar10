import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import snip
import utils

# Code obtained by TENAS: https://github.com/VITA-Group/TENAS/blob/main/prune_tenas.py
def get_ntk_n(xloader, networks, recalbn=0, train_mode=False, num_batch=-1, num_classes=10, samples_per_class=1):
    device = torch.cuda.current_device()
    # if recalbn > 0:
    #     network = recal_bn(network, xloader, recalbn, device)
    #     if network_2 is not None:
    #         network_2 = recal_bn(network_2, xloader, recalbn, device)
    ntks = []
    for network in networks:
        if train_mode:
            network.train()
        else:
            network.eval()
    ######
    data_iter = iter(xloader)
    grads = [[] for _ in range(len(networks))]
    for i in range(num_batch):
        inputs = snip.GraSP_fetch_data(data_iter, num_classes, samples_per_class)[0].cuda(device=device, non_blocking=True)
        for net_idx, network in enumerate(networks):
            network.zero_grad()
            inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
            logit = network(inputs_)
            if isinstance(logit, tuple):
                logit = logit[1]  # 201 networks: return features and logits
            for _idx in range(len(inputs_)):
                logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True, create_graph = True)
                grad = []
                for name, W in network.named_parameters():
                    if 'weight' in name and W.grad is not None:
#                         grad.append(W.grad.view(-1).detach())
                        grad.append(W.grad.view(-1))
                grads[net_idx].append(torch.cat(grad, -1))
                network.zero_grad()
                torch.cuda.empty_cache()
    ######
    grads = [torch.stack(_grads, 0) for _grads in grads]
    ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]
#     conds = []
#     for ntk in ntks:
#         eigenvalues, _ = torch.symeig(ntk)  # ascending
#         conds.append((eigenvalues[-1] / eigenvalues[0]).item())
#     return conds
    return ntks

# def get_ntk_n(xloader, networks, recalbn=0, train_mode=False, num_batch=-1):
#     device = torch.cuda.current_device()
#     # if recalbn > 0:
#     #     network = recal_bn(network, xloader, recalbn, device)
#     #     if network_2 is not None:
#     #         network_2 = recal_bn(network_2, xloader, recalbn, device)
#     ntks = []
#     for network in networks:
#         if train_mode:
#             network.train()
#         else:
#             network.eval()
#     ######
#     grads = [[] for _ in range(len(networks))]
#     for i, (inputs, targets) in enumerate(xloader):
#         if num_batch > 0 and i >= num_batch: break
#         inputs = inputs.cuda(device=device, non_blocking=True)
#         for net_idx, network in enumerate(networks):
#             network.zero_grad()
#             inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
#             logit = network(inputs_)
#             if isinstance(logit, tuple):
#                 logit = logit[1]  # 201 networks: return features and logits
#             for _idx in range(len(inputs_)):
#                 logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True, create_graph = True)
#                 grad = []
#                 for name, W in network.named_parameters():
#                     if 'weight' in name and W.grad is not None:
#                         # grad.append(W.grad.view(-1).detach())
#                         grad.append(W.grad.view(-1))
#                 grads[net_idx].append(torch.cat(grad, -1))
#                 network.zero_grad()
#                 torch.cuda.empty_cache()
#     ######
#     grads = [torch.stack(_grads, 0) for _grads in grads]
#     ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]
#     return ntks


def ntk_prune(args, nets, data_loader, num_classes, samples_per_class = 1):
    for net in nets:
        net.train()
        net.zero_grad()
        for layer in net.modules():
            snip.add_mask_ones(layer)
    model = nets[0]

    if args.iter_prune:
        num_iter = 10
    else:
        num_iter = 1

    for i in range(num_iter):
        # data_iter = iter(snip_loader)
        ntk = get_ntk_n(data_loader, [model], train_mode = True, num_batch=1, num_classes=num_classes, samples_per_class=samples_per_class)
        # ntk = get_ntk_n(data_loader, [model], train_mode = True, num_batch=1)
        # torch.svd(ntk[0])[1].sum().backward()
        torch.norm(ntk[0]).backward()

        snip.net_iterative_prune(model, args.sparse_lvl**((i+1)/num_iter))
        snip.weight_mask_grad_zero(model)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            weight_check = module.weight
            print(((weight_check!=0).float().sum()/weight_check.numel()))

    for net in nets:
        net.zero_grad()
        net.train()
    print('[*] NTK pruning done!')

def ntk_ep_prune(args, nets, data_loader, num_classes, samples_per_class = 1):
    print('[*] Using NTK+EP pruning.')
    print('[*] Coefficient used is {}'.format(args.ep_coe))
    for net in nets:
        net.train()
        net.zero_grad()
        for layer in net.modules():
            snip.add_mask_ones(layer)
    model = nets[0]

    if args.iter_prune:
        num_iter = 10
    else:
        num_iter = 1

    for i in range(num_iter):
        # data_iter = iter(snip_loader)
        ntk = get_ntk_n(data_loader, [model], train_mode = True, num_batch=1, num_classes=num_classes, samples_per_class=samples_per_class)
        # ntk = get_ntk_n(data_loader, [model], train_mode = True, num_batch=1)
        # torch.svd(ntk[0])[1].sum().backward()
        ntk_loss = torch.norm(ntk[0])
        ep_loss = get_ep(model)
        (ntk_loss + args.ep_coe / ep_loss).backward()

        snip.net_iterative_prune(model, args.sparse_lvl**((i+1)/num_iter))
        snip.weight_mask_grad_zero(model)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            weight_check = module.weight
            print(((weight_check!=0).float().sum()/weight_check.numel()))

    for net in nets:
        net.zero_grad()
        net.train()
    print('[*] NTK+EP pruning done!')

def ntk_prune_adv(args, nets, data_loader, num_classes, samples_per_class = 1):
    for net in nets:
        net.train()
        net.zero_grad()
        for layer in net.modules():
            snip.add_mask_ones(layer)
            snip.add_criterion(layer)
    model = nets[0]
    for _ in range(10):
        utils.kaiming_initialize(model)
        # data_iter = iter(snip_loader)
        ntk = get_ntk_n(data_loader, [model], train_mode = True, num_batch=1, num_classes=num_classes, samples_per_class=samples_per_class)
        # torch.svd(ntk[0])[1].sum().backward()
        torch.norm(ntk[0]).backward()
        snip.update_criterion(model)
        snip.weight_mask_grad_zero(model)

    for net in nets:
        snip.net_prune_advsnip(net, args.sparse_lvl)

    print('[*] NTK pruning done!')

def get_ep(net):
    ep_loss = 0
    for layer in net.modules():          
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
            flatten = layer.weight_mask.view(layer.weight_mask.size(0), -1)
            ep_loss += torch.norm(flatten @ flatten.T - torch.eye(layer.weight_mask.size(0), device = flatten.device))
    return ep_loss