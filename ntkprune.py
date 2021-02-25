import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import snip

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

def ntk_prune(args, nets, data_loader, num_classes, samples_per_class = 1):
    for net in nets:
        net.train()
        net.zero_grad()
        for layer in net.modules():
            snip.add_mask_ones(layer)
    model = nets[0]
    # data_iter = iter(snip_loader)
    ntk = get_ntk_n(data_loader, [model], train_mode = True, num_batch=1, num_classes=num_classes, samples_per_class=samples_per_class)
    torch.svd(ntk[0])[1].sum().backward()

    for net in nets:
        snip.net_prune_snip(net, args.sparse_lvl)

    print('[*] NTK pruning done!')