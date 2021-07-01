import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import snip
import numpy as np

# Apply feature pruning methods
def apply_zenprune(args, nets, data_loader):
    print('[*] Zen-Prune starts.')
    for net in nets:
        # net.eval()
        net.train()
        net.zero_grad()
        for layer in net.modules():
            snip.add_mask_ones(layer)
            # if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
            #     nn.init.normal_(layer.weight)
    
    model = nets[0]
    data_iter = iter(data_loader)
    imagesize = data_iter.next()[0].shape
    
    if args.iter_prune:
        num_iter = 100
    else:
        num_iter = 1

    n_x = 10
    n_eta = 10
    eta = 0.01

    for i in range(num_iter):
        # Zero out gradients for weight_mask so to start a new round of iterative pruning
        model.zero_grad()
        for layer in model.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                layer.weight_mask.grad = None
            if isinstance(layer, nn.BatchNorm2d):
                layer.running_mean.fill_(0)
                layer.running_var.fill_(1)

        for _ in range(n_x):
            # initialize weights drawn from Normal distribution N(0,1)
            # with torch.no_grad():
            #     for module in model.modules():
            #         if hasattr(module,'weight') and isinstance(module,(nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            #             module.weight.data = torch.randn(module.weight.size(),device=module.weight.device)

            # Taking expectation w.r.t eta
            for _ in range(n_eta):
                input = torch.empty(imagesize)
                nn.init.normal_(input)
                noise = torch.empty(imagesize)
                nn.init.normal_(noise)
                input = input.cuda()
                noise = noise.cuda()
                output = model(input)
                output_perturb = model(input+0.01*noise)
                zen_score = torch.norm(output-output_perturb)
                zen_score.backward()

        # snip.prune_net_increaseloss(model, args.sparse_lvl**((i+1)/num_iter))
        # snip.net_iterative_prune(model, args.sparse_lvl**((i+1)/num_iter))
        snip.net_prune_grasp(model, args.sparse_lvl**((i+1)/num_iter))
        if i % 5 ==0:
            print('Prune ' + str(i) + ' iterations.')
            print('Zen-score is {}'.format(zen_score.item()))
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


## Zen-Score Transfer
def apply_zentransfer(args, nets, data_loader):
    mask_update_freq = 5
    num_iter = 100

    print('[*] Zen-Transfer starts.')
    for net in nets:
        # net.eval()
        net.train()
        net.zero_grad()
        for layer in net.modules():
            snip.add_mask_rand(layer, args.sparse_lvl, modify_weight=False, structured=False, requires_grad = False)
            # if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
            #     nn.init.normal_(layer.weight)
    
    model = nets[0]
    data_iter = iter(data_loader)
    imagesize = data_iter.next()[0].shape
    optim = torch.optim.Adam(model.parameters(), 0.1)

    n_eta = 5
    eta = 0.01

    for i in range(num_iter):
        # Zero out gradients for weight so to start a new round of iterative pruning
        model.zero_grad()

        # Taking expectation w.r.t eta
        for _ in range(n_eta):
            input = torch.empty(imagesize)
            nn.init.normal_(input)
            noise = torch.empty(imagesize)
            nn.init.normal_(noise)
            input = input.cuda()
            noise = noise.cuda()
            output = model(input)
            output_perturb = model(input+0.01*noise)
            zen_score = torch.norm(output-output_perturb)
            (-zen_score).backward()

        optim.step()
        if i % 5 == 0:
            print('Prune ' + str(i) + ' iterations.')
            print('Zen-score is {}'.format(zen_score.item()))

        if i % mask_update_freq == 0:
            snip.net_prune_magnitude(model, args.sparse_lvl, modify_weight = False)
            # Reinitialize BatchNorm layers statistics
            # for layer in model.modules():
            #     if isinstance(layer, (nn.Conv2d, nn.Linear)):
            #         layer.weight_mask.grad = None
            #     if isinstance(layer, nn.BatchNorm2d):
            #         layer.running_mean.fill_(0)
            #         layer.running_var.fill_(1)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            mask_check = module.weight_mask
            weight_check = module.weight
            print('Mask sparsity: {}'.format((mask_check!=0).float().sum()/mask_check.numel()))
            print('Weight sparsity: {}'.format((weight_check!=0).float().sum()/weight_check.numel()))
    
    print('-'*20)

    # # remove hooks
    # for handle in handles:
    #     handle.remove()

    # zero out gradients of weights
    for net in nets:
        net.zero_grad()
        net.train()
        # net.reinit()

def apply_cont_zenprune(args, nets, data_loader):

    print('[*] Continuous Zen-Prune starts.')

    # A list of parameters that we want to optimize
    mask_params = []
    # Layers that we want to prune
    candidate_layers = []
    # Total number of parameters that we want to prune
    total_param_num = 0

    for net in nets:
        # net.eval()
        net.train()
        net.zero_grad()
        for layer in net.modules():
            # Add continuous relaxed mask as parameters, initialized with gaussian
            # The forward method of layers when cont_relax is True is that
            # using weight: self.weight * torch.sigmoid(self.weight_mask)
            if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                snip.add_mask_cont(layer, cont_relax=True)
                current_weight_mask = layer.weight_mask
                assert isinstance(current_weight_mask, nn.Parameter)
                candidate_layers.append(layer)
                mask_params.append(current_weight_mask)
                total_param_num += current_weight_mask.nelement()
    
    # Calculating number of parameters that will remain:
    left_params_num = int(total_param_num * args.sparse_lvl)

    model = nets[0]
    data_iter = iter(data_loader)
    imagesize = data_iter.next()[0].shape

    # n_x = 1
    n_eta = 5
    eta = 0.01

    mask_optim = torch.optim.Adam(mask_params, 0.1)
    num_optim = 10

    # We use Adam optimizer to optimize num_optim iterations
    for i in range(num_optim):
        # Zero out gradients for weight_mask and parameters
        model.zero_grad()

        # Taking expectation w.r.t eta
        for _ in range(n_eta):

            # Calculating Zen-scores with batchnorm set to train mode and not 
            # initialize parameters with i.i.d Gaussian
            input = torch.empty(imagesize)
            nn.init.normal_(input)
            noise = torch.empty(imagesize)
            nn.init.normal_(noise)
            input = input.cuda()
            noise = noise.cuda()
            output = model(input)
            output_perturb = model(input+0.01*noise)
            zen_score = torch.norm(output-output_perturb)

            # We want to maximize Zen score with a constraint that torch.sigmoid(weight_mask)
            # is roughly remaining number of params
            current_param_num = 0
            for layer in candidate_layers:
                current_param_num += (torch.sigmoid(layer.weight_mask)).sum()
            loss = -zen_score + 0.1 * (current_param_num - left_params_num)**2
            loss.backward()

        # Update weight_mask
        mask_optim.step()

        if i % 5 ==0:
            print('Prune ' + str(i) + ' iterations.')
            print('Zen-score is {}'.format(zen_score.item()))

    print('[*] Final Zen-score (Continuous): {}'.format(zen_score))

    # First choose top total_param_num * args.sparse_lvl parameters
    all_weight_mask = {}
    for layer in candidate_layers:
        all_weight_mask[layer] = layer.weight_mask
    all_weight_mask_flattened = torch.cat([torch.flatten(a) for a in all_weight_mask.values()])
    normalizer = torch.sum(all_weight_mask_flattened)
    all_weight_mask_flattened /= normalizer

    weight_mask_topk, _ = torch.topk(all_weight_mask_flattened, left_params_num)
    threshold = weight_mask_topk[-1]

    # Modify weight mask of each layer
    with torch.no_grad():
        for layer in candidate_layers:
            temp_mask = (layer.weight_mask >= threshold).float()
            print('sparsity level: {}'.format(temp_mask.sum()/temp_mask.nelement()))
            # Unregister weight_mask from parameters and replace it with tensors
            delatrr(layer, 'weight_mask')
            layer.weight_mask = temp_mask
            # Modify forward method to mask forward
            snip.modify_mask_forward(layer, cont_relax = False)
    
    print('-'*20)

    for net in nets:
        net.zero_grad()
        net.train()

def apply_nsprune(args, nets, data_loader, num_classes, samples_per_class = 10, GAP=True):
    print('Using GAP is {}'.format(GAP))

    for net in nets:
        net.eval()
        # net.train()
        net.zero_grad()
        for layer in net.modules():
            snip.add_mask_ones(layer)
    
    model = nets[0]
    data_iter = iter(data_loader)
    imagesize = data_iter.next()[0].shape
    
    if args.iter_prune:
        num_iter = 100
    else:
        num_iter = 1

    n_x = 10
    n_eta = 10
    eta = 0.01
    for i in range(num_iter):
        # Taking expectaion w.r.t x
        model.zero_grad()
        for layer in model.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                layer.weight_mask.grad = None

        ns_tracking = 0
        for _ in range(n_x):
            try:
                (input, target) = snip.GraSP_fetch_data(data_iter, num_classes, samples_per_class)
            except:
                data_iter = iter(data_loader)
                (input, target) = snip.GraSP_fetch_data(data_iter, num_classes, samples_per_class)
            input = input.cuda()
            norm_x = torch.norm(input)
            # Taking expectation w.r.t  eta
            for _ in range(n_eta):
                
                # noise = torch.ones(input.size())*torch.randn(1,)*eta*norm_x
                noise = torch.randn(input.size())*eta*norm_x
                input_perturb = input + noise.cuda()
                if GAP:
                    output = model.GAP(input)
                    output_perturb = model.GAP(input_perturb)
                else:
                    output = model(input)
                    output_perturb = model(input_perturb)
                perturbation = torch.norm(output-output_perturb)/norm_x
                perturbation.backward()
                ns_tracking += perturbation.item()

        snip.net_iterative_prune_wolinear(model, args.sparse_lvl**((i+1)/num_iter))
        # snip.prune_net_decreaseloss(model, args.sparse_lvl**((i+1)/num_iter), True)
        if i % 10 ==0:
            print('Prune ' + str(i) + ' iterations, noise sensitivity:{}'.format(ns_tracking))

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            weight_check = module.weight
            print(((weight_check!=0).float().sum()/weight_check.numel()))
    
    print('-'*20)

    for net in nets:
        net.zero_grad()
        net.train()

def apply_SAP(args, nets, data_loader, criterion, num_classes, samples_per_class = 10):
    print('[*] Currently using SAP pruning.')
    for net in nets:
        # net.eval()
        net.train()
        net.zero_grad()
        for layer in net.modules():
            snip.add_mask_ones(layer)
    
    model = nets[0]
    data_iter = iter(data_loader)
    imagesize = data_iter.next()[0].shape
    
    if args.iter_prune:
        num_iter = 100
    else:
        num_iter = 1


    for layer in model.modules():
        if isinstance(layer,(nn.Conv2d, nn.Linear)):
            layer.base_weight = layer.weight.detach()

    n_x = 10
    n_eta = 10
    eta = 0.01
    
    for i in range(num_iter):
        # Taking expectaion w.r.t x
        model.zero_grad()
        for layer in model.modules():
            if isinstance(layer,(nn.Conv2d, nn.Linear)):
                layer.weight_mask.grad = None
                
        for _ in range(n_x):
            try:
                (input, target) = snip.GraSP_fetch_data(data_iter, num_classes, samples_per_class)
            except:
                data_iter = iter(data_loader)
                (input, target) = snip.GraSP_fetch_data(data_iter, num_classes, samples_per_class)
            target_var = target.cuda()
            input_var = input.cuda()
            # Taking expectation w.r.t  eta
            for _ in range(n_eta):
                with torch.no_grad():
                    for layer in model.modules():
                        if isinstance(layer,(nn.Conv2d, nn.Linear)):
                            layer.weight.data = layer.base_weight + eta*torch.randn(layer.weight.size(),device = layer.weight.device)
                # compute output
                output = model(input_var)
                loss = criterion(output, target_var)
                loss.backward()
            ##################################################
            # This part is for adversarial perturbations
            ##################################################
            # output = model(input_var)
            # loss = criterion(output, target_var)
            # loss.backward()

            # with torch.no_grad():
            #     for layer in model.modules():
            #         if isinstance(layer,(nn.Conv2d, nn.Linear)):
            #             layer.weight_mask.grad = None
            #             layer.weight.data = layer.base_weight + eta*layer.weight.grad/torch.norm(eta*layer.weight.grad)
            # # compute output
            # output = model(input_var)
            # loss = criterion(output, target_var)
            # loss.backward()

            
        with torch.no_grad():
            for layer in model.modules():
                if isinstance(layer,(nn.Conv2d, nn.Linear)):
                    layer.weight.data = layer.base_weight
        snip.net_iterative_prune(model, args.sparse_lvl**((i+1)/num_iter))
        # snip.prune_net_decreaseloss(model, args.sparse_lvl**((i+1)/num_iter), True)
        if i % 10 ==0:
            print('Prune ' + str(i) + ' iterations')

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            weight_check = module.weight
            print(((weight_check!=0).float().sum()/weight_check.numel()))
    
    print('-'*20)

    for net in nets:
        net.zero_grad()
        net.train()
# def apply_zenprune(args, nets, data_loader, num_classes, samples_per_class = 10):
#     for net in nets:
#         net.eval()
#         net.zero_grad()
#         for layer in net.modules():
#             snip.add_mask_ones(layer)
    
#     model = nets[0]
#     data_iter = iter(data_loader)
#     imagesize = data_iter.next()[0].shape
    
#     if args.iter_prune:
#         num_iter = 100
#     else:
#         num_iter = 1

#     for i in range(num_iter):
#         for _ in range(10):
#             # input = torch.empty(imagesize)
#             try:
#                 (input, target) = snip.GraSP_fetch_data(data_iter, num_classes, samples_per_class)
#             except:
#                 data_iter = iter(data_loader)
#                 (input, target) = snip.GraSP_fetch_data(data_iter, num_classes, samples_per_class)
#             # nn.init.normal_(input)
#             input = input.cuda()
#             output = model.GAP(input)
#             GAP_norm = torch.norm(output)
#             GAP_norm.backward()
#         print('The total GAP norm is:{}'.format(GAP_norm))
#         snip.net_iterative_prune_wolinear(model, args.sparse_lvl**((i+1)/num_iter))
#         if i % 10 ==0:
#             print('Prune ' + str(i) + ' iterations.')

#     for module in model.modules():
#         if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
#             weight_check = module.weight
#             print(((weight_check!=0).float().sum()/weight_check.numel()))
    
#     print('-'*20)

#     # # remove hooks
#     # for handle in handles:
#     #     handle.remove()

#     # zero out gradients of weights
#     for net in nets:
#         net.zero_grad()
#         net.train()
