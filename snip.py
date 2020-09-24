import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import utils

# Defining masked layer forward methods for normal layers
def mask_Conv2d(self, x):
    return F.conv2d(x, self.weight*self.weight_mask, self.bias, self.stride, 
        self.padding, self.dilation, self.groups)

def mask_Linear(self, x):
    return F.linear(x, self.weight*self.weight_mask, self.bias)

# Copy from nn.forward
def mask_ConvTranspose2d(self, x, output_size = None):
    if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')
    output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
    return F.conv_transpose2d(
            x, self.weight*self.weight_mask, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)

# Apply SNIP pruning methods
def apply_snip(args, nets, snip_loader, criterion, device):
    
    # first add masks to each layer of nets
    for net in nets:
        net.train()
        net.zero_grad()
        for layer in net.modules():
            add_mask_ones(layer)

    # Let the neural network run one forward pass to get connect sensitivity (CS)
    for i in range(int(args.prunesets_num/30)+1):
        (input, target) = snip_loader[i]
        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        loss.backward()

    
    # prune the network using CS
    for net in nets:
        net_prune_snip(net, args.sparse_lvl)

    print('[*] SNIP pruning done!')


# Calculate gradient for SNIP pruning
def find_loss(args, nets, loader, g_loss_func, d_loss_func, device):
    # Calculate loss and find connect sensitivity
    netG, netD = nets

    # Use first args.prunesets_num images for SNIP
    for _ in range(args.prunesets_num):

        utils.net_require_grad([netD], False)
        utils.net_require_grad([netG], True)

        # Get one sample from data loader
        try:
            x = next(iter(loader))[0].cuda().float()
        except StopIteration:
            data_iter = iter(loader)
            x = next(data_iter)[0].cuda().float()
        # x = next(loader)[0].cuda().float()

        # Generate noise 
        z = torch.randn(args.batch_size, args.input_dim, device=device) 

        # Forward pass of normal GAN
        x_hat = netG(z)
        y_hat = netD(x_hat)
        y = netD(x)

        if args.losstype == 'HH':
            g_loss = g_loss_func('hinge', y_hat, y, sortz=args.sortz)
        else:
            g_loss = g_loss_func('log', y_hat, y, sortz=args.sortz)

        g_loss.backward()

        utils.net_require_grad([netG], False)
        utils.net_require_grad([netD], True)

        # Do the same thing for netD
        try:
            x = next(iter(loader))[0].cuda().float()
        except StopIteration:
            data_iter = iter(loader)
            x = next(data_iter)[0].cuda().float()
        # x = next(loader)[0].cuda().float()

        z = torch.randn(args.batch_size, args.input_dim, device=device)

        x_hat = netG(z).detach()
        y_hat = netD(x_hat)
        y = netD(x)
        if args.losstype == 'log':
            d_loss = d_loss_func('log', y_hat, y, sortz=args.sortz)
        else:
            d_loss = d_loss_func('hinge', y_hat, y, sortz=args.sortz)
        d_loss.backward()

    # Set networks back to active after calculating the gradients
    utils.net_require_grad([netD, netG], True)

# Prune network based on gradient absolute values
def net_prune_snip(net, sparse_lvl):
    grad_mask = {}
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
            grad_mask[layer]=torch.abs(layer.weight_mask.grad)

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


# RANDOM PRUNING METHODS
def apply_rand_prune(nets, sparse_lvl, only_G):
    # first add masks to each layer of nets
    with torch.no_grad():
        if only_G:
            for layer in nets[0].modules():
                add_mask_rand(layer, sparse_lvl)
        else:
            for net in nets:
                for layer in net.modules():
                    add_mask_rand(layer, sparse_lvl)

def add_mask_rand(layer, sparse_lvl):
    # It means such layers are using spectral_norm layers
    with torch.no_grad():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
            # Handling spectral_norm layer
            if hasattr(layer, 'weight_orig'):
                layer.weight_mask = (torch.empty(layer.weight_orig.size(), device=layer.weight_orig.device).uniform_() > (1-sparse_lvl)).float()
                layer.weight_orig.data = layer.weight_mask * layer.weight_orig.data
            else:
                layer.weight_mask = (torch.empty(layer.weight.size(), device=layer.weight.device).uniform_() > (1-sparse_lvl)).float()
                layer.weight.data = layer.weight_mask * layer.weight
        modify_mask_forward(layer)


def add_mask_ones(layer):
    with torch.no_grad():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
            # Handling spectral_norm layer
            if hasattr(layer, 'weight_orig'): 
                layer.weight_mask = torch.ones_like(layer.weight_orig, requires_grad = True, device=layer.weight_orig.device)
            else:
                layer.weight_mask = torch.ones_like(layer.weight, requires_grad = True, device=layer.weight.device)
        modify_mask_forward(layer)

def activate_weight_mask(layer):
    with torch.no_grad():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
            layer.weight_mask.requires_grad = True

def modify_mask_forward(layer):
    if isinstance(layer, nn.Conv2d):
        layer.forward = types.MethodType(mask_Conv2d, layer)
    elif isinstance(layer, nn.ConvTranspose2d):
        layer.forward = types.MethodType(mask_ConvTranspose2d, layer)
    elif isinstance(layer, nn.Linear):
        layer.forward = types.MethodType(mask_Linear, layer)

def get_snip_mask(args, nets, loader, g_loss_func, d_loss_func, device):
    
    # first add masks to each layer of nets
    for net in nets:
        for layer in net.modules():
            if hasattr(layer, 'weight_mask'):
                activate_weight_mask(layer)
            else:
                add_mask_ones(layer)
        net.zero_grad()

    # Let the neural network run one forward pass to get connect sensitivity (CS)
    find_loss(args, nets, loader, g_loss_func, d_loss_func, device)

    modified_masks = []
    # prune the network using CS
    for net in nets:
        grad_mask = {}
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
                grad_mask[layer] = torch.abs(layer.weight_mask.grad)

        # find top sparse_lvl number of elements
        grad_mask_flattened = torch.cat([torch.flatten(a) for a in grad_mask.values()])
        grad_mask_sum = torch.sum(grad_mask_flattened)
        grad_mask_flattened /= grad_mask_sum

        left_params_num = int (len(grad_mask_flattened) * args.sparse_lvl)
        grad_mask_topk, _ = torch.topk(grad_mask_flattened, left_params_num)
        threshold = grad_mask_topk[-1]

        modified_mask = {}
        for layer, mask in grad_mask.items():
            modified_mask[layer] = ((mask/grad_mask_sum)>=threshold).detach().float()
        modified_masks.append(modified_mask)
    for net in nets:
        net.zero_grad()
    return modified_masks