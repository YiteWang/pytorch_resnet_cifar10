import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import utils

# code obtained from GraSP paper: https://github.com/alecwangcq/GraSP
def GraSP_fetch_data(dataloader_iter, num_classes, samples_per_class):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    # dataloader_iter = iter(dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx+1], targets[idx:idx+1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)
    return X, y

# Defining masked layer forward methods for normal layers
def mask_Conv2d(self, x):
    if hasattr(self, 'weight_q'):
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        self.weight_q = weight_q
        out = F.conv2d(input_f, weight_q*self.weight_mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out
    else:
        return F.conv2d(x, self.weight*self.weight_mask, self.bias, self.stride, 
            self.padding, self.dilation, self.groups)

def mask_Linear(self, x):
    if hasattr(self, 'weight_q'):
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        self.weight_q = weight_q
        out = F.linear(input_f, weight_q*self.weight_mask, self.bias)
        return out
    else:
        return F.linear(x, self.weight*self.weight_mask, self.bias)

# Copy from nn.forward
def mask_ConvTranspose2d(self, x, output_size = None):
    if hasattr(self, 'weight_q'):
        output_padding = self._output_padding(input_f, output_size, self.stride, self.padding, self.kernel_size)
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.scale
        self.weight_q = weight_q
        out = F.conv_transpose2d(input_f, weight_q*self.weight_mask, self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)
        return out
    else:
        if self.padding_mode != 'zeros':
                raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        return F.conv_transpose2d(
                x, self.weight*self.weight_mask, self.bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation)

# Apply SNIP pruning methods
def apply_snip(args, nets, data_loader, criterion, num_classes, samples_per_class = 10):
    
    # first add masks to each layer of nets
    for net in nets:
        net.train()
        net.zero_grad()
        for layer in net.modules():
            add_mask_ones(layer)
    model = nets[0]
    # data_iter = iter(snip_loader)
    data_iter = iter(data_loader)
    # Let the neural network run one forward pass to get connect sensitivity (CS)
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
        loss = criterion(output, target_var)
        loss.backward()

    
    # prune the network using CS
    for net in nets:
        net_prune_snip(net, args.sparse_lvl)

    print('[*] SNIP pruning done!')

def update_criterion(net):
    with torch.no_grad():
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
                layer.criterion += torch.abs(layer.weight_mask.grad)

# Apply SNIP pruning methods
def apply_advsnip(args, nets, data_loader, criterion, num_classes, samples_per_class = 10):
    
    # first add masks to each layer of nets
    for net in nets:
        net.train()
        net.zero_grad()
        for layer in net.modules():
            add_mask_ones(layer)  
            add_criterion(layer)
    model = nets[0]
    # data_iter = iter(snip_loader)
    if args.iter_prune:
        num_iter = 10
    else:
        num_iter = 1 
    # Let the neural network run one forward pass to get connect sensitivity (CS)
    for i in range(num_iter):
        data_iter = iter(data_loader)
        for _ in range(10): # number of iterations for taking expectation
            utils.kaiming_initialize(model)
            for i in range(1):
                (input, target) = GraSP_fetch_data(data_iter, num_classes, samples_per_class)
                # (input, target) = data_iter.next()
                target = target.cuda()
                input_var = input.cuda()
                target_var = target
                if args.half:
                    input_var = input_var.half()
                # compute output
                output = model(input_var)
                loss = criterion(output, target_var)
                loss.backward()

            # Summing the weight mask absolute gradient
            update_criterion(model)
            # Zero out gradient of weight mask
            weight_mask_grad_zero(model)

        # prune the network using CS
        for net in nets:
            net_prune_advsnip(net, args.sparse_lvl**((i+1)/num_iter))

    print('[*] SNIP pruning done!')

# zero out gradient of weight mask
def weight_mask_grad_zero(net):
    with torch.no_grad():
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
                layer.weight_mask.grad = None

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

# Prune network based on criterion
def net_prune_advsnip(net, sparse_lvl):
    grad_mask = {}
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
            grad_mask[layer]=torch.abs(layer.criterion*layer.weight_mask)

    # find top sparse_lvl number of elements
    grad_mask_flattened = torch.cat([torch.flatten(a) for a in grad_mask.values()])
    grad_mask_sum = torch.sum(grad_mask_flattened)
    # grad_mask_sum = 1
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
                # layer.weight_mask.requires_grad = False

                # Set those pruned weight as 0 as well, Here needs to address spectral_norm layer
                if hasattr(layer, 'weight_orig'): 
                    layer.weight_orig *= layer.weight_mask
                else:
                    layer.weight *= layer.weight_mask
                layer.criterion = torch.zeros(layer.criterion.size(), device = layer.criterion.device)

# RANDOM PRUNING METHODS
def apply_rand_prune(nets, sparse_lvl, only_G=False):
    # first add masks to each layer of nets
    with torch.no_grad():
        if only_G:
            for layer in nets[0].modules():
                add_mask_rand(layer, sparse_lvl)
        else:
            for net in nets:
                for layer in net.modules():
                    add_mask_rand(layer, sparse_lvl)

def apply_rand_prune_givensparsity(nets, sparsity):
    applied_layer = 0
    with torch.no_grad():
        for net in nets:
            for layer in net.modules():
                if isinstance(layer, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                    if sparsity[applied_layer] != 1:
                        add_mask_rand(layer, sparsity[applied_layer])
                    else:
                        add_mask_ones(layer)
                    applied_layer += 1
            deactivate_mask_update(net)

def apply_prune_active(nets):
    with torch.no_grad():
        for net in nets:
            for layer in net.modules():
                if isinstance(layer, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                    add_mask_current_active(layer)

def add_mask_current_active(layer):
    # It means such layers are using spectral_norm layers
    with torch.no_grad():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
            # Handling spectral_norm layer
            if hasattr(layer, 'weight_orig'):
                layer.weight_mask = (layer.weight_orig.data!=0).float()
                layer.weight_orig.data = layer.weight_mask * layer.weight_orig.data
            else:
                layer.weight_mask = (layer.weight.data!=0).float()
                layer.weight.data = layer.weight_mask * layer.weight
            modify_mask_forward(layer)

def add_mask_rand(layer, sparse_lvl, modify_weight=True, structured=False):
    # It means such layers are using spectral_norm layers
    with torch.no_grad():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
            # Handling spectral_norm layer
            if hasattr(layer, 'weight_orig'):
                weight_num = layer.weight_orig.nelement()
                weight_mask_temp = torch.zeros(weight_num, device=layer.weight_orig.device)
                weight_mask_temp[:int(weight_num*sparse_lvl)] = 1
                if not structured:
                    layer.weight_mask = weight_mask_temp[torch.randperm(weight_num)].view(layer.weight_orig.size())
                else:
                    layer.weight_mask = weight_mask_temp.view(layer.weight_orig.size())
                if modify_weight:
                    layer.weight_orig.data = layer.weight_mask * layer.weight_orig.data
            else:
                weight_num = layer.weight.nelement()
                weight_mask_temp = torch.zeros(weight_num, device=layer.weight.device)
                weight_mask_temp[:int(weight_num*sparse_lvl)] = 1
                if not structured:
                    layer.weight_mask = weight_mask_temp[torch.randperm(weight_num)].view(layer.weight.size())
                else:
                    layer.weight_mask = weight_mask_temp.view(layer.weight.size())
                if modify_weight:
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

def add_criterion(layer):
    with torch.no_grad():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
            # Handling spectral_norm layer
            if hasattr(layer, 'weight_orig'): 
                layer.criterion = torch.zeros(layer.weight_orig.size(), device=layer.weight_orig.device)
            else:
                layer.criterion = torch.zeros(layer.weight.size(), device=layer.weight.device)

def activate_weight_mask(layer):
    with torch.no_grad():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
            layer.weight_mask.requires_grad = True

def deactivate_mask_update(net):
    with torch.no_grad():
        for layer in net.modules():          
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
                layer.weight_mask.requires_grad = False

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