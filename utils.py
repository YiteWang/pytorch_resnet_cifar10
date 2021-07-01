import torch.nn as nn
import numpy as np
import torch
from torch import Tensor
import numpy.linalg as la
import math
import os
import snip
from operator import mul
from functools import reduce
from torchvision.models.resnet import BasicBlock, ResNet
import copy

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

# Code from TE-NAS: https://github.com/VITA-Group/TENAS/blob/main/lib/procedures/linear_region_counter.py
class LinearRegionCount(object):
    """Computes and stores the average and current value"""
    def __init__(self, n_samples):
        self.ActPattern = {}
        self.n_LR = -1
        self.n_samples = n_samples
        self.ptr = 0
        self.activations = None

    @torch.no_grad()
    def update2D(self, activations):
        n_batch = activations.size()[0]
        n_neuron = activations.size()[1]
        self.n_neuron = n_neuron
        if self.activations is None:
            self.activations = torch.zeros(self.n_samples, n_neuron).cuda()
        self.activations[self.ptr:self.ptr+n_batch] = torch.sign(activations)  # after ReLU
        self.ptr += n_batch

    @torch.no_grad()
    def calc_LR(self):
        res = torch.matmul(self.activations.half(), (1-self.activations).T.half()) # each element in res: A * (1 - B)
        res += res.T # make symmetric, each element in res: A * (1 - B) + (1 - A) * B, a non-zero element indicate a pair of two different linear regions
        res = 1 - torch.sign(res) # a non-zero element now indicate two linear regions are identical
        res = res.sum(1) # for each sample's linear region: how many identical regions from other samples
        res = 1. / res.float() # contribution of each redudant (repeated) linear region
        self.n_LR = res.sum().item() # sum of unique regions (by aggregating contribution of all regions)
        del self.activations, res
        self.activations = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def update1D(self, activationList):
        code_string = ''
        for key, value in activationList.items():
            n_neuron = value.size()[0]
            for i in range(n_neuron):
                if value[i] > 0:
                    code_string += '1'
                else:
                    code_string += '0'
        if code_string not in self.ActPattern:
            self.ActPattern[code_string] = 1

    def getLinearReginCount(self):
        if self.n_LR == -1:
            self.calc_LR()
        print('Number of neurons:{}'.format(self.n_neuron))
        return self.n_LR


class Linear_Region_Collector:
    def __init__(self, dataloader, models=[], input_size=(64, 3, 32, 32), sample_batch=100):
        self.models = []
        self.input_size = input_size  # BCHW
        self.sample_batch = sample_batch
        self.input_numel = reduce(mul, self.input_size, 1)
        self.interFeature = []
        self.hook_handles = []
        self.reinit(models, input_size, sample_batch, dataloader)

    def reinit(self, models=None, input_size=None, sample_batch=None, dataloader=None):
        if models is not None:
            assert isinstance(models, list)
            del self.models
            self.models = models
            for model in self.models:
                self.register_hook(model)
            self.LRCounts = [LinearRegionCount(self.input_size[0]*self.sample_batch) for _ in range(len(models))]
        if input_size is not None or sample_batch is not None:
            if input_size is not None:
                self.input_size = input_size  # BCHW
                self.input_numel = reduce(mul, self.input_size, 1)
            if sample_batch is not None:
                self.sample_batch = sample_batch
            if dataloader is not None:
                self.train_loader = dataloader
                self.loader = iter(self.train_loader)
        del self.interFeature
        self.interFeature = []
        torch.cuda.empty_cache()

    def clear(self):
        self.LRCounts = [LinearRegionCount(self.input_size[0]*self.sample_batch) for _ in range(len(self.models))]
        del self.interFeature
        self.interFeature = []
        torch.cuda.empty_cache()

    def register_hook(self, model):
        for m in model.modules():
            if isinstance(m, nn.ReLU):
                hook_handle = m.register_forward_hook(hook=self.hook_in_forward)
                self.hook_handles.append(hook_handle)

    def clear_hooks(self):
        for hook_handle in self.hook_handles:
            hook_handle.remove()

    def hook_in_forward(self, module, input, output):
        if isinstance(input, tuple) and len(input[0].size()) == 4:
            self.interFeature.append(output.detach())  # for ReLU

    def forward_batch_sample(self):
        for _ in range(self.sample_batch):
            try:
                inputs, targets = self.loader.next()
            except Exception:
                del self.loader
                self.loader = iter(self.train_loader)
                inputs, targets = self.loader.next()
            for model, LRCount in zip(self.models, self.LRCounts):
                self.forward(model, LRCount, inputs)
        return [LRCount.getLinearReginCount() for LRCount in self.LRCounts]

    def forward(self, model, LRCount, input_data):
        self.interFeature = []
        with torch.no_grad():
            model.forward(input_data.cuda())
            if len(self.interFeature) == 0: return
            feature_data = torch.cat([f.view(input_data.size(0), -1) for f in self.interFeature], 1)
            LRCount.update2D(feature_data)

def get_ntk_eig(xloader, networks, recalbn=0, train_mode=False, num_batch=-1, num_classes=10, samples_per_class=1, grasp_fetch=True):
    device = torch.cuda.current_device()
    # if recalbn > 0:
    #     network = recal_bn(network, xloader, recalbn, device)
    #     if network_2 is not None:
    #         network_2 = recal_bn(network_2, xloader, recalbn, device)
    if num_classes>=32:
        num_classes = 32
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
        if grasp_fetch:
            inputs = snip.GraSP_fetch_data(data_iter, num_classes, samples_per_class)[0].cuda(device=device, non_blocking=True)
        else:
            inputs = data_iter.next()[0].cuda(device=device, non_blocking=True)
        for net_idx, network in enumerate(networks):
            network.zero_grad()
            inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
            logit = network(inputs_)
            if isinstance(logit, tuple):
                logit = logit[1]  # 201 networks: return features and logits
            for _idx in range(len(inputs_)):
                logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
                grad = []
                for name, W in network.named_parameters():
                    if 'weight' in name and W.grad is not None:
                        grad.append(W.grad.view(-1).detach())
                grads[net_idx].append(torch.cat(grad, -1))
                network.zero_grad()
                torch.cuda.empty_cache()
    ######
    grads = [torch.stack(_grads, 0) for _grads in grads]
    ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]
    conds = []
    for ntk in ntks:
        eigenvalues, _ = torch.symeig(ntk)  # ascending
        conds.append(eigenvalues.detach().cpu().numpy())
    return conds


class BasicBlock_wo_skip(BasicBlock):
    def __init__(self, inplanes,planes,stride=1,downsample= None,groups= 1,base_width= 64,dilation= 1,norm_layer= None):
        super(BasicBlock_wo_skip, self).__init__(inplanes,planes,stride,downsample,groups,base_width,dilation,norm_layer)
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out)

        return out
    
class ResNet_with_gap(ResNet):
    def __init__(self,block,layers,num_classes= 1000,zero_init_residual= False,groups= 1,width_per_group= 64,replace_stride_with_dilation= None,norm_layer= None):
        super(ResNet_with_gap, self).__init__(block,layers,num_classes,zero_init_residual,groups,width_per_group,replace_stride_with_dilation,norm_layer)
    def GAP(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

def get_zenscore(model, data_loader, arch, num_classes):
    # Clear stats of BN and make BN layer eval mode
    if arch == 'resnet18':
        # print('[*] Creating resnet18 clone without skip connection.')
        # model_clone = ResNet_wo_skip(BasicBlock_wo_skip, [2,2,2,2]).cuda()
        # model_clone.fc = nn.Linear(512, num_classes).cuda()
        # for (layer_clone, layer) in zip(model_clone.modules(), model.modules()):
        #     if isinstance(layer, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        #         if hasattr(layer, 'weight_mask'):
        #             snip.add_mask_ones(layer_clone)
        #             layer_clone.weight_mask.data = layer.weight_mask.data
        #         print('[*] Creating resnet18 clone without skip connection.')
        model_clone = ResNet_with_gap(BasicBlock, [2,2,2,2]).cuda()
        model_clone.fc = nn.Linear(512, num_classes).cuda()
        for (layer_clone, layer) in zip(model_clone.modules(), model.modules()):
            if isinstance(layer, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                if hasattr(layer, 'weight_mask'):
                    snip.add_mask_ones(layer_clone)
                    layer_clone.weight_mask.data = layer.weight_mask.data
    else:
        model_clone = copy.deepcopy(model).cuda()

    for module in model_clone.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.running_mean.fill_(0)
            module.running_var.fill_(1)
    model_clone.train()

    data_iter = iter(data_loader)
    input_size = data_iter.next()[0].shape

    # Calculate Zen-Score
    nx = 10
    ne = 10
    GAP_zen = []
    output_zen = []
    with torch.no_grad():
        for _ in range(nx):
            # for module in model_clone.modules():
                # if hasattr(module,'weight') and isinstance(module,(nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                    # module.weight.data = torch.randn(module.weight.size(),device=module.weight.device)
            for _ in range(ne):
                input = torch.empty(input_size)
                nn.init.normal_(input)
                noise = torch.empty(input_size)
                nn.init.normal_(noise)
                input = input.cuda()
                noise = noise.cuda()
                GAP_feature = model_clone.GAP(input)
                GAP_feature_perturb = model_clone.GAP(input+0.01*noise)
                output = model_clone(input)
                output_perturb = model_clone(input+0.01*noise)
                GAP_zen.append(torch.norm(GAP_feature_perturb-GAP_feature).item())
                output_zen.append(torch.norm(output_perturb-output).item())
    
    var_prod = 1.0
    for layer in model_clone.modules():
        if isinstance(layer, nn.BatchNorm2d):
            var = layer.running_var
            var_prod += np.log(np.sqrt((var.sum()/len(var)).item()))
    print('[*] Product of variances is: {}'.format(var_prod))
    print('[*] Original Zen are: {},{}'.format(np.mean(GAP_zen), np.mean(output_zen)))
    GAP_zen = np.log(np.mean(GAP_zen))+var_prod
    output_zen = np.log(np.mean(output_zen))+var_prod
    del model_clone
    return [GAP_zen, output_zen]

# Calculate accurate eigenvalues distribution
def get_sv(net, size_hook):
    # Here, iter_sv stores singular values for different layers
    iter_sv = []
    # iter_std = [] # normalized standard deviation by normalizing using the largest SV.
    # iter_avg = []
    iter_svmax = [] # maximum singular value
    iter_sv20 = [] # 50% singular value
    iter_sv50 = [] # 50% singular value
    iter_sv80 = [] # 80% singular value
    iter_kclip = [] # singular values larger than 1e-12
    # iter_sv50p = [] # 50% non-zero singular value/
    # iter_sv80p = [] # 80% non-zero singular value
    # iter_kavg = [] # max condition number/average condition number

    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
            if isinstance(layer, nn.Linear):
                if hasattr(layer, 'weight_q'):
                    weight = layer.weight_q.detach()
                else:
                    weight = layer.weight.detach()
                if hasattr(layer, 'weight_mask'):
                    weight *= layer.weight_mask.detach()
                sv_result = np.zeros(20,)
                s,v,d = torch.svd(weight.view(weight.size(0),-1), compute_uv=False)
                sorted_sv = v.detach().cpu().numpy()
                # sorted_sv_pos = np.array([i for i in sorted_sv if i>0])
                sorted_sv_clip = np.array([i for i in sorted_sv if i>1e-12])
                top10_sv = sorted_sv[:10]
                bot10_sv = sorted_sv[-10:]
                sv_result[:len(top10_sv)] = top10_sv
                sv_result[-len(bot10_sv):] = bot10_sv
            elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                # Notice that layer.weight has shape (C_out, C_in, H, W) and we want to transform it to
                # (H, W, C_in, C_out)
                # Notice that size_hook returns the input size of the layer, which is (Batch, in_channel, H, W)
                # We only want (H, W)
                sv_result = np.zeros(20,)
                if hasattr(layer, 'weight_q'):
                    weight = layer.weight_q.detach()
                else:
                    weight = layer.weight.detach()
                if hasattr(layer, 'weight_mask'):
                    weight *= layer.weight_mask.detach()
                sv = SVD_Conv_Tensor_NP(weight.detach().cpu().permute(2,3,1,0), size_hook[layer].input_shape[2:])
                sorted_sv = np.flip(np.sort(sv.flatten()),0)
                # sorted_sv_pos = np.array([i for i in sorted_sv if i>0])
                sorted_sv_clip = np.array([i for i in sorted_sv if i>1e-12])
                top10_sv = sorted_sv[:10]
                bot10_sv = sorted_sv[-10:]
                sv_result[:len(top10_sv)] = top10_sv
                sv_result[-len(bot10_sv):] = bot10_sv
                # iter_sv.append(sv_result.copy())
                # iter_avg.append(np.mean(sorted_sv))
                # iter_std.append(np.std(sorted_sv))
            iter_sv.append(sv_result.copy())
            # iter_std.append(np.std(sorted_sv/sorted_sv.max()))
            # iter_avg.append(np.mean(sorted_sv))
            iter_svmax.append(sorted_sv.max())
            iter_sv20.append(sorted_sv[int(len(sorted_sv)*0.2)])
            iter_sv50.append(sorted_sv[int(len(sorted_sv)*0.5)])
            iter_sv80.append(sorted_sv[int(len(sorted_sv)*0.8)])
            # iter_kclip12.append(sorted_sv_clip[0]/sorted_sv_clip[-1])
            try:
                iter_kclip.append(sorted_sv_clip[0]/sorted_sv_clip[-1])
            except:
                iter_kclip.append(sorted_sv_clip[0]/1e-12)
            # iter_kavg.append(sorted_sv.max()/sorted_sv.mean())
    return (np.array(iter_sv),  np.array(iter_svmax),  np.array(iter_sv20),
        np.array(iter_sv50), np.array(iter_sv80), 
        np.array(iter_kclip)) 

def Project_weight(net, size_hook, clip_to=0.01):
    with torch.no_grad():
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                layer.weight.data = Clip_OperatorNorm_NP(layer.weight.detach().cpu(), size_hook[layer].input_shape[2:], clip_to)

def Clip_OperatorNorm_NP(filter, inp_shape, clip_to=0.01):
    # compute the singular values using FFT
    # first compute the transforms for each pair of input and output channels
    filter = filter.permute(2,3,1,0)
    transform_coeff = np.fft.fft2(filter, inp_shape, axes=[0, 1])

    # now, for each transform coefficient, compute the singular values of the
    # matrix obtained by selecting that coefficient for
    # input-channel/output-channel pairs
    U, D, V = np.linalg.svd(transform_coeff, compute_uv=True, full_matrices=False)
    D_clipped = np.maximum(D, clip_to)
    if filter.shape[2] > filter.shape[3]:
        clipped_transform_coeff = np.matmul(U, D_clipped[..., None] * V)
    else:
        clipped_transform_coeff = np.matmul(U * D_clipped[..., None, :], V)
    clipped_filter = np.fft.ifft2(clipped_transform_coeff, axes=[0, 1]).real
    args = [range(d) for d in filter.shape]
    return torch.Tensor(clipped_filter[np.ix_(*args)]).permute(3,2,0,1).cuda()

class Hook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input_shape = np.array(input[0].shape)

    def close(self):
        # print(self.input_shape)
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

def kaiming_initialize(network):
    for m in network.modules():
        if isinstance(m, nn.Conv2d):
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    print('Kaiming initialization done!')

def reset_initialization(network):
    for m in network.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.GroupNorm)):
            m.reset_parameters()
    print('Reset initialization done!')

def save_sparsity(network, save_path):
    sparsity = []
    for m in network.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            sparsity.append((m.weight_mask.sum()/m.weight_mask.numel()).item())
    np.save(os.path.join(save_path, 'saved_sparsity.npy'), sparsity)

def check_layer_sparsity(network):
    sparsity = []
    for m in network.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            sparsity.append(((m.weight!=0).float().sum()/m.weight.numel()).item())
    return np.array(sparsity)


def check_sparsity(network):
    param_remain = 0
    param_total = 0
    for m in network.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            param_remain += m.weight_mask.sum()
            param_total += m.weight_mask.numel()
    print('[*] Number of parameters remaining:{}'.format(param_remain))
    return param_remain/param_total

def get_apply_layer(network):
    num_layers = 0
    for m in network.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            num_layers += 1
    return num_layers