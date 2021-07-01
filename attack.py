import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

# attack methods
def shuffle_mask(net):
    with torch.no_grad():
        for layer in net.modules():
            shuffle_layer_mask(layer)

def shuffle_layer_mask(layer):
    # It means such layers are using spectral_norm layers
    with torch.no_grad():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
            weight_num = layer.weight.nelement()
            nonzero_num = torch.sum(layer.weight_mask)
            weight_mask_temp = torch.zeros(weight_num, device=layer.weight.device)
            weight_mask_temp[:int(nonzero_num)] = 1
            layer.weight_mask = weight_mask_temp[torch.randperm(weight_num)].view(layer.weight.size())