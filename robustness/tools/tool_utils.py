import torch as ch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector as flatten
import copy

def get_params(model):
    # Get flattened model weights
    return flatten(model.parameters()).clone().detach()

def get_norm(v):
    # Get l2 norm of tensor
    return ch.norm(v).cpu().numpy()

## Get model weights with BN parameters absorbed ##
def get_abs_weights(orig_model):
    copy_model = copy.deepcopy(orig_model)
    absorb_bn_model(copy_model)
    return get_params(copy_model)

def absorb_bn_model(model):
    modules = list(model.modules())
    for midx, m in enumerate(modules):
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            absorb_bn(modules[midx-1], m)

def absorb_bn(module, bn_module): 
    with ch.no_grad():
        w = module.weight
        if module.bias is None:
            zeros = ch.zeros(module.out_channels,
                                dtype=w.dtype, device=w.device)
            bias = nn.Parameter(zeros)
            module.register_parameter('bias', bias)
        b = module.bias

        if hasattr(bn_module, 'running_mean'):
            b.add_(-bn_module.running_mean)
        if hasattr(bn_module, 'running_var'):
            invstd = bn_module.running_var.clone().add_(bn_module.eps).pow_(-0.5)
            w.mul_(invstd.view(w.size(0), 1, 1, 1))
            b.mul_(invstd)

        if hasattr(bn_module, 'weight'):
            w.mul_(bn_module.weight.view(w.size(0), 1, 1, 1))
            b.mul_(bn_module.weight)
        if hasattr(bn_module, 'bias'):
            b.add_(bn_module.bias)

        bn_module.register_buffer('running_mean', None)
        bn_module.register_buffer('running_var', None)
        bn_module.register_parameter('weight', None)
        bn_module.register_parameter('bias', None)
