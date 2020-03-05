import torch as ch
import torch.nn as nn
import numpy as np
from scipy.ndimage import rotate

def rand_label_transform(nclasses):
    # Random labels
    def make_rand_labels(ims, targs):
        new_targs = (ch.randint(1, high=nclasses,size=targs.shape).long() + targs) % nclasses        
        return ims, new_targs
    return make_rand_labels


def apply_noise(xi, noise_type='block'):
    assert noise_type in ['block', 'img', 'gaussian', 'constant']
    if noise_type == 'block':
        s = xi.shape[-1] // 4
        x = np.kron(np.random.rand(xi.shape[0], xi.shape[1], 4, 4), np.ones((1, 1, s, s))) / 4.0 + 0.5
        x = np.stack([rotate(x[i], np.random.randint(0, 90), axes=[1, 2],
                     reshape=False, 
                     mode='reflect') for i in range(x.shape[0])], axis=0)
        x = ch.tensor(x, dtype=xi.dtype)
    elif noise_type == 'img':
        m = ch.mean(ch.mean(xi, dim=2), dim=2)[:, :, None, None].repeat(1, 1, xi.shape[-2], xi.shape[-1])
        s = ch.std(ch.std(xi, dim=2), dim=2)[:, :, None, None].repeat(1, 1, xi.shape[-2], xi.shape[-1])
        x = ch.normal(m, s)
    elif noise_type == 'gaussian':
        x = ch.randn_like(xi) / 20.0 + 0.5
    elif noise_type == 'constant': 
        x = ch.ones_like(xi) * ch.rand(xi.shape[0], 3, 1, 1).expand_as(xi)
    else:
        x = ch.rand_like(xi) / 8.0 + 0.5
    assert ch.all(ch.eq(ch.tensor(x.shape), ch.tensor(xi.shape))) 
    return ch.clamp(x, 0, 1)


def noise_transform(noise_type):
    # Noise
    def noisy_samples(ims, targs):
        new_targs = -1 * ch.ones_like(targs)
        new_ims = apply_noise(ims, noise_type=noise_type)
        return (new_ims, new_targs)
    return noisy_samples

