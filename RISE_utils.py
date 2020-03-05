##########################################################################################
 ## Taken from here - https://github.com/eclique/RISE/blob/master/utils.py
 # USED for IOU computation
##########################################################################################

import numpy as np
from matplotlib import pyplot as plt
import torch, skimage
from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets
from PIL import Image


# Dummy class to store arguments
class Dummy():
    pass


# # Function that opens image from disk, normalizes it and converts to tensor
# read_tensor = transforms.Compose([
#     lambda x: Image.open(x),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                           std=[0.229, 0.224, 0.225]),
#     lambda x: torch.unsqueeze(x, 0)
# ])

def read_tensor(img_path, transform, if_noise=0):
    img = Image.open(img_path).convert('RGB')
    if if_noise == 1:
        img = skimage.util.random_noise(np.asarray(img), mode='gaussian',
                                        mean=0, var=0.1, seed=0,
                                        )  # numpy, dtype=float64,range (0, 1)
        img = Image.fromarray(np.uint8(img * 255))

    img = transform(img)
    return img.unsqueeze(0)


# Plots image from tensor
def tensor_imshow(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, **kwargs)
    if title is not None:
        plt.title(title)


# Given label number returns class name
def get_class_name(c):
    labels = np.loadtxt('synset_words.txt', str, delimiter='\t')
    return ' '.join(labels[c].split(',')[0].split()[1:])


# Image preprocessing function
preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # Normalization for ImageNet
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])


# Sampler for pytorch loader. Given range r loader will only
# return dataset[r] instead of whole dataset.
class RangeSampler(Sampler):
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        return iter(self.r)

    def __len__(self):
        return len(self.r)
