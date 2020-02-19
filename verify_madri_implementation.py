import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import ipdb
from srblib import abs_path

import sys, glob
import numpy as np
from PIL import Image
import ipdb
# import shutil
import time
import os
# from utils import mkdir_p

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
# import imageio
from termcolor import colored
from robustness import model_utils, datasets
from user_constants import DATA_PATH_DICT

# ipdb.set_trace()
# sys.path.append(os.path.abspath('../'))
# import utils as eutils

import warnings
warnings.filterwarnings("ignore")

import argparse

use_cuda = torch.cuda.is_available()


def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Input paramters for meaningful perturbation explanation of the image')

    parser.add_argument('-idp', '--img_dir_path', help='Path of the image directory', metavar='DIR')

    parser.add_argument('-op', '--out_path',
                        help='Path of the output directory where you want to save the results (Default is ./img_name/)')

    parser.add_argument('-gpu', '--gpu', type=int, choices=range(8),
                        help='GPU index', default=0,
                        )

    parser.add_argument('-wn', '--which_number', type=int,
                        help='Which numbered image to load from validation set (Max:50000). Default=1', default=1,
                        )

    parser.add_argument('-ifpp', '--if_pre_post', type=int, choices=range(2),
                        help='It is clear from name. Default: Pre (1)', default=1,
                        )

    # Parse the arguments
    args = parser.parse_args()
    # ipdb.set_trace()


    if args.img_dir_path is None:
        print('Please provide image dir path. Exiting')
        sys.exit(1)
    args.img_dir_path = abs_path(args.img_dir_path)

    if args.out_path is None:
        args.out_path = './'
    args.out_path = os.path.abspath(args.out_path)

    return args


class DataProcessing:
    def __init__(self, data_path, pytorch_transform, madry_transform, which_number=None):
        self.data_path = data_path
        self.pytorch_transform = pytorch_transform
        self.madry_transform = madry_transform

        self.img_filenames = []

        for file in glob.glob(os.path.join(data_path, "*.JPEG")):
            self.img_filenames.append(file)
        self.img_filenames.sort()

        if which_number is not None:
            self.img_filenames = [self.img_filenames[which_number-1] ]

    def __getitem__(self, index):
        # ipdb.set_trace()
        img = Image.open(os.path.join(self.data_path, self.img_filenames[index])).convert('RGB')
        y = self.get_image_class(os.path.join(self.data_path, self.img_filenames[index]))
        img = img.convert('RGB')

        pytorch_img = self.pytorch_transform(img)
        madry_img = self.madry_transform(img)
        return pytorch_img, madry_img, y, os.path.join(self.data_path, self.img_filenames[index])

    def __len__(self):
        return len(self.img_filenames)

    def get_image_class(self, filepath):
        base_dir = '/home/naman/CS231n/heatmap_tests/'

        # ipdb.set_trace()

        # ImageNet 2012 validation set images?
        with open(os.path.join(base_dir, "imagenet_class_mappings", "ground_truth_val2012")) as f:
            ground_truth_val2012 = {x.split()[0]: int(x.split()[1])
                                    for x in f.readlines() if len(x.strip()) > 0}
        with open(os.path.join(base_dir, "imagenet_class_mappings", "synset_id_to_class")) as f:
            synset_to_class = {x.split()[1]: int(x.split()[0])
                               for x in f.readlines() if len(x.strip()) > 0}

        def get_class(f):
            # ipdb.set_trace()
            # File from ImageNet 2012 validation set
            ret = ground_truth_val2012.get(f, None)
            if ret is None:
                # File from ImageNet training sets
                ret = synset_to_class.get(f.split("_")[0], None)
            if ret is None:
                # Random JPEG file
                ret = 1000
            return ret

        image_class = get_class(filepath.split('/')[-1])
        return image_class

def load_data(img_dir, batch_size=8, which_number=None):

    # 1. Define the appropriate image pre-processing function.
    pytorch_preprocessFn = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

    madry_preprocessFn = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       ])

    data = DataProcessing(img_dir, pytorch_preprocessFn, madry_preprocessFn, which_number=which_number)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    return test_loader, len(data)

def load_madry_model(dev_idx):
    DATA = 'ImageNet'  # Choices: ['CIFAR', 'ImageNet', 'RestrictedImageNet']
    DATA_SHAPE = 32 if DATA == 'CIFAR' else 224  # Image size (fixed for dataset)
    BATCH_SIZE = 2
    NUM_WORKERS = 1

    dataset_function = getattr(datasets, DATA)
    dataset = dataset_function(DATA_PATH_DICT[DATA])

    # _, test_loader = dataset.make_loaders(workers=NUM_WORKERS,
    #                                       batch_size=BATCH_SIZE // 2,
    #                                       data_aug=False, only_val=True)
    # data_iterator = enumerate(test_loader)
    #
    # _, (im, targ, im_path) = next(data_iterator)
    #
    # print(f'Orig Class loaded by Madry is: {targ}')
    # print(f'Image path by Madry is {im_path}')

    ## ResNet - 50
    # # Load model
    # model_kwargs = {
    #     'arch': 'resnet50',
    #     'dataset': dataset,
    #     'resume_path': f'./models/{DATA}.pt'
    # }

    ## GoogleNet
    # Load model
    model_kwargs = {
        'arch': 'googlenet',
        'dataset': dataset,
        'resume_path': f'./models/ImageNet_GoogleNet.pt.best',
        'my_attacker': True,
        'parallel': False,
    }

    model_kwargs['state_dict_path'] = 'model'
    model, _ = model_utils.make_and_restore_model(**model_kwargs)
    model.eval()

    ipdb.set_trace()

    if use_cuda:
        model.cuda(dev_idx)

    # logits = model(im)
    # # ipdb.set_trace()
    # probs = F.softmax(logits[0], dim=1)
    #
    # print(f'Prediction by Madry Method: Class: {torch.argmax(probs)}, Prob: {torch.max(probs)}')
    # # ipdb.set_trace()


    return model #, int(im_path[0].split('/')[-1].split('.')[0].split('_')[-1]), im


def his_code():
    # Constants
    DATA = 'ImageNet'  # Choices: ['CIFAR', 'ImageNet', 'RestrictedImageNet']
    BATCH_SIZE = 1000
    # BATCH_SIZE = 2
    NUM_WORKERS = 8
    NOISE_SCALE = 20

    DATA_SHAPE = 32 if DATA == 'CIFAR' else 224  # Image size (fixed for dataset)
    REPRESENTATION_SIZE = 2048  # Size of representation vector (fixed for model)

    # Load dataset
    dataset_function = getattr(datasets, DATA)
    dataset = dataset_function(DATA_PATH_DICT[DATA])
    _, test_loader = dataset.make_loaders(workers=NUM_WORKERS,
                                          batch_size=BATCH_SIZE // 2,
                                          data_aug=False, only_val=True)
    data_iterator = enumerate(test_loader)

    # Load model
    model_kwargs = {
        'arch': 'resnet50',
        'dataset': dataset,
        'resume_path': f'./models/{DATA}.pt'
    }
    model_kwargs['state_dict_path'] = 'model'
    model, _ = model_utils.make_and_restore_model(**model_kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # _, (im, targ, path) = next(data_iterator)
    # probs = model(im)

    correct = 0
    for idx, (im, targ, path) in data_iterator:
        print(f'Batch is: {idx}')
        targ = targ.cpu().numpy()
        log = model(im)
        probs = F.softmax(log, dim=1).cpu()

        labels = torch.argmax(probs, dim=-1).numpy()
        correct += np.size(np.where((targ - labels) == 0))

        print(f'Correct images are {correct}/50000')

    print(f'Correct images are {correct}/50000')
    ipdb.set_trace()
    aa = 1



    return im, probs, path[0], targ.cpu().item()


if __name__ == '__main__':
    f_time = ''.join(str(time.time()).split('.'))
    args = get_arguments()

    ###################################
    # ## His code
    # im, probs, path, targ = his_code()
    # im = im.cpu()


    ####################################

    madry_model = load_madry_model(args.gpu)

    ipdb.set_trace()

    #########################################################
    # ipdb.set_trace()
    args.which_number = int(path.split('/')[-1].split('.')[0].split('_')[-1])

    # ipdb.set_trace()
    data_loader, img_count = load_data(args.img_dir_path, batch_size=1, which_number=args.which_number)
    data_iterator = enumerate(data_loader)

    _, (pytorch_img, madry_img, targ_class, img_path) = next(data_iterator) #orig_img (PIL)

    targ_class = targ_class.cpu().item()
    if use_cuda:
        pytorch_img = pytorch_img.cuda(args.gpu)
        madry_img = madry_img.cuda(args.gpu)

    my_probs = madry_model(madry_img)
    madry_img = madry_img.cpu()

    ipdb.set_trace()





    ##Since we want to compute gradients as well
    pytorch_img = Variable(pytorch_img, requires_grad=True)
    madry_img = Variable(madry_img, requires_grad=True)


    print(f'Orig class is {targ_class}')
    print(f'Orig image path is {img_path}')

    pytorch_logits = pytorch_model(pytorch_img)
    pytorch_probs = F.softmax(pytorch_logits, dim=1)
    print(f'Pytroch Prediction: Class: {torch.argmax(pytorch_probs)}, Prob: {torch.max(pytorch_probs)}')

    madry_logits = madry_model(madry_img) # tuple - (logits, img)
    madry_probs = F.softmax(madry_logits[0], dim=1)
    print(f'Madry Prediction: Class: {torch.argmax(madry_probs)}, Prob: {torch.max(madry_probs)}')

    # Computing gradients
    if args.if_pre_post == 1:
        pytorch_logits[:, targ_class].backward()
    else:
        pytorch_probs[:, targ_class].backward()
    pytorch_grad = pytorch_img.grad.cpu().numpy()[0]
    pytorch_grad = np.rollaxis(pytorch_grad, 0, 3)
    pytorch_grad = np.mean(pytorch_grad, axis=-1)

    if args.if_pre_post == 1:
        madry_logits[0][:, targ_class].backward()
        softmax = 'pre'
    else:
        # ipdb.set_trace()
        madry_probs[:, targ_class].backward()
        softmax = 'post'
    madry_grad = madry_img.grad.cpu().numpy()[0]
    madry_grad = np.rollaxis(madry_grad, 0, 3)
    madry_grad = np.mean(madry_grad, axis=-1)

    ipdb.set_trace()
    ############################################
    # Saving the heatmaps
    out_dir = os.path.join(args.out_path, img_path[0].split('/')[-1].split('.')[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # ipdb.set_trace()

    orig_img = madry_img.cpu().detach().numpy()[0]
    orig_img = np.rollaxis(orig_img, 0, 3)
    grid = [[orig_img, pytorch_grad, madry_grad]]
    col_labels = ['Orig Image', 'ResNet_Grad', 'Madry_ResNet_Grad']
    img_shape = orig_img.shape[0]

    ## For the orig image
    text = []
    text.append(("%.3f" % torch.max(madry_probs).cpu().item(),  # Madry prob (pL)
                 "%3d" % torch.argmax(madry_probs).cpu().item(),  # Madry Label (pL)
                 "%.3f" % torch.max(pytorch_probs).cpu().item(),  # pytorch_prob (pL)
                 "%3d" % torch.argmax(pytorch_probs).cpu().item(),  # Pytorch Label (pL)
                 "%3d" % targ_class,  # label for given neuron (cNL)
                 ))

    madryProb, madryLabel, pytorchProb, pytorchLabel, trueLabel = zip(*text)

    row_labels_left = [(f'Madry:\n{madryLabel[i]}: {madryProb[i]}\n',
                        f'Pytorch:\n{pytorchLabel[i]}: {pytorchProb[i]}\n',
                        f'Target: {trueLabel}')
                       for i in range(len(madryProb))]

    row_labels_right = []

    zero_out_plot_multiple_patch(grid,
                                 out_dir,
                                 row_labels_left,
                                 row_labels_right,
                                 col_labels,
                                 file_name=f'Grad_heatmap_softmax_{softmax}_time_{f_time}.png',
                                 dpi=img_shape,
                                 )


    # ipdb.set_trace()
    aa = 1







