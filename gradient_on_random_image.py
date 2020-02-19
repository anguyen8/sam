import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import ipdb, skimage

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

    parser.add_argument('-ip', '--img_path', help='Path to the input image', metavar='IMG_FILE')

    parser.add_argument('-op', '--out_path',
                        help='Path of the output directory where you want to save the results (Default is ./img_name/)')

    parser.add_argument('-gpu', '--gpu', type=int, choices=range(8),
                        help='GPU index', default=0,
                        )


    parser.add_argument('-ifp', '--if_pre', type=int, choices=range(2),
                        help='It is clear from name. Default: Pre (1)', default=1,
                        )

    parser.add_argument('-n_mean', '--noise_mean', type=float,
                        help='Mean of gaussian noise. Default: 0', default=0,
                        )

    parser.add_argument('-n_seed', '--noise_seed', type=int,
                        help='Seed for the Gaussian noise. Default: 0', default=0,
                        )

    parser.add_argument('-n_var', '--noise_var', type=float,
                        help='Variance of gaussian noise. Default: 0.1', default=0.1,
                        )

    # Parse the arguments
    args = parser.parse_args()

    if args.img_path is None:
        print('Please provide path to image file. Exiting')
        sys.exit(1)

    if args.out_path is None:
        args.out_path = './'
    args.out_path = os.path.abspath(args.out_path)

    return args


def load_madry_model(dev_idx, if_pre=0):
    DATA = 'ImageNet'  # Choices: ['CIFAR', 'ImageNet', 'RestrictedImageNet']
    dataset_function = getattr(datasets, DATA)
    dataset = dataset_function(DATA_PATH_DICT[DATA])
    # Load model
    model_kwargs = {
        'arch': 'resnet50',
        'dataset': dataset,
        'resume_path': f'./models/{DATA}.pt',
        'parallel': False,
    }
    model_kwargs['state_dict_path'] = 'model'
    model, _ = model_utils.make_and_restore_model(**model_kwargs)
    if if_pre == 1:
        pass
    else:
        model = nn.Sequential(model, nn.Softmax(dim=1))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if use_cuda:
        model.cuda(dev_idx)
    return model

def load_orig_imagenet_model(dev_idx, arch='resnet50', if_pre=0): #resnet50
    if arch == 'googlenet':
        model = models.googlenet(pretrained=True)
    else:
        model = models.resnet50(pretrained=True)

    if if_pre == 1:
        pass
    else:
        model = nn.Sequential(model, nn.Softmax(dim=1))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if use_cuda:
        model.cuda(dev_idx)
    return model

## Plotting for zero_out (InpxGrad)
def zero_out_plot_multiple_patch(grid,
                  folderName,
                  row_labels_left,
                  row_labels_right,
                  col_labels,
                  file_name=None,
                  dpi=224,
                  ):

    plt.rcParams.update({'font.size': 5})
    plt.rc("font", family="sans-serif")

    plt.rc("axes.spines", top=True, right=True, left=True, bottom=True)
    image_size = (grid[0][0]).shape[0]

    nRows = len(grid)
    nCols = len(grid[0])


    tRows = nRows + 2  # total rows
    tCols = nCols + 1  # total cols

    wFig = tCols
    hFig = tRows  # Figure height (one more than nRows becasue I want to add xlabels to the top of figure)

    fig, axes = plt.subplots(nrows=tRows, ncols=tCols, figsize=(wFig, hFig))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    axes = np.reshape(axes, (tRows, tCols))

    #########
    ## Creating colormap
    uP = cm.get_cmap('Reds', 129)
    dowN = cm.get_cmap('Blues_r', 128)
    newcolors = np.vstack((
        dowN(np.linspace(0, 1, 128)),
        uP(np.linspace(0, 1, 129))
    ))
    cMap = ListedColormap(newcolors, name='RedBlues')
    cMap.colors[257//2, :] = [1, 1, 1, 1]
    #######

    scale = 0.80
    fontsize=5

    for r in range(tRows):
        # if r <= 1:
        for c in range(tCols):
            ax = axes[r][c]
            l, b, w, h = ax.get_position().bounds
            ax.set_position([l, b, w * scale, h * scale])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])


            if r > 0 and c > 0 and r < tRows - 1:
                img_data = grid[r - 1][c - 1]
                abs_min = np.amin(img_data)
                abs_max = np.amax(img_data)
                abs_mx = max(np.abs(abs_min), np.abs(abs_max))
                r_abs_min = round(np.amin(img_data), 2)
                r_abs_max = round(np.amax(img_data), 2)
                r_abs_mx = round(max(np.abs(abs_min), np.abs(abs_max)), 2)

                # Orig Image
                if c == 1:
                    im = ax.imshow(img_data, interpolation='none')

                else:
                    im = ax.imshow(img_data, interpolation='none', cmap=cMap, vmin=-abs_mx, vmax=abs_mx)

                zero = 0
                if not r - 1:
                    if col_labels != []:
                        # ipdb.set_trace()
                        ax.set_title(col_labels[c - 1] + '\n' + f'max: {str(r_abs_max)}, min: {str(r_abs_min)}',
                                     horizontalalignment='center',
                                     verticalalignment='bottom',
                                     fontsize=fontsize, pad=5)

                if c == tCols - 2:

                    if row_labels_right != []:
                        txt_right = [l + '\n' for l in row_labels_right[r - 1]]
                        ax2 = ax.twinx()
                        ax2.set_xticks([])
                        ax2.set_yticks([])
                        ax2.spines['top'].set_visible(False)
                        ax2.spines['right'].set_visible(False)
                        ax2.spines['bottom'].set_visible(False)
                        ax2.spines['left'].set_visible(False)
                        ax2.set_ylabel(''.join(txt_right), rotation=0,
                                       verticalalignment='center',
                                       horizontalalignment='left',
                                       fontsize=fontsize)

                if not c - 1:

                    if row_labels_left != []:
                        txt_left = [l + '\n' for l in row_labels_left[r - 1]]
                        ax.set_ylabel(''.join(txt_left),
                                      rotation=0,
                                      verticalalignment='center',
                                      horizontalalignment='right',
                                      fontsize=fontsize)

                # else:
                if c > 1: #!= 1:
                    w_cbar = 0.005
                    h_cbar = h * scale
                    b_cbar = b
                    l_cbar = l + scale * w + 0.001
                    cbaxes = fig.add_axes([l_cbar, b_cbar, w_cbar, h_cbar])
                    cbar = fig.colorbar(im, cax=cbaxes)
                    cbar.outline.set_visible(False)
                    cbar.ax.tick_params(labelsize=4, width=0.2, length=1.2, direction='inout', pad=0.5)
                    tt = abs_mx
                    cbar.set_ticks([-tt, zero, tt])
                    cbar.set_ticklabels([-r_abs_mx, zero, r_abs_mx])

        #####################################################################################

    dir_path = folderName
    print(f'Saving figure to {os.path.join(dir_path, file_name)}')

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.savefig(os.path.join(dir_path, file_name), orientation='landscape', dpi=dpi / scale, transparent=True, frameon=False)
    plt.close(fig)

def imagenet_label_mappings():
    with open("imagenet_label_mapping") as f:
        image_label_mapping = {int(x.split(":")[0]): x.split(":")[1].strip()
                               for x in f.readlines() if len(x.strip()) > 0}
        return image_label_mapping


# resize and take the center part of image to what our model expects
def pytorch_get_input_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    return transf

def madry_get_input_transform():
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    return transf

def get_input_tensors(img, model_name='madry'):
    if model_name == 'madry':
        transf = madry_get_input_transform()
    else:
        transf = pytorch_get_input_transform()
    # unsqeeze converts single image to batch of 1
    return transf(img).unsqueeze(0)

def get_image_class(filepath):
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

def get_probs_from_img_if_pre_true(im, model, if_pre):
    assert if_pre == 1
    logits = model(im)
    probs = F.softmax(logits, dim=1).cpu()
    logits = logits.cpu()
    return logits, probs

def get_normalized_grads(im):
    grad = im.grad.cpu().numpy()[0]
    grad = np.rollaxis(grad, 0, 3)
    grad = np.mean(grad, axis=-1)
    return grad

if __name__ == '__main__':
    s_time = time.time()
    f_time = ''.join(str(s_time).split('.'))
    args = get_arguments()

    im_label_map = imagenet_label_mappings()

    madry_model = load_madry_model(args.gpu, if_pre=args.if_pre)
    pytorch_model = load_orig_imagenet_model(args.gpu, arch='resnet50', if_pre=args.if_pre)
    gNet_model = load_orig_imagenet_model(args.gpu, arch='googlenet', if_pre=args.if_pre)

    #########################################################
    #data
    orig_img = Image.open(os.path.abspath(args.img_path)).convert('RGB')
    targ_class = get_image_class(os.path.abspath(args.img_path))
    img = np.asarray(orig_img) # numpy image, dtype=uint8, range (0-255)
    # ipdb.set_trace()
    mean = args.noise_mean
    var = args.noise_var

    print(f'Orig class is {targ_class}')
    print(f'Orig image path is {os.path.abspath(args.img_path)}')

    # noise_idx = 0
    # while True:
    print(f'Trying to add noise to the image.')
    # ipdb.set_trace()
    noisy_img = skimage.util.random_noise(img.copy(),
                                          mode='gaussian',
                                          mean=mean,
                                          var=var,
                                          seed=args.noise_seed) #numpy, dtype=float64,range (0, 1)
    # noise_idx += 1
    noisy_img = Image.fromarray(np.uint8(noisy_img*255))


    ## Preprocess the image for madry and resnet50
    madry_img_orig = get_input_tensors(orig_img, model_name='madry')
    pytorch_img_orig = get_input_tensors(orig_img, model_name='pytorch')
    gNet_img_orig = get_input_tensors(orig_img, model_name='pytorch')

    madry_img_noisy = get_input_tensors(noisy_img, model_name='madry')
    pytorch_img_noisy = get_input_tensors(noisy_img, model_name='pytorch')
    gNet_img_noisy = get_input_tensors(noisy_img, model_name='pytorch')

    if use_cuda:
        pytorch_img_orig = pytorch_img_orig.cuda(args.gpu)
        gNet_img_orig = gNet_img_orig.cuda(args.gpu)
        madry_img_orig = madry_img_orig.cuda(args.gpu)

        pytorch_img_noisy = pytorch_img_noisy.cuda(args.gpu)
        gNet_img_noisy = gNet_img_noisy.cuda(args.gpu)
        madry_img_noisy = madry_img_noisy.cuda(args.gpu)

    ##Since we want to compute gradients as well
    pytorch_img_orig = Variable(pytorch_img_orig, requires_grad=True)
    pytorch_img_noisy = Variable(pytorch_img_noisy, requires_grad=True)
    gNet_img_orig = Variable(gNet_img_orig, requires_grad=True)
    gNet_img_noisy = Variable(gNet_img_noisy, requires_grad=True)
    madry_img_orig = Variable(madry_img_orig, requires_grad=True)
    madry_img_noisy = Variable(madry_img_noisy, requires_grad=True)

    # Prob
    if args.if_pre == 1:
        print('Pre softmax analysis')

        pytorch_logits_orig, pytorch_probs_orig = get_probs_from_img_if_pre_true(pytorch_img_orig, pytorch_model, args.if_pre)
        pytorch_logits_noisy, pytorch_probs_noisy = get_probs_from_img_if_pre_true(pytorch_img_noisy, pytorch_model, args.if_pre)

        gNet_logits_orig, gNet_probs_orig = get_probs_from_img_if_pre_true(gNet_img_orig, gNet_model, args.if_pre)
        gNet_logits_noisy, gNet_probs_noisy = get_probs_from_img_if_pre_true(gNet_img_noisy, gNet_model, args.if_pre)

        madry_logits_orig, madry_probs_orig = get_probs_from_img_if_pre_true(madry_img_orig, madry_model, args.if_pre)
        madry_logits_noisy, madry_probs_noisy = get_probs_from_img_if_pre_true(madry_img_noisy, madry_model, args.if_pre)

        softmax = 'pre'

    else:
        print('Post softmax analysis')
        pytorch_probs_orig = pytorch_model(pytorch_img_orig).cpu()
        pytorch_probs_noisy = pytorch_model(pytorch_img_noisy).cpu()

        gNet_probs_orig = gNet_model(gNet_img_orig).cpu()
        gNet_probs_noisy = gNet_model(gNet_img_noisy).cpu()

        madry_probs_orig = madry_model(madry_img_orig).cpu()
        madry_probs_noisy = madry_model(madry_img_noisy).cpu()

        softmax = 'post'

    pytorch_pred_label_orig = torch.argmax(pytorch_probs_orig).cpu().item()
    pytorch_pred_prob_orig = torch.max(pytorch_probs_orig).cpu().item()
    print(f'Pytroch Prediction for Orig Image: Class: {pytorch_pred_label_orig}, '
          f'Prob: {pytorch_pred_prob_orig}')

    pytorch_pred_label_noisy = torch.argmax(pytorch_probs_noisy).cpu().item()
    pytorch_pred_prob_noisy = torch.max(pytorch_probs_noisy).cpu().item()
    print(f'Pytroch Prediction for Noisy Image: Class: {pytorch_pred_label_noisy}, '
          f'Prob: {pytorch_pred_prob_noisy}')

    gNet_pred_label_orig = torch.argmax(gNet_probs_orig).cpu().item()
    gNet_pred_prob_orig = torch.max(gNet_probs_orig).cpu().item()
    print(f'GoogleNet Prediction for Orig Image: Class: {gNet_pred_label_orig}, '
          f'Prob: {gNet_pred_prob_orig}')

    gNet_pred_label_noisy = torch.argmax(gNet_probs_noisy).cpu().item()
    gNet_pred_prob_noisy = torch.max(gNet_probs_noisy).cpu().item()
    print(f'GoogleNet Prediction for Noisy Image: Class: {gNet_pred_label_noisy}, '
          f'Prob: {gNet_pred_prob_noisy}')

    madry_pred_label_orig = torch.argmax(madry_probs_orig).cpu().item()
    madry_pred_prob_orig = torch.max(madry_probs_orig).cpu().item()
    print(f'Madry Prediction for Orig Image: Class: {madry_pred_label_orig}, '
          f'Prob: {madry_pred_prob_orig}')

    madry_pred_label_noisy = torch.argmax(madry_probs_noisy).cpu().item()
    madry_pred_prob_noisy = torch.max(madry_probs_noisy).cpu().item()
    print(f'Madry Prediction for Noisy Image: Class: {madry_pred_label_noisy}, '
          f'Prob: {madry_pred_prob_noisy}')

    ## #TODO: Uncomment this depending on use case
    # if madry_pred_label_orig == targ_class and pytorch_pred_label_orig == targ_class and gNet_pred_label_orig == targ_class:
    #     pass
    # else:
    #     print('Predicted labels of the image for Madry model, ResNet50 and GoogleNet does not match the ture label')
    #     print('Please provide a different image.\nExiting')
    #     sys.exit(1)
    ###################


    # if targ_class == madry_pred_label_noisy and targ_class == pytorch_pred_label_noisy and targ_class == gNet_pred_label_noisy:
    #     break

    print(f'Seed provided to generate the Gaussian noise is {args.noise_seed}')
    # Computing gradients
    if args.if_pre == 1:
        pytorch_logits_orig[:, targ_class].backward()
        pytorch_logits_noisy[:, targ_class].backward()

        gNet_logits_orig[:, targ_class].backward()
        gNet_logits_noisy[:, targ_class].backward()

        madry_logits_orig[:, targ_class].backward()
        madry_logits_noisy[:, targ_class].backward()
    else:
        pytorch_probs_orig[:, targ_class].backward()
        pytorch_probs_noisy[:, targ_class].backward()

        gNet_probs_orig[:, targ_class].backward()
        gNet_probs_noisy[:, targ_class].backward()

        madry_probs_orig[:, targ_class].backward()
        madry_probs_noisy[:, targ_class].backward()

    pytorch_grad_orig = get_normalized_grads(pytorch_img_orig)
    pytorch_grad_noisy = get_normalized_grads(pytorch_img_noisy)

    gNet_grad_orig = get_normalized_grads(gNet_img_orig)
    gNet_grad_noisy = get_normalized_grads(gNet_img_noisy)

    madry_grad_orig = get_normalized_grads(madry_img_orig)
    madry_grad_noisy = get_normalized_grads(madry_img_noisy)

    ############################################
    # Saving the heatmaps
    out_dir = os.path.join(args.out_path, os.path.abspath(args.img_path).split('/')[-1].split('.')[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # ipdb.set_trace()

    orig_img = madry_img_orig.cpu().detach().numpy()[0]
    orig_img = np.rollaxis(orig_img, 0, 3)

    noisy_img = madry_img_noisy.cpu().detach().numpy()[0]
    noisy_img = np.rollaxis(noisy_img, 0, 3)

    grid = []
    grid.append([orig_img, gNet_grad_orig, pytorch_grad_orig, madry_grad_orig])
    grid.append([noisy_img, gNet_grad_noisy, pytorch_grad_noisy, madry_grad_noisy])
    col_labels = ['Orig Image', 'GoogleNet', 'ResNet_Grad', 'Madry_ResNet_Grad']
    img_shape = orig_img.shape[0]

    ## For the orig image
    text = []
    text.append(("%.3f" % madry_pred_prob_orig,  # Madry prob (pL)
                 "%3d" % madry_pred_label_orig,  # Madry Label (pL)
                 "%.3f" % pytorch_pred_prob_orig,  # pytorch_prob (pL)
                 "%3d" % pytorch_pred_label_orig,  # Pytorch Label (pL)
                 "%.3f" % gNet_pred_prob_orig,  # pytorch_prob (pL)
                 "%3d" % gNet_pred_label_orig,  # Pytorch Label (pL)
                 "%3d" % targ_class,  # label for given neuron (cNL)
                 ))

    text.append(("%.3f" % madry_pred_prob_noisy,  # Madry prob (pL)
                 "%3d" % madry_pred_label_noisy,  # Madry Label (pL)
                 "%.3f" % pytorch_pred_prob_noisy,  # pytorch_prob (pL)
                 "%3d" % pytorch_pred_label_noisy,  # Pytorch Label (pL)
                 "%.3f" % gNet_pred_prob_noisy,  # pytorch_prob (pL)
                 "%3d" % gNet_pred_label_noisy,  # Pytorch Label (pL)
                 "%3d" % targ_class,  # label for given neuron (cNL)
                 ))

    madryProb, madryLabel, pytorchProb, pytorchLabel, gNetProb, gNetLabel, trueLabel = zip(*text)

    row_labels_left = [(f'Madry: Top-1:\n{im_label_map[int(madryLabel[i])]}: {madryProb[i]}\n',
                        f'ResNet: Top-1:\n{im_label_map[int(pytorchLabel[i])]}: {pytorchProb[i]}\n',
                        f'GoogleNet: Top-1:\n{im_label_map[int(gNetLabel[i])]}: {gNetProb[i]}\n',
                        f'Target:\n{im_label_map[int(trueLabel[i])]}')
                       for i in range(len(madryProb))]

    row_labels_right = []

    zero_out_plot_multiple_patch(grid,
                                 out_dir,
                                 row_labels_left,
                                 row_labels_right,
                                 col_labels,
                                 file_name=f'Grad_heatmap_noise_mean_{mean}_noise_var_{var}_noise_seed_{args.noise_seed}_'
                                           f'softmax_{softmax}_time_{f_time}.png',
                                 dpi=img_shape,
                                 )

    print(f'Time taken is {time.time() - s_time}')
    aa = 1







