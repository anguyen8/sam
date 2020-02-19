import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import ipdb

import sys, glob
import numpy as np
from PIL import Image
import ipdb
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

    parser.add_argument('-ops', '--occ_patch_size', type=int,
                        help='Patch size for occlusion. Default=5', default=5,
                        )

    parser.add_argument('-os', '--occ_stride', type=int,
                        help='Stride for occlusion. Default=1', default=1,
                        )

    # Parse the arguments
    args = parser.parse_args()
    # ipdb.set_trace()
    if args.occ_patch_size > 224: #Img_size=224
        print('Patch size can not be greater than image size.\nExiting')
        sys.exit(1)

    if args.occ_stride > args.occ_patch_size:
        print('Please provide stride lower than the patch size for better res.\nExiting')
        sys.exit(1)

    if args.img_dir_path is None:
        print('Please provide image dir path. Exiting')
        sys.exit(1)
    args.img_dir_path = os.path.abspath(args.img_dir_path)

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
    pytorch_preprocessFn = transforms.Compose([transforms.Resize((256, 256)),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

    madry_preprocessFn = transforms.Compose([transforms.Resize((256, 256)),
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

    # Load model
    model_kwargs = {
        'arch': 'resnet50',
        'dataset': dataset,
        'resume_path': f'./models/{DATA}.pt'
    }

    model_kwargs['state_dict_path'] = 'model'
    model, _ = model_utils.make_and_restore_model(**model_kwargs)
    model.eval()

    if use_cuda:
        model.cuda(dev_idx)

    # logits = model(im)
    # # ipdb.set_trace()
    # probs = F.softmax(logits[0], dim=1)
    #
    # print(f'Prediction by Madry Method: Class: {torch.argmax(probs)}, Prob: {torch.max(probs)}')
    # # ipdb.set_trace()


    return model #, int(im_path[0].split('/')[-1].split('.')[0].split('_')[-1]), im

def load_orig_imagenet_model(dev_idx): #resnet50
    model = models.resnet50(pretrained=True)
    model.eval()
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

class occlusion_analysis:
    def __init__(self, image, model, model_name='ImageNet', num_classes=1000, img_size=224, if_pre=0):
        self.image = image
        self.model = model
        self.num_classes = num_classes
        self.img_size = img_size
        self.if_pre = if_pre
        self.model_name = model_name

    def _batch_generator(self, sampled_data, curr_x, curr_y, patch_size, occ_val=0):
        sampled_size = sampled_data.shape[0]
        half_patch_size = patch_size//2

        mask_size = end_idx - start_idx
        mask = np.ones((mask_size, mask_size))
        np.fill_diagonal(mask, 0) #This is a in-place operation
        sampled_data[:, start_idx:end_idx] = np.multiply(mask, sampled_data[:, start_idx:end_idx]) + occ_val*np.identity(mask_size)
        return sampled_data

    def explain(self, neuron, batch_size=8, occ_val=0, patch_size=5, stride=1):

        image = self.image
        image.requires_grad = False
        model = self.model
        clamped_class = neuron
        num_classes = self.num_classes
        img_size = self.img_size
        if_pre = self.if_pre
        model_name = self.model_name

        if if_pre == 0:
            if model_name == 'ImageNet':
                orig_pred_prob = F.softmax(model(image), dim=1)[0, neuron]
            else: #Madri Model
                orig_pred_prob = F.softmax(model(image), dim=1)[0][0, neuron]
        else:
            if model_name == 'ImageNet':
                orig_pred_prob = model(image)[0, neuron]
            else:  # Madri Model
                orig_pred_prob = model(image)[0][0, neuron]
        print(f'Original predicted prob for the clamped class {neuron} is {orig_pred_prob}')
        num_features = img_size * img_size

        half_patch_size = patch_size//2
        final_size = (img_size - patch_size)//stride + 1 # No padding

        curr_x = 0 + half_patch_size
        curr_y = 0 + half_patch_size


        current_idx = 0
        num_rel_calc = 0
        total_num_rel_to_calc = final_size * final_size

        relevances = np.zeros((final_size, final_size))

        while curr_x < img_size and curr_y < img_size:
            tmp = image.clone() #[1, 3, 224, 224]

            sampled_size = min(batch_size, (total_num_rel_to_calc - num_rel_calc)) #5
            end_idx = current_idx + sampled_size

            sampled_data = tmp.expand(sampled_size, -1, -1, -1)  # [5, 3, 224, 224]
            sampled_data = self._batch_generator(sampled_data, curr_x, curr_y, patch_size, occ_val=occ_val)

            pred_probs = model.predict(sampled_data, batch_size=sampled_size)
            del sampled_data
            relevances[0, current_idx:end_idx] = orig_pred_prob - pred_probs[:, clamped_class]

            current_idx = end_idx

        return relevances

if __name__ == '__main__':
    f_time = ''.join(str(time.time()).split('.'))
    args = get_arguments()

    # ipdb.set_trace()

    # madry_model, img_number_madry, img_by_madry = load_madry_model(args.gpu)
    madry_model = load_madry_model(args.gpu)
    pytorch_model = load_orig_imagenet_model(args.gpu)

    #########################################################
    #data
    data_loader, img_count = load_data(args.img_dir_path, batch_size=1, which_number=args.which_number)
    data_iterator = enumerate(data_loader)

    _, (pytorch_img, madry_img, targ_class, img_path) = next(data_iterator) #orig_img (PIL)

    targ_class = targ_class.cpu().item()
    if use_cuda:
        pytorch_img = pytorch_img.cuda(args.gpu)
        madry_img = madry_img.cuda(args.gpu)

    ##Since we want to compute gradients as well
    pytorch_img = Variable(pytorch_img, requires_grad=True)
    madry_img = Variable(madry_img, requires_grad=True)


    print(f'Orig class is {targ_class}')
    print(f'Orig image path is {img_path}')

    pytorch_logits = pytorch_model(pytorch_img)
    pytorch_probs = F.softmax(pytorch_logits, dim=1)
    print(f'Pytroch Prediction: Class: {torch.argmax(pytorch_probs)}, Prob: {torch.max(pytorch_probs)}')

    madry_logits = madry_model(madry_img) # tuple - (logits, img)
    madry_probs = F.softmax(madry_logits, dim=1)
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
        madry_logits[:, targ_class].backward()
        softmax = 'pre'
    else:
        # ipdb.set_trace()
        madry_probs[:, targ_class].backward()
        softmax = 'post'
    madry_grad = madry_img.grad.cpu().numpy()[0]
    madry_grad = np.rollaxis(madry_grad, 0, 3)
    madry_grad = np.mean(madry_grad, axis=-1)

    ############################################
    # Saving the heatmaps
    out_dir = os.path.join(args.out_path, img_path[0].split('/')[-1].split('.')[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


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







