from __future__ import absolute_import
import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import torch.backends.cudnn as cudnn

import sys, glob
import numpy as np
from PIL import Image
import ipdb
import time
import os
from tqdm import tqdm


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
# import imageio
from termcolor import colored
from robustness import model_utils, datasets
from user_constants import DATA_PATH_DICT

## RISE Imports
# from RISE import utils as RISE_utils
from RISE.explanations import RISEBatch
cudnn.benchmark = True
from srblib import abs_path
import utils as eutils

# ipdb.set_trace()

import warnings
warnings.filterwarnings("ignore")

import argparse

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

my_attacker = False


def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Input paramters for meaningful perturbation explanation of the image')

    parser.add_argument('-idp', '--img_dir_path', help='Path of the image directory', metavar='DIR')

    parser.add_argument('-op', '--out_path',
                        help='Path of the output directory where you want to save the results (Default is ./img_name/)')

    parser.add_argument('-gpu', '--gpu', type=int, nargs='+',
                        help='GPU index', default=[0],
                        )

    # parser.add_argument('-wn', '--which_number', type=int,
    #                     help='Which numbered image to load from validation set (Max:50000). Default=1', default=1,
    #                     )

    parser.add_argument('-ifp', '--if_pre', type=int, choices=range(2),
                        help='It is clear from name. Default: Pre (1)', default=1,
                        )

    parser.add_argument('-ifs', '--if_save', type=int, choices=range(2),
                        help='Whether save the results. Default: Yes (1)', default=1,
                        )

    parser.add_argument('-ops', '--occ_patch_size', type=int,
                        help='Patch size for occlusion. Default=5', default=5,
                        )

    parser.add_argument('-os', '--occ_stride', type=int,
                        help='Stride for occlusion. Default=1', default=1,
                        )

    parser.add_argument('-np_s', '--np_seed', type=int,
                        help='Numpy seed for selecting random images. Default=0', default=0,
                        )

    parser.add_argument('-num_im', '--num_imgs', type=int,
                        help='Number of images to be analysed. Max 50K. Default=1', default=1,
                        )

    parser.add_argument('-bs', '--batch_size', type=int,
                        help='Batch size for data loader. \n The batch size should be larger than the number of GPUs used. Default=2', default=2,
                        )

    parser.add_argument('-r_gnm', '--rise_gen_new_mask', type=int, choices=range(2),
                        help='Gen new mask or load from dir. Default=True (1)', default=1,
                        )

    parser.add_argument('-r_mmp', '--rise_madry_mask_path',
                        help='Path to madrys mask to be used for RISE algo', default='None',
                        )

    parser.add_argument('-r_pmp', '--rise_pytorch_mask_path',
                        help='Path to pytorch mask to be used for RISE algo', default='None',
                        )

    # Parse the arguments
    args = parser.parse_args()

    # ipdb.set_trace()

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
    args.img_dir_path = abs_path(args.img_dir_path)

    if args.out_path is None:
        args.out_path = './'
    args.out_path = os.path.abspath(args.out_path)

    return args


class DataProcessing:
    def __init__(self, data_path, pytorch_transform, madry_transform, img_idxs=[0]):
        self.data_path = data_path
        self.pytorch_transform = pytorch_transform
        self.madry_transform = madry_transform

        self.img_filenames = []

        for file in glob.glob(os.path.join(data_path, "*.JPEG")):
            self.img_filenames.append(file)
        self.img_filenames.sort()

        self.img_filenames = [self.img_filenames[i] for i in img_idxs]

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

def load_data(img_dir, batch_size=8, img_idxs=[0]):

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

    if my_attacker:
        madry_preprocessFn = pytorch_preprocessFn

    data = DataProcessing(img_dir, pytorch_preprocessFn, madry_preprocessFn, img_idxs=img_idxs)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    return test_loader, len(data)

def load_madry_model(dev_idx):
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

    ## GoogleNet
    # Load model
    model_kwargs = {
        'arch': 'googlenet',
        'dataset': dataset,
        'resume_path': f'./models/ImageNet_GoogleNet.pt.best',
        'my_attacker': my_attacker,
        'parallel': False,
    }

    model_kwargs['state_dict_path'] = 'model'
    model, _ = model_utils.make_and_restore_model(**model_kwargs)
    model.eval()

    # if use_cuda:
    #     # model = model.cuda()
    #     # model.cuda(dev_idx)
    #
    #     # To use multiple GPUs
    #     model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()

    for p in model.parameters():
        p.requires_grad = False

    if torch.cuda.device_count() > 1:
        print('Using more than one gpus')
        # ipdb.set_trace()
        model = nn.DataParallel(model, device_ids=[0]).to(device) #, device_ids=[0])
    else:
        model.to(device)

    return model

def load_orig_imagenet_model(dev_idx): #resnet50
    model = models.resnet50(pretrained=True)
    model.eval()
    # if use_cuda:
    #     # model = model.cuda()
    #     # model.cuda(dev_idx)
    #
    #     # To use multiple GPUs
    #     model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()

    for p in model.parameters():
        p.requires_grad = False

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0]).to(device) #, device_ids=[0])
    else:
        model.to(device)
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
                abs_min = round(np.amin(img_data), 2)
                abs_max = round(np.amax(img_data), 2)
                abs_mx = round(max(np.abs(abs_min), np.abs(abs_max)), 2)

                # Orig Image
                if c == 1:
                    im = ax.imshow(img_data, interpolation='none')

                else:
                    im = ax.imshow(img_data, interpolation='none', cmap=cMap, vmin=-abs_mx, vmax=abs_mx)

                zero = 0
                if not r - 1:
                    if col_labels != []:
                        # ipdb.set_trace()
                        ax.set_title(col_labels[c - 1] + '\n' + f'max: {str(abs_max)}, min: {str(abs_min)}',
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

                    cbar.set_ticklabels([-tt, zero, tt])


        #####################################################################################

    dir_path = folderName
    print(f'Saving figure to {os.path.join(dir_path, file_name)}')

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.savefig(os.path.join(dir_path, file_name), orientation='landscape', dpi=dpi / scale, transparent=True, frameon=False)
    plt.close(fig)


if __name__ == '__main__':
    f_time = ''.join(str(time.time()).split('.'))
    args = get_arguments()

    input_size = (224, 224)
    # torch.cuda.set_device(args.gpu)

    # madry_model, img_number_madry, img_by_madry = load_madry_model(args.gpu)
    madry_model = load_madry_model(args.gpu)
    pytorch_model = load_orig_imagenet_model(args.gpu)

    #########################################################
    #data
    # ipdb.set_trace()
    np.random.seed(args.np_seed)
    img_idxs = np.random.randint(low=0, high=50000, size=(1, args.num_imgs))
    img_idxs = img_idxs.tolist()[0] #list of idxs

    data_loader, img_count = load_data(args.img_dir_path, batch_size=args.batch_size, img_idxs=img_idxs)

    # ###################################################################
    # #################################################
    # # RISE analysis
    # out_dir = os.path.join(args.out_path, f'RISE')
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    #
    # ## Madry
    # if args.if_pre == 1:
    #     pass
    # else:
    #     madry_model_softmax = nn.Sequential(madry_model, nn.Softmax(dim=1))
    #     madry_model_softmax = madry_model_softmax.eval()
    #     madry_model_softmax = madry_model_softmax.to(device)
    #
    # madry_explainer = RISEBatch(madry_model_softmax, input_size, 1)
    # if args.rise_gen_new_mask != 1 or not os.path.isfile(args.rise_madry_mask_path):
    #     mask_name = 'madry_masks.npy'
    #     print(f'Saving madry masks here: {os.path.join(out_dir, mask_name)}')
    #     ipdb.set_trace()
    #     madry_explainer.generate_masks(N=6000, s=8, p1=0.1, savepath=os.path.join(out_dir, mask_name))
    # else:
    #     madry_explainer.load_masks(args.rise_madry_mask_path)
    #     print(f'Mask loaded from here: {args.rise_madry_mask_path}')
    #
    # ## Pytorch
    # if args.if_pre == 1:
    #     pass
    # else:
    #     pytorch_model_softmax = nn.Sequential(pytorch_model, nn.Softmax(dim=1))
    #     pytorch_model_softmax = pytorch_model_softmax.eval()
    #     pytorch_model_softmax = pytorch_model_softmax.to(device)
    #
    # pytorch_explainer = RISEBatch(pytorch_model_softmax, input_size, 300)
    # if args.rise_gen_new_mask != 1 or not os.path.isfile(args.rise_pytorch_mask_path):
    #     mask_name = 'pytorch_masks.npy'
    #     print(f'Saving pytorch masks here: {os.path.join(out_dir, mask_name)}')
    #     pytorch_explainer.generate_masks(N=6000, s=8, p1=0.1, savepath=os.path.join(out_dir, mask_name))
    # else:
    #     pytorch_explainer.load_masks(args.rise_pytorch_mask_path)
    #     print(f'Mask loaded from here: {args.rise_pytorch_mask_path}')
    #
    # ################################################################


    for i, (pytorch_img, madry_img, targ_class, img_path) in enumerate(data_loader):

        print(f'Analysing batch: {i}')

        # if use_cuda:
        #     pytorch_img = pytorch_img.cuda()
        #     madry_img = madry_img.cuda()
        #     targ_class = targ_class.cuda()

        pytorch_img = pytorch_img.to(device)
        madry_img = madry_img.to(device)
        targ_class = targ_class.to(device)

        ##Since we want to compute gradients as well
        pytorch_img = Variable(pytorch_img, requires_grad=True)
        madry_img = Variable(madry_img, requires_grad=True)

        #Pytorch
        pytorch_logits = pytorch_model(pytorch_img)
        pytorch_probs = F.softmax(pytorch_logits, dim=1)
        # print(f'Pytroch Prediction: Class: {torch.argmax(pytorch_probs)}, Prob: {torch.max(pytorch_probs)}')

        #Madry
        # ipdb.set_trace()
        madry_logits = madry_model(madry_img)
        madry_probs = F.softmax(madry_logits, dim=1)
        # print(f'Madry Prediction: Class: {torch.argmax(madry_probs)}, Prob: {torch.max(madry_probs)}')

        # ipdb.set_trace()
        # Computing gradients
        if args.if_pre == 1:
            sel_nodes = pytorch_logits[torch.arange(len(targ_class)), targ_class]
            sel_nodes_shape = sel_nodes.shape
            ones = torch.ones(sel_nodes_shape)
            # if use_cuda:
            #     ones = ones.cuda()
            ones = ones.to(device)
            sel_nodes.backward(ones)

            sel_nodes_madry = madry_logits[torch.arange(len(targ_class)), targ_class]
            sel_nodes_madry.backward(ones)
            softmax = 'pre'
        else:
            sel_nodes = pytorch_probs[torch.arange(len(targ_class)), targ_class]
            sel_nodes_shape = sel_nodes.shape
            ones = torch.ones(sel_nodes_shape)
            # if use_cuda:
            #     ones = ones.cuda()
            ones = ones.to(device)
            sel_nodes.backward(ones)

            sel_nodes_madry = madry_probs[torch.arange(len(targ_class)), targ_class]
            sel_nodes_madry.backward(ones)
            softmax = 'post'

        pytorch_grad = pytorch_img.grad.cpu().numpy()
        pytorch_grad = np.rollaxis(pytorch_grad, 1, 4)
        pytorch_grad = np.mean(pytorch_grad, axis=-1)

        madry_grad = madry_img.grad.cpu().numpy() #[8, 3, 224, 224]
        madry_grad = np.rollaxis(madry_grad, 1, 4) #[8, 224, 224, 3]
        madry_grad = np.mean(madry_grad, axis=-1) #[8, 224, 224]

        ## Zero out gradients
        pytorch_img.grad.data.zero_()
        madry_img.grad.data.zero_()

        # #################################################
        # # RISE analysis
        #
        # ## Madry
        # madry_saliency_maps = madry_explainer(madry_img)
        #
        # ipdb.set_trace()
        #
        # ## PyTorch
        #
        # pytorch_saliency_maps = pytorch_explainer(pytorch_img)
        # #################################################

        madry_img = madry_img.cpu().detach().numpy()
        orig_img = np.rollaxis(madry_img, 1, 4)

        if args.if_save==1:
            for idx in range(len(targ_class)):
                ############################################
                # Saving the heatmaps
                img_name = img_path[idx].split('/')[-1].split('.')[0]
                # out_dir = os.path.join(args.out_path, f'{img_name}')
                out_dir = args.out_path
                # ipdb.set_trace()
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                grid = [[orig_img[idx], pytorch_grad[idx], madry_grad[idx]]]
                col_labels = ['Orig Image', 'GoogleNet_Grad', 'Madry_GoogleNet_Grad']
                img_shape = grid[0][0].shape[0]

                # ipdb.set_trace()

                ## For the orig image
                text = []
                text.append(("%.3f" % torch.max(madry_probs[idx, :]).cpu().item(),  # Madry prob (pL)
                             "%3d" % torch.argmax(madry_probs[idx, :]).cpu().item(),  # Madry Label (pL)
                             "%.3f" % torch.max(pytorch_probs[idx, :]).cpu().item(),  # pytorch_prob (pL)
                             "%3d" % torch.argmax(pytorch_probs[idx, :]).cpu().item(),  # Pytorch Label (pL)
                             "%3d" % targ_class[idx].cpu().item(),  # label for given neuron (cNL)
                             ))

                madryProb, madryLabel, pytorchProb, pytorchLabel, trueLabel = zip(*text)

                row_labels_left = [(f'Madry_GNet: Top-1:\n{madryLabel[i]}: {madryProb[i]}\n',
                                    f'GoogleNet: Top-1:\n{pytorchLabel[i]}: {pytorchProb[i]}\n',
                                    f'Target: {trueLabel}')
                                   for i in range(len(madryProb))]

                row_labels_right = []

                eutils.zero_out_plot_multiple_patch(grid,
                                                    out_dir,
                                                    row_labels_left,
                                                    row_labels_right,
                                                    col_labels,
                                                    file_name=f'time_{f_time}_Grad_{img_name}_softmax_{softmax}.png',
                                                    dpi=img_shape,
                                                    )

########################################







