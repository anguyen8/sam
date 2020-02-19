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
from RISE.explanations import RISEBatch, RISE
cudnn.benchmark = True

import warnings
warnings.filterwarnings("ignore")

import argparse

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Input paramters for meaningful perturbation explanation of the image')

    parser.add_argument('-idp', '--img_dir_path', help='Path of the image directory', metavar='DIR')

    parser.add_argument('-op', '--out_path',
                        help='Path of the output directory where you want to save the results (Default is ./img_name/)')

    parser.add_argument('-gpu', '--gpu', type=int, nargs='+',
                        help='GPU index', default=[0],
                        )

    parser.add_argument('-ifp', '--if_pre', type=int, choices=range(2),
                        help='It is clear from name. Default: Pre (1)', default=1,
                        )

    parser.add_argument('-ifs', '--if_save', type=int, choices=range(2),
                        help='Whether save the results. Default: Yes (1)', default=1,
                        )

    parser.add_argument('-np_s', '--np_seed', type=int,
                        help='Numpy seed for selecting random images. Default=0', default=0,
                        )

    parser.add_argument('-num_im', '--num_imgs', type=int,
                        help='Number of images to be analysed. Max 50K. Default=1', default=1,
                        )

    parser.add_argument('-bs', '--batch_size', type=int,
                        help='Batch size for data loader. \n The batch size should be larger than the number of GPUs used. Default=1',
                        default=1,
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

    parser.add_argument('-r_mn', '--rise_mask_num', type=int,
                       help='Number of random masks to be used by RISE', default=6000,
                       )

    parser.add_argument('-r_ms', '--rise_mask_size', type=int,
                        help='Mask size for RISE. Default=7', default=7,
                        )

    parser.add_argument('-r_np_s', '--rise_np_seed', type=int,
                        help='Seed for generating masks for RISE. Default=0', default=0,
                        )

    # Parse the arguments
    args = parser.parse_args()

    if args.rise_madry_mask_path is not None:
        args.rise_madry_mask_path = os.path.abspath(args.rise_madry_mask_path)

    if args.rise_pytorch_mask_path is not None:
        args.rise_pytorch_mask_path = os.path.abspath(args.rise_pytorch_mask_path)

    if args.img_dir_path is None:
        print('Please provide image dir path. Exiting')
        sys.exit(1)
    args.img_dir_path = os.path.abspath(args.img_dir_path)

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

        # #ipdb.set_trace()
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
    pytorch_preprocessFn = transforms.Compose([transforms.Resize((256, 256)),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

    madry_preprocessFn = transforms.Compose([transforms.Resize((256, 256)),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       ])

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

    model_kwargs['state_dict_path'] = 'model'
    model, _ = model_utils.make_and_restore_model(**model_kwargs)

    if args.if_pre == 1:
        pass
    else:
        model = nn.Sequential(model, nn.Softmax(dim=1))

    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=dev_idx).to(device)
    else:
        model.to(device)

    return model

def load_orig_imagenet_model(dev_idx): #resnet50
    model = models.resnet50(pretrained=True)
    if args.if_pre == 1:
        pass
    else:
        model = nn.Sequential(model, nn.Softmax(dim=1))

    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=dev_idx).to(device)
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


def imagenet_label_mappings():
    with open("imagenet_label_mapping") as f:
        image_label_mapping = {int(x.split(":")[0]): x.split(":")[1].strip()
                               for x in f.readlines() if len(x.strip()) > 0}
        return image_label_mapping


if __name__ == '__main__':
    s_time = time.time()
    f_time = ''.join(str(s_time).split('.'))
    args = get_arguments()

    input_size = (224, 224)

    im_label_map = imagenet_label_mappings()

    madry_model = load_madry_model(args.gpu)
    pytorch_model = load_orig_imagenet_model(args.gpu)

    #########################################################
    #data
    # ipdb.set_trace()
    np.random.seed(args.np_seed)
    img_idxs = np.random.choice(50000, args.num_imgs, replace=False).tolist()
    # img_idxs = np.random.choice(1000, args.num_imgs, replace=False).tolist()
    img_idxs = sorted(img_idxs)
    # ipdb.set_trace()

    data_loader, img_count = load_data(args.img_dir_path, batch_size=args.batch_size, img_idxs=img_idxs)

    ###################################################################
    #################################################
    # RISE analysis
    out_dir = os.path.join(args.out_path, f'RISE')
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)

    mask_num = args.rise_mask_num
    mask_size = args.rise_mask_size
    rise_seed = args.rise_np_seed

    ## #Madry
    madry_explainer = RISEBatch(madry_model, input_size)
    if args.rise_gen_new_mask == 1 or not os.path.isfile(args.rise_madry_mask_path):
        mask_name = f'madry_masks_mask_count_{mask_num}.npy'
        # print(f'Saving madry masks here: {os.path.join(out_dir, mask_name)}')
        madry_explainer.generate_masks(N=mask_num, s=mask_size, p1=0.1, seed=rise_seed,
                                       savepath=os.path.join(out_dir, mask_name), save=False)
    else:
        if str(mask_num) not in args.rise_madry_mask_path.split('/')[-1]:
            print('Incorrect path provided for given mask numbers. Please check.\nExiting')
            sys.exit(0)
        madry_explainer.load_masks(args.rise_madry_mask_path)
        print(f'Mask loaded from here: {args.rise_madry_mask_path}')

    ## #Pytorch
    pytorch_explainer = RISEBatch(pytorch_model, input_size)
    if args.rise_gen_new_mask == 1 or not os.path.isfile(args.rise_pytorch_mask_path):
        mask_name = f'pytorch_masks_mask_count_{mask_num}.npy'
        # print(f'Saving pytorch masks here: {os.path.join(out_dir, mask_name)}')
        pytorch_explainer.generate_masks(N=mask_num, s=mask_size, p1=0.1, seed=rise_seed,
                                         savepath=os.path.join(out_dir, mask_name), save=False)
    else:
        if str(mask_num) not in args.rise_pytorch_mask_path.split('/')[-1]:
            print('Incorrect path provided for given mask numbers. Please check.\nExiting')
            sys.exit(0)
        pytorch_explainer.load_masks(args.rise_pytorch_mask_path)
        print(f'Mask loaded from here: {args.rise_pytorch_mask_path}')

    ################################################################
    madry_correct = 0
    pytorch_correct  = 0

    for i, (pytorch_img, madry_img, targ_class, img_path) in enumerate(data_loader):

        print(f'Analysing batch: {i}')

        pytorch_img = pytorch_img.to(device)
        madry_img = madry_img.to(device)
        targ_class = targ_class.cpu()

        #Prob
        if args.if_pre == 1:
            print('Pre softmax analysis')
            pytorch_logits = pytorch_model(pytorch_img)
            pytorch_probs = F.softmax(pytorch_logits, dim=1).cpu()
            pytorch_logits = pytorch_logits.cpu()

            madry_logits = madry_model(madry_img)
            madry_probs = F.softmax(madry_logits, dim=1).cpu()
            madry_logits = madry_logits .cpu()
            softmax = 'pre'

        else:
            print('Post softmax analysis')
            pytorch_probs = pytorch_model(pytorch_img).cpu()
            madry_probs = madry_model(madry_img).cpu()
            softmax = 'post'

        # ipdb.set_trace()

        if pytorch_img.shape[0] == 1:
            madry_prediction = torch.argmax(madry_probs, dim=-1).cpu().item()
            pytorch_prediction = torch.argmax(pytorch_probs, dim=-1).cpu().item()
            true_class = targ_class.cpu().item()
            if madry_prediction == true_class:
                madry_correct += 1

            if pytorch_prediction == true_class:
                pytorch_correct += 1

            print(f'Madry Prediction: {madry_prediction}\nResNet Prediction {pytorch_prediction}')
            print(f'True class: {true_class}')

            if madry_prediction == true_class and pytorch_prediction == true_class:
                print(f'Condition satisfied for image: {i}. Analyzing')
            else:
                print(f'Condition did not satisfied for image: {i}. Trying for next image')
                continue

        #################################################
        # RISE analysis
        ## Madry
        madry_saliency_maps = madry_explainer(madry_img).detach().cpu().numpy()

        ## PyTorch
        pytorch_saliency_maps = pytorch_explainer(pytorch_img).detach().cpu().numpy()
        # #################################################

        madry_img = madry_img.cpu().detach().numpy()
        orig_img = np.rollaxis(madry_img, 1, 4)
#
#         ipdb.set_trace()
#
        if args.if_save==1:
            for idx in range(len(targ_class)):
                ############################################
                # Saving the heatmaps
                img_name = img_path[idx].split('/')[-1].split('.')[0]
                out_dir = os.path.join(args.out_path,
                                       f'rise_res/exp_RISE_mask_count_{mask_num}_mask_size_{mask_size:03d}_rise_seed_{rise_seed}_softmax_{softmax}_numpy_seed_{args.np_seed}_time_{f_time}/{img_name}')
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                t_cls = targ_class[idx].item()

                # ipdb.set_trace()
                grid = [[orig_img[idx], pytorch_saliency_maps[idx, t_cls], madry_saliency_maps[idx, t_cls]]]
                col_labels = ['Orig Image', 'RISE_on_ResNet', 'RISE_on_Madry_ResNet']
                img_shape = grid[0][0].shape[0]

                ## For the orig image
                text = []
                text.append(("%.3f" % torch.max(madry_probs[idx, :]).item(),  # Madry prob (pL)
                             "%3d" % torch.argmax(madry_probs[idx, :]).item(),  # Madry Label (pL)
                             "%.3f" % torch.max(pytorch_probs[idx, :]).item(),  # pytorch_prob (pL)
                             "%3d" % torch.argmax(pytorch_probs[idx, :]).item(),  # Pytorch Label (pL)
                             "%3d" % t_cls,  # label for given neuron (cNL)
                             ))

                madryProb, madryLabel, pytorchProb, pytorchLabel, trueLabel = zip(*text)

                row_labels_left = [(f'Madry: Top-1:\n{im_label_map[int(madryLabel[i])]}: {madryProb[i]}\n',
                                    f'ResNet: Top-1:\n{im_label_map[int(pytorchLabel[i])]}: {pytorchProb[i]}\n',
                                    f'Target:\n{im_label_map[int(trueLabel[i])]}')
                                   for i in range(len(madryProb))]

                row_labels_right = []

                zero_out_plot_multiple_patch(grid,
                                             out_dir,
                                             row_labels_left,
                                             row_labels_right,
                                             col_labels,
                                             file_name=f'RISE_heatmap_rise_mask_count_{mask_num}_mask_size_{mask_size:03d}_rise_seed_{rise_seed}_softmax_{softmax}_numpy_seed_{args.np_seed}_time_{f_time}.png',
                                             dpi=img_shape,
                                             )



########################################

    print(f'Time taken is {time.time() - s_time}')
    if pytorch_img.shape[0] == 1:
        print(f'Madry correct count is: {madry_correct}')
        print(f'ResNet correct count is: {pytorch_correct}')
    # ipdb.set_trace()
    aa = 1






