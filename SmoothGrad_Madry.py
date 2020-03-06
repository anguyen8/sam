################################################################################################################
## To make our implementation fater, we generate all the noisy samples on gpu and directly pass the entire batch of samples thorugh the model.
# So if you set a huge value, say 500, you might face memory issue.
## If you want huge value, you would have to use dataparalle (CUDA_VISIBLE_DEVICES=0,1,2,3..) will work since parallel=True (by default)
################################################################################################################

from __future__ import print_function
import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import ipdb, skimage

import sys, glob, ipdb, time, os, cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from srblib import abs_path

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

from termcolor import colored
# from robustness import model_utils, datasets
from user_constants import DATA_PATH_DICT

import utils as eutils
import settings

import warnings

warnings.filterwarnings("ignore")

import argparse

use_cuda = torch.cuda.is_available()
text_file = abs_path(settings.paper_img_txt_file)
# text_file = f'/home/naman/CS231n/heatmap_tests/' \
#             f'Madri/Madri_New/robustness_applications/img_name_files/' \
#             f'time_15669152608009198_seed_0_' \
#             f'common_correct_imgs_model_names_madry_ressnet50_googlenet.txt'
img_name_list = []
with open(text_file, 'r') as f:
    for line in f:
        img_name_list.append(line.split('\n')[0])

## For reproducebility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if use_cuda else "cpu")


def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Input paramters for meaningful perturbation explanation of the image')

    parser.add_argument('-idp', '--img_dir_path', help='Path to the input image dir', metavar='DIR')

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

    parser.add_argument('-n_var', '--noise_var', type=float,
                        help='Variance of gaussian noise. Default: 0.1', default=0.1,
                        )

    parser.add_argument('-n_seed', '--noise_seed', type=int,
                        help='Seed for the Gaussian noise. Default: 0', default=0,
                        )

    parser.add_argument('-if_n', '--if_noise', type=int, choices=range(2),
                        help='Whether to add noise to the image or not. Default: 0', default=0,
                        )

    parser.add_argument('-s_idx', '--start_idx', type=int,
                        help='Start index for selecting images. Default: 0', default=0,
                        )

    parser.add_argument('-e_idx', '--end_idx', type=int,
                        help='End index for selecting images. Default: 1735', default=1735,
                        )

    parser.add_argument('--idx_flag', type=int,
                        help=f'Flag whether to use some images in the folder (1) or all (0). '
                             f'This is just for testing purposes. '
                             f'Default=0', default=0,
                        )

    parser.add_argument('-n_sam', '--num_samples', type=int,
                        help=f'Size of the number of samples for SmoothGrad.'
                             f'Given our GPU memory, max value can be 100.'
                             f'Has to be positive integer.'
                             f' Default: 50',
                        default=[50], nargs='+',
                        )

    parser.add_argument('-std', '--stdev_spread', type=float,
                        help='Standard deviation spread to be used by SmoothGrad algo. Default: 0.15',
                        default=0.15,
                        )

    parser.add_argument('-if_sp', '--if_save_plot', type=int, choices=range(2),
                        help='Whether save the plots. Default: No (0)', default=0,
                        )

    parser.add_argument('-if_sn', '--if_save_npy', type=int, choices=range(2),
                        help='Whether save the plots. Default: Yes (1)', default=1,
                        )

    # Parse the arguments
    args = parser.parse_args()

    if not min(args.num_samples) > 0:
        print('\nnum_samples has to be a positive integer.\nExiting')
        sys.exit(0)

    if args.noise_seed is not None:
        print(f'Setting the numpy seed with value: {args.noise_seed}')
        np.random.seed(args.noise_seed)

    if args.img_dir_path is None:
        print('Please provide path to image dir. Exiting')
        sys.exit(1)
    else:
        args.img_dir_path = os.path.abspath(args.img_dir_path)

    if args.out_path is None:
        args.out_path = './'
    args.out_path = os.path.abspath(args.out_path)

    return args


########################################################################################################################
if __name__ == '__main__':
    s_time = time.time()
    f_time = ''.join(str(s_time).split('.'))
    args = get_arguments()
    args.noise_mean = 0  ##Explicity set to zero

    im_label_map = eutils.imagenet_label_mappings()

    if args.if_pre == 1:
        softmax = 'pre'
    else:
        softmax = 'post'

    ############################################
    ## #Data Loader
    pytorch_preprocessFn = transforms.Compose([transforms.Resize((256, 256)),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

    madry_preprocessFn = transforms.Compose([transforms.Resize((256, 256)),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             ])

    pytorch_data_loader, img_count = eutils.load_data(args.img_dir_path, pytorch_preprocessFn, batch_size=1,
                                                      img_idxs=[args.start_idx, args.end_idx],
                                                      idx_flag=args.idx_flag, args=args)
    madry_data_loader, img_count = eutils.load_data(args.img_dir_path, madry_preprocessFn, batch_size=1,
                                                    img_idxs=[args.start_idx, args.end_idx],
                                                    idx_flag=args.idx_flag, args=args)

    print(f'Total number of images to be analyzed are {img_count}')

    ############################
    model_names = []
    model_names.append('pytorch')
    model_names.append('googlenet')
    model_names.append('madry') #Robust_ResNet
    model_names.append('madry_googlenet')  # Robust GoogleNet

    print(f'Model is {model_names}')

    my_attacker = True
    parallel = True
    if my_attacker:
        data_loader_dict = {'pytorch': pytorch_data_loader,
                            'madry': pytorch_data_loader,
                            'madry_googlenet': pytorch_data_loader,
                            'googlenet': pytorch_data_loader}
    else:
        data_loader_dict = {'pytorch': pytorch_data_loader,
                            'madry': madry_data_loader,
                            'madry_googlenet': madry_data_loader,
                            'googlenet': pytorch_data_loader}

    load_model_fns = {'pytorch': eval('eutils.load_orig_imagenet_model'),
                      'madry': eval('eutils.load_madry_model'),
                      'madry_googlenet': eval('eutils.load_madry_model'),
                      'googlenet': eval('eutils.load_orig_imagenet_model')}

    im_sz_dict = {'pytorch': 224,
                  'madry': 224,
                  'madry_googlenet': 224,
                  'googlenet': 224}
    load_model_args = {'pytorch': 'resnet50',
                       'madry': 'madry',
                       'madry_googlenet': 'madry_googlenet',
                       'googlenet': 'googlenet'}

    ############################
    for idx, model_name in enumerate(model_names):
        print(f'\nAnalyzing for model: {model_name}')
        load_model = load_model_fns[model_name]
        model_arg = load_model_args[model_name]
        data_loader = data_loader_dict[model_name]
        im_sz = im_sz_dict[model_name]

        ## Load Model
        print(f'Loading model {model_arg}')
        model = load_model(arch=model_arg, if_pre=args.if_pre, my_attacker=my_attacker, parallel=parallel)  # Returns logits

        par_name = f'stdev_spread_{args.stdev_spread:.2f}_softmax_{softmax}_' \
                   f'idx_flag_{args.idx_flag}_start_idx_{args.start_idx}_' \
                   f'end_idx_{args.end_idx}_seed_{args.noise_seed}_' \
                   f'if_noise_{args.if_noise}_noise_mean_{args.noise_mean}_' \
                   f'noise_var_{args.noise_var}_model_name_{model_name}'
        print(f'Par name is - {par_name}')

        for i, (img, targ_class, img_path) in enumerate(data_loader):
            batch_time = time.time()
            print(f'Analysing batch: {i} of size {len(targ_class)}')

            ## Creating the save path
            img_name = img_path[0].split('/')[-1].split('.')[0]
            # out_dir = os.path.join(args.out_path, f'SmoothGrad_{model_name}/{img_name}')
            out_dir = os.path.join(args.out_path, f'{img_name}')
            eutils.mkdir_p(out_dir)
            print(f'Saving results in {out_dir}')

            targ_class = targ_class.cpu()
            sz = len(targ_class)
            if use_cuda:
                img = img.cuda()

            if args.if_pre == 1:
                orig_probs = F.softmax(model(img), dim=1).cpu()
            else:
                orig_probs = model(img).cpu()

            pred_prob = orig_probs[0, targ_class.item()].item()

            max_samples = max(args.num_samples)
            ## Noise for SmoothGrad
            stdev = ((torch.max(img) - torch.min(img)) * args.stdev_spread).item()
            noised_img_batch = img + torch.from_numpy(np.random.normal(0, stdev,
                                                                       (max_samples, 3, 224, 224))
                                                      ).float().to(device)
            ## #We want to compute gradients
            noised_img_batch = Variable(noised_img_batch, requires_grad=True)

            ## #Prob and gradients
            repeated_targ_class = torch.cat(max_samples * [targ_class])
            sel_nodes_shape = repeated_targ_class.shape
            ones = torch.ones(sel_nodes_shape)
            if use_cuda:
                ones = ones.cuda()

            if args.if_pre == 1:
                print('Pre softmax analysis')
                logits = model(noised_img_batch)
                probs = F.softmax(logits, dim=1).cpu()
                sel_nodes = logits[torch.arange(len(repeated_targ_class)), repeated_targ_class]
                sel_nodes.backward(ones)
                logits = logits.cpu()

            else:
                print('Post softmax analysis')
                probs = model(noised_img_batch)
                sel_nodes = probs[torch.arange(len(repeated_targ_class)), repeated_targ_class]
                sel_nodes.backward(ones)
                probs = probs.cpu()

            orig_grad = noised_img_batch.grad.cpu().numpy()  # [50, 3, 224, 224]
            orig_grad = np.rollaxis(orig_grad, 1, 4)  # [50, 224, 224, 3]

            for samples in args.num_samples:
                print(f'No. of samples are {samples}')
                grad = np.mean(orig_grad[:samples], axis=0) #Since it is SmoothGrad, you get mean across sample dimesnion
                grad = np.mean(grad, axis=-1) #Mean across channel dimension for plotting

                if args.if_save_npy == 1:
                    np.save(os.path.join(out_dir,
                                         f'sg_prob_{pred_prob:.3f}_'
                                         f'num_samples_{samples}_{par_name}_model_name_{model_name}.npy'),
                            grad)
                    # np.save(os.path.join(out_dir,
                    #                      f'time_{f_time}_{img_name}_prob_{pred_prob:.3f}_'
                    #                      f'heatmaps_num_samples_{samples}_{par_name}.npy'),
                    #         grad)

                ## Only saving the Madry results
                if args.if_save_plot == 1:
                    if model_name == 'madry':
                        orig_img = img.cpu().detach().numpy()[0]
                        orig_img = np.rollaxis(orig_img, 0, 3)

                        grid = []
                        grid.append([orig_img, grad])
                        col_labels = ['Orig Image', 'Madry_SmoothGrad']
                        row_labels_left = []
                        row_labels_right = []

                        eutils.zero_out_plot_multiple_patch(grid,
                                                            out_dir,
                                                            row_labels_left,
                                                            row_labels_right,
                                                            col_labels,
                                                            file_name=f'time_{f_time}_{img_name}_'
                                                                      f'heatmaps_num_samples_{samples}_'
                                                                      f'{par_name}.jpeg',
                                                            dpi=224,
                                                            )
            print(f'Time taken for batch is {time.time() - batch_time}\n')

    ##########################################
    print(f'Time stamp is {f_time}')
    print(f'Time taken is {time.time() - s_time}')
########################################################################################################################
