import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import ipdb

import sys, glob, cv2
from skimage.util import view_as_windows
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

from utils import *

# ipdb.set_trace()
# sys.path.append(os.path.abspath('../'))
# import utils as eutils

import warnings
warnings.filterwarnings("ignore")

import argparse

use_cuda = torch.cuda.is_available()

## For reproducebility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Input paramters for meaningful perturbation explanation of the image')

    parser.add_argument('-idp', '--img_dir_path', help='Path of the image directory', metavar='DIR')

    parser.add_argument('-op', '--out_path',
                        help='Path of the output directory where you want to save the results (Default is ./img_name/)')

    parser.add_argument('-gpu', '--gpu', type=int, choices=range(8),
                        help='GPU index', default=0,
                        )

    parser.add_argument('-num_im', '--num_imgs', type=int,
                        help='Number of images to be analysed. Max 50K. Default=1', default=1,
                        )

    parser.add_argument('-np_s', '--np_seed', type=int,
                        help='Numpy seed for selecting random images.', default=None,
                        )

    parser.add_argument('-s_idx', '--start_idx', type=int,
                        help='Start index for selecting images. Default: 0', default=0,
                        )

    parser.add_argument('-e_idx', '--end_idx', type=int,
                        help='End index for selecting images. Default: 2K', default=2000,
                        )

    parser.add_argument('--idx_flag', type=int,
                        help=f'Flag whether to use some images in the folder (1) or all (0). '
                             f'This is just for testing purposes. '
                             f'Default=1', default=0,
                        )

    # parser.add_argument('-ifp', '--if_pre', type=int, choices=range(2),
    #                     help='It is clear from name. Default: Post (0)', default=0,
    #                     )

    parser.add_argument('-ops', '--occ_patch_size', type=int,
                        help='Patch size for occlusion. Default=5', default=50,
                        )

    parser.add_argument('-os', '--occ_stride', type=int,
                        help='Stride for occlusion. Default=1', default=5,
                        )

    parser.add_argument('-obs', '--occ_batch_size', type=int,
                        help='Batch Size for occlusion. Default=64', default=100,
                        )

    parser.add_argument('-if_sp', '--if_save_plot', type=int, choices=range(2),
                        help='Whether save the plots. Default: No (0)', default=0,
                        )

    parser.add_argument('-if_sn', '--if_save_npy', type=int, choices=range(2),
                        help='Whether save the plots. Default: Yes (1)', default=1,
                        )

    # Parse the arguments
    args = parser.parse_args()

    args.if_pre = 0

    if args.np_seed is not None:
        print('Setting the numpy seed ')
        np.random.seed(args.np_seed)

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
    def __init__(self, data_path, transform, img_idxs=[0], idx_flag=1):
        self.data_path = data_path
        self.transform = transform

        self.img_filenames = []
        for file in glob.glob(os.path.join(data_path, "*.JPEG")):
            self.img_filenames.append(file)
        self.img_filenames.sort()

        if idx_flag == 1:
            print('Using the provided img_idxs')
            img_idxs=[0]
            self.img_filenames = [self.img_filenames[i] for i in img_idxs]

    def __getitem__(self, index):
        # ipdb.set_trace()
        img = Image.open(os.path.join(self.data_path, self.img_filenames[index])).convert('RGB')
        y = self.get_image_class(os.path.join(self.data_path, self.img_filenames[index]))
        img = img.convert('RGB')

        img = self.transform(img)
        return img, y, os.path.join(self.data_path, self.img_filenames[index])

    def __len__(self):
        return len(self.img_filenames)

    def get_image_class(self, filepath):
        # base_dir = '/home/naman/CS231n/heatmap_tests/'
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

def load_data(img_dir, preprocessFn, batch_size=1, img_idxs=[0], idx_flag=1):

    data = DataProcessing(img_dir, preprocessFn,
                          img_idxs=img_idxs, idx_flag=idx_flag)
    test_loader = torch.utils.data.DataLoader(data, batch_size=1)
    return test_loader, len(data)


class occlusion_analysis:
    def __init__(self, image, net, num_classes=1000, img_size=224, batch_size=64,
                 org_shape=(224, 224)):
        self.image = image
        self.model = net
        self.num_classes = num_classes
        self.img_size = img_size
        self.org_shape = org_shape
        self.batch_size = batch_size

    def explain(self, neuron, trainloader, stride=50, patch_size=5):
        input_shape = self.image.shape[1:]
        total_dim = np.prod(input_shape)

        # Compute original output
        org_softmax = self.model(self.image)
        eval0 = org_softmax.data[0, neuron]

        batch_heatmap = torch.Tensor().to('cuda')
        for i, data in enumerate(trainloader):
            data = data.to('cuda')
            softmax_out = self.model(data * self.image)
            amax, aind = softmax_out.max(dim=1)
            gt_val = softmax_out.data[:, neuron]
            delta = eval0 - gt_val

            batch_heatmap = torch.cat((batch_heatmap, delta))

        sqrt_shape = int(np.sqrt(batch_heatmap.shape[0]))
        attribution = np.reshape(batch_heatmap.cpu().numpy(), (sqrt_shape, sqrt_shape))
        return attribution

def create_occlusion_loader(args):
    batch_size = int((224 - args.occ_patch_size) / args.occ_stride) + 1
    # Create all occlusion masks initially to save time
    input_shape = (3, 224, 224)
    total_dim = np.prod(input_shape)
    index_matrix = np.arange(total_dim).reshape(input_shape)
    idx_patches = view_as_windows(index_matrix,
                                  (3, args.occ_patch_size, args.occ_patch_size),
                                  args.occ_stride).reshape((-1,) + \
                                                           (3, args.occ_patch_size, args.occ_patch_size))
    # Start perturbation loop
    batch_mask = torch.Tensor().to('cuda')
    for i, p in enumerate(idx_patches):
        mask = torch.ones(input_shape).reshape(-1).to('cuda')
        mask[p.reshape(-1)] = 0  # occ_val
        batch_mask = torch.cat((batch_mask, mask.reshape((1,) + input_shape)))
        del mask
        torch.cuda.empty_cache()
    occ_loader = torch.utils.data.DataLoader(batch_mask.cpu(), batch_size=args.occ_batch_size, shuffle=False,
                                             num_workers=0)
    del batch_mask
    torch.cuda.empty_cache()
    return occ_loader


if __name__ == '__main__':
    s_time = time.time()
    f_time = ''.join(str(s_time).split('.'))
    args = get_arguments()

    im_label_map = imagenet_label_mappings()

    if args.if_pre == 1:
        softmax = 'pre'
    else:
        softmax = 'post'

    ############################################
    ## #Indices for images
    pytorch_preprocessFn = transforms.Compose([transforms.Resize((256, 256)),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

    madry_preprocessFn = transforms.Compose([transforms.Resize((256, 256)),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             ])
    img_idxs = np.random.choice(50000, args.num_imgs, replace=False).tolist()
    img_idxs = sorted(img_idxs)
    pytorch_data_loader, img_count = load_data(args.img_dir_path, pytorch_preprocessFn,
                                       img_idxs=img_idxs, idx_flag=args.idx_flag)
    madry_data_loader, img_count = load_data(args.img_dir_path, madry_preprocessFn,
                                               img_idxs=img_idxs, idx_flag=args.idx_flag)

    ############################
    model_names = []
    model_names.append('pytorch')
    model_names.append('googlenet')
    model_names.append('madry')

    data_loader_dict = {'pytorch': pytorch_data_loader, 'madry': madry_data_loader, 'googlenet': pytorch_data_loader}
    load_model_fns = {'pytorch': eval('load_orig_imagenet_model'),
                      'madry': eval('load_madry_model'),
                      'googlenet': eval('load_orig_imagenet_model')}
    im_sz_dict = {'pytorch': 224, 'madry': 224, 'googlenet': 224}
    load_model_args = {'pytorch': 'resnet50', 'madry': 'madry', 'googlenet': 'googlenet'}

    heatmaps = {'pytorch': 0, 'madry': 0, 'googlenet': 0}
    probs_dict = {'pytorch': 0, 'madry': 0, 'googlenet': 0}
    ############################

    ## #Creating Occlusion masks
    o_time = time.time()
    print(f'Creating occlusion mask loader')
    occ_loader = create_occlusion_loader(args)
    print(f'Created in {time.time() - o_time}')

    for idx, model_name in enumerate(model_names):
        print(f'\nAnalyzing for model: {model_name}')
        load_model = load_model_fns[model_name]
        model_arg = load_model_args[model_name]
        data_loader = data_loader_dict[model_name]
        im_sz = im_sz_dict[model_name]

        ## Load Model
        print(f'Loading model {model_arg}')
        model = load_model(arch=model_arg, if_pre=0)  # Returns probs

        for i, (img, targ_class, img_path) in enumerate(data_loader):
            batch_time = time.time()
            ############################################
            # Saving the heatmaps
            out_dir = os.path.join(args.out_path, img_path[0].split('/')[-1].split('.')[0])
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            ########################################

            print(f'Analysing batch: {i}')
            targ_class = targ_class.cpu()
            if use_cuda:
                img = img.cuda(args.gpu)

            # Prob
            if args.if_pre == 1:
                print('Pre softmax analysis')
                logits = model(img)
                probs = F.softmax(logits, dim=1).cpu()
                logits = logits.cpu()
                softmax = 'pre'

            else:
                print('Post softmax analysis')
                probs = model(img).cpu()
                softmax = 'post'

            with torch.no_grad():
                analysis_obj = occlusion_analysis(img, model, img_size=im_sz,
                                                        num_classes=1000, batch_size=args.occ_batch_size)
                heatmap = analysis_obj.explain(neuron = targ_class, trainloader=occ_loader,
                                               stride=args.occ_stride,
                                               patch_size=args.occ_patch_size,
                                               )

                name = f'model_name_{model_name}_' \
                       f'patch_size_{args.occ_patch_size:03d}_' \
                       f'stride_{args.occ_stride:02d}_' \
                       f'softmax_{softmax}'

                if args.if_save_npy == 1:
                    np.save(os.path.join(out_dir, f'heatmap_time_{f_time}_{name}.npy'),
                            heatmap)
            print(f'Batch time is {time.time() - batch_time}')
    #######################
    print(f'Saving the results in {args.out_path}')
    print(f'Time string is {f_time}')
    print(f'Total time taken is {time.time() - s_time}')







########################################################################################################################
## #My Backup
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms

import sys, argparse, warnings, os, time, ipdb
from skimage.util import view_as_windows
import numpy as np
import utils as eutils

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
## For reproducebility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Input paramters for meaningful perturbation explanation of the image')

    parser.add_argument('-idp', '--img_dir_path', help='Path of the image directory', metavar='DIR')

    parser.add_argument('-op', '--out_path',
                        help='Path of the output directory where you want to save the results (Default is ./img_name/)')

    # parser.add_argument('-gpu', '--gpu', type=int, choices=range(8),
    #                     help='GPU index', default=0,
    #                     )

    # parser.add_argument('-num_im', '--num_imgs', type=int,
    #                     help='Number of images to be analysed. Max 50K. Default=1', default=1,
    #                     )

    parser.add_argument('-np_s', '--np_seed', type=int,
                        help='Numpy seed for selecting random images.', default=None,
                        )

    parser.add_argument('-s_idx', '--start_idx', type=int,
                        help='Start index for selecting images. Default: 0', default=0,
                        )

    parser.add_argument('-e_idx', '--end_idx', type=int,
                        help='End index for selecting images. Default: 2K', default=2000,
                        )

    parser.add_argument('--idx_flag', type=int,
                        help=f'Flag whether to use some images in the folder (1) or all (0). '
                             f'This is just for testing purposes. '
                             f'Default=1', default=0,
                        )

    # parser.add_argument('-ifp', '--if_pre', type=int, choices=range(2),
    #                     help='It is clear from name. Default: Post (0)', default=0,
    #                     )

    parser.add_argument('-ops', '--occ_patch_size', type=int,
                        help='Patch size for occlusion. Default=5', default=50,
                        )

    parser.add_argument('-os', '--occ_stride', type=int,
                        help='Stride for occlusion. Default=1', default=3,
                        )

    parser.add_argument('-obs', '--occ_batch_size', type=int,
                        help='Batch Size for occlusion. Default=64', default=100,  # 700,
                        )

    parser.add_argument('-if_sp', '--if_save_plot', type=int, choices=range(2),
                        help='Whether save the plots. Default: No (0)', default=0,
                        )

    parser.add_argument('-if_sn', '--if_save_npy', type=int, choices=range(2),
                        help='Whether save the plots. Default: Yes (1)', default=1,
                        )

    parser.add_argument('-if_n', '--if_noise', type=int, choices=range(2),
                        help='Whether to add noise to the image or not. Default: 0', default=0,
                        )

    parser.add_argument('-n_mean', '--noise_mean', type=float,
                        help='Mean of gaussian noise. Default: 0', default=0,
                        )

    parser.add_argument('-n_var', '--noise_var', type=float,
                        help='Variance of gaussian noise. Default: 0.1', default=0.1,
                        )

    # Parse the arguments
    args = parser.parse_args()

    args.if_pre = 0  # post-softmax
    args.if_noise = 0  # No noise
    args.gpu = 0  # GPU 0, use CUDA VISIBLE DEVICES to set gpu

    if args.np_seed is not None:
        print('Setting the numpy seed ')
        np.random.seed(args.np_seed)

    if args.occ_patch_size > 224:  # Img_size=224
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


########################################################################################################################
class occlusion_analysis:
    def __init__(self, image, net, num_classes=1000, img_size=224, batch_size=64,
                 org_shape=(224, 224)):
        self.image = image
        self.model = net
        self.num_classes = num_classes
        self.img_size = img_size
        self.org_shape = org_shape
        self.batch_size = batch_size

    def explain(self, neuron, trainloader, batch_size):
        out_total_dim = len(trainloader.dataset)
        # input_shape = self.image.shape[1:]

        # Compute original output
        org_softmax = self.model(self.image)
        eval0 = org_softmax.data[0, neuron]

        batch_heatmap = torch.zeros((out_total_dim), device='cuda')

        l_time = time.time()
        for i, data in enumerate(trainloader):
            print(f'TIme taken to sample data: {time.time() - l_time}')
            # data = data.to('cuda')
            # print(f'Batch size is {data.shape[0]}')

            softmax_out = self.model(data * self.image)
            gt_val = softmax_out.data[:, neuron]
            batch_heatmap[i * batch_size:(i + 1) * batch_size] = eval0 - gt_val
            l_time = time.time()

        sqrt_shape = int(np.sqrt(out_total_dim))
        attribution = batch_heatmap.data.reshape(sqrt_shape, sqrt_shape).cpu().numpy()
        # attribution = np.reshape(batch_heatmap.detach().cpu().numpy(), (sqrt_shape, sqrt_shape))
        del data
        del batch_heatmap
        torch.cuda.empty_cache()
        return attribution


########################################################################################################################
def create_occlusion_loader(args):
    # Create all occlusion masks initially to save time
    input_shape = (3, 224, 224)
    total_dim = np.prod(input_shape)
    index_matrix = np.arange(total_dim).reshape(input_shape)
    idx_patches = view_as_windows(index_matrix,
                                  (3, args.occ_patch_size, args.occ_patch_size),
                                  args.occ_stride).reshape((-1,) + \
                                                           (3, args.occ_patch_size, args.occ_patch_size))
    # Start perturbation loop
    batch_mask = torch.zeros(((idx_patches.shape[0],) + input_shape), device='cuda')
    for i, p in enumerate(idx_patches):
        mask = torch.ones(total_dim, device='cuda')
        mask[p.reshape(-1)] = 0  # occ_val
        batch_mask[i] = mask.reshape(input_shape)

    occ_loader = torch.utils.data.DataLoader(batch_mask, #.cpu(),
                                             batch_size=args.occ_batch_size, shuffle=False,
                                             num_workers=0)

    del batch_mask
    del mask
    torch.cuda.empty_cache()
    return occ_loader

#
# ########################################################################################################################
# if __name__ == '__main__':
#     s_time = time.time()
#     f_time = ''.join(str(s_time).split('.'))
#     args = get_arguments()
#
#     im_label_map = eutils.imagenet_label_mappings()
#
#     ############################################
#     ## #Indices for images
#     pytorch_preprocessFn = transforms.Compose([transforms.Resize((256, 256)),
#                                                transforms.CenterCrop(224),
#                                                transforms.ToTensor(),
#                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                                     std=[0.229, 0.224, 0.225])])
#
#     madry_preprocessFn = transforms.Compose([transforms.Resize((256, 256)),
#                                              transforms.CenterCrop(224),
#                                              transforms.ToTensor(),
#                                              ])
#     pytorch_data_loader, img_count = eutils.load_data(args.img_dir_path, pytorch_preprocessFn, batch_size=1,
#                                                       img_idxs=[args.start_idx, args.end_idx],
#                                                       idx_flag=args.idx_flag, args=args)
#     madry_data_loader, img_count = eutils.load_data(args.img_dir_path, madry_preprocessFn, batch_size=1,
#                                                     img_idxs=[args.start_idx, args.end_idx],
#                                                     idx_flag=args.idx_flag, args=args)
#
#     ############################
#     model_names = []
#     model_names.append('pytorch')
#     model_names.append('googlenet')
#     model_names.append('madry')
#
#     data_loader_dict = {'pytorch': pytorch_data_loader, 'madry': madry_data_loader, 'googlenet': pytorch_data_loader}
#     load_model_fns = {'pytorch': eval('eutils.load_orig_imagenet_model'),
#                       'madry': eval('eutils.load_madry_model'),
#                       'googlenet': eval('eutils.load_orig_imagenet_model')}
#     im_sz_dict = {'pytorch': 224, 'madry': 224, 'googlenet': 224}
#     load_model_args = {'pytorch': 'resnet50', 'madry': 'madry', 'googlenet': 'googlenet'}
#
#     # heatmaps = {'pytorch': 0, 'madry': 0, 'googlenet': 0}
#     # probs_dict = {'pytorch': 0, 'madry': 0, 'googlenet': 0}
#     par_name = f'sI_{args.start_idx:04d}_eI_{args.end_idx:04d}_' \
#                f'patch_size_{args.occ_patch_size:03d}_' \
#                f'stride_{args.occ_stride:02d}_' \
#                f'ifN_{args.if_noise}'
#     ############################
#
#     ## #Creating Occlusion masks
#     o_time = time.time()
#     print(f'Creating occlusion mask loader')
#     occ_loader = create_occlusion_loader(args)
#     print(f'Created in {time.time() - o_time}')
#
#     for idx, model_name in enumerate(model_names):
#         print(f'\nAnalyzing for model: {model_name}')
#         load_model = load_model_fns[model_name]
#         model_arg = load_model_args[model_name]
#         data_loader = data_loader_dict[model_name]
#         im_sz = im_sz_dict[model_name]
#
#         ## Load Model
#         print(f'Loading model {model_arg}')
#         model = load_model(arch=model_arg, if_pre=0)  # Returns probs
#
#         for i, (img, targ_class, img_path) in enumerate(data_loader):
#
#             print(f'Analysing batch: {i}')
#             ############################################
#             img_path = img_path[0]
#             img_name = img_path.split('/')[-1].split('.')[0]
#             print(f'Image Name is {img_name}')
#             # Saving the heatmaps
#             out_dir = os.path.join(args.out_path, img_name)
#             if args.if_save_npy == 1:
#                 eutils.mkdir_p(out_dir)
#                 print(f'Saving in {out_dir}')
#             ########################################
#             targ_class = targ_class.cpu()
#             if use_cuda:
#                 img = img.to('cuda')
#
#             batch_time = time.time()
#             with torch.no_grad():
#                 analysis_obj = occlusion_analysis(img, model, img_size=im_sz,
#                                                   num_classes=1000, batch_size=args.occ_batch_size)
#                 heatmap = analysis_obj.explain(neuron=targ_class.item(),
#                                                trainloader=occ_loader,
#                                                batch_size=args.occ_batch_size,
#                                                )
#
#                 print(f'Heatmap shape is {heatmap.shape}')
#
#                 if args.if_save_npy == 1:
#                     np.save(os.path.join(out_dir, f'time_{f_time}_heatmap_'
#                                                   f'img_name_{img_name}_'
#                                                   f'{par_name}_model_name_'
#                                                   f'{model_name}.npy'),
#                             heatmap)
#
#             print(f'Batch time is {time.time() - batch_time}')
#     #######################
#     print(f'Saving the results in {args.out_path}')
#     print(f'Time string is {f_time}')
#     print(f'Total time taken is {time.time() - s_time}')
#
# ########################################################################################################################










