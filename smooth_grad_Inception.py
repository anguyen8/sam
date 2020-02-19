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
from robustness import model_utils, datasets
from user_constants import DATA_PATH_DICT

import utils as eutils

import warnings

warnings.filterwarnings("ignore")

import argparse

use_cuda = torch.cuda.is_available()
text_file = f'/home/naman/CS231n/heatmap_tests/' \
            f'Madri/Madri_New/robustness_applications/img_name_files/' \
            f'time_15669152608009198_seed_0_' \
            f'common_correct_imgs_model_names_madry_ressnet50_googlenet.txt'
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
                        help='End index for selecting images. Default: 2K', default=2000,
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
class DataProcessing:
    def __init__(self, data_path, transform, img_idxs=[0, 1], idx_flag=1, args=None):
        self.data_path = data_path
        self.transform = transform
        self.args = args

        if data_path == abs_path('~/CS231n/heatmap_tests/images/ILSVRC2012_img_val/'):
            aa = img_name_list[img_idxs[0]:img_idxs[1]]
            self.img_filenames = [os.path.join(data_path, f'{ii}.JPEG') for ii in aa]
            self.img_filenames.sort()
        else:
            self.img_filenames = []
            for file in glob.glob(os.path.join(data_path, "*.JPEG")):
                self.img_filenames.append(file)
            self.img_filenames.sort()

        if idx_flag == 1:
            print('Only prodicing results for 1 image')
            img_idxs = [0]
            self.img_filenames = [self.img_filenames[i] for i in img_idxs]

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, self.img_filenames[index])).convert('RGB')
        y = self.get_image_class(os.path.join(self.data_path, self.img_filenames[index]))

        if self.args is not None:
            if self.args.if_noise == 1:
                img = skimage.util.random_noise(np.asarray(img), mode='gaussian',
                                                mean=self.args.noise_mean, var=self.args.noise_var,
                                                )  # numpy, dtype=float64,range (0, 1)
                img = Image.fromarray(np.uint8(img * 255))

        img = self.transform(img)
        return img, y, os.path.join(self.data_path, self.img_filenames[index])

    def __len__(self):
        return len(self.img_filenames)

    def get_image_class(self, filepath):
        base_dir = abs_path('~/CS231n/heatmap_tests/')

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


###########################
def load_data(img_dir, preprocessFn, batch_size=1, img_idxs=[0, 1], idx_flag=1, args=None):
    data = DataProcessing(img_dir, preprocessFn,
                          img_idxs=img_idxs, idx_flag=idx_flag, args=args)
    test_loader = torch.utils.data.DataLoader(data, batch_size=1)
    return test_loader, len(data)

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
    ## #Indices for images
    pytorch_preprocessFn = transforms.Compose([transforms.Resize((299, 299)),
                                               transforms.CenterCrop(299),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

#     madry_preprocessFn = transforms.Compose([transforms.Resize((256, 256)),
#                                              transforms.CenterCrop(224),
#                                              transforms.ToTensor(),
#                                              ])

    pytorch_data_loader, img_count = eutils.load_data(args.img_dir_path, pytorch_preprocessFn, batch_size=1,
                                                      img_idxs=[args.start_idx, args.end_idx],
                                                      idx_flag=args.idx_flag, args=args)
#     madry_data_loader, img_count = eutils.load_data(args.img_dir_path, madry_preprocessFn, batch_size=1,
#                                                     img_idxs=[args.start_idx, args.end_idx],
#                                                     idx_flag=args.idx_flag, args=args)
    madry_data_loader = []

    print(f'Total number of images to be analyzed are {img_count}')

    ############################
    model_names = []
    model_names.append('pytorch')
#     model_names.append('googlenet')
#     model_names.append('madry')

    data_loader_dict = {'pytorch': pytorch_data_loader, 'madry': madry_data_loader,
                        'googlenet': pytorch_data_loader}
    load_model_fns = {'pytorch': eval('eutils.load_orig_imagenet_model'),
                      'madry': eval('eutils.load_madry_model'),
                      'googlenet': eval('eutils.load_orig_imagenet_model')}
    im_sz_dict = {'pytorch': 224, 'madry': 224, 'googlenet': 224}
    load_model_args = {'pytorch': 'inception', 'madry': 'madry', 'googlenet': 'googlenet'}

    ############################
    for idx, model_name in enumerate(model_names):
        print(f'\nAnalyzing for model: {model_name}')
        load_model = load_model_fns[model_name]
        model_arg = load_model_args[model_name]
        data_loader = data_loader_dict[model_name]
        im_sz = im_sz_dict[model_name]

        ## Load Model
        print(f'Loading model {model_arg}')
        model = load_model(arch=model_arg, if_pre=args.if_pre)  # Returns logits

        par_name = f'stdev_spread_{args.stdev_spread}_softmax_{softmax}_' \
                   f'idx_flag_{args.idx_flag}_start_idx_{args.start_idx}_' \
                   f'end_idx_{args.end_idx}_seed_{args.noise_seed}_' \
                   f'if_noise_{args.if_noise}_noise_mean_{args.noise_mean}_' \
                   f'var_{args.noise_var}_model_name_{model_name}'
        print(f'Par name is - {par_name}')

        for i, (img, targ_class, img_path) in enumerate(data_loader):
            print(f'Analysing batch: {i} of size {len(targ_class)}')

            ## Creating the save path
            img_name = img_path[0].split('/')[-1].split('.')[0]
            out_dir = os.path.join(args.out_path, f'SmoothGrad_{model_name}/{img_name}')
            eutils.mkdir_p(out_dir)
            print(f'Saving results in {out_dir}')

            targ_class = targ_class.cpu()
            sz = len(targ_class)
            if use_cuda:
                img = img.cuda()

            ## Prediction
            if args.if_pre == 1:
                ps = F.softmax(model(img), dim=1).cpu()
            else:
                ps = model(img).cpu()
            ls = torch.argmax(ps)

            if ls.item() != targ_class.item():
                continue

            max_samples = max(args.num_samples)
            ## Noise for SmoothGrad
            stdev = ((torch.max(img) - torch.min(img)) * args.stdev_spread).item()
            noised_img_batch = img + torch.from_numpy(np.random.normal(0, stdev,
                                                                       (max_samples, 3, 299, 299))
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

            ##
            img = Variable(img, requires_grad=True)

            assert img.grad == None, 'Grad should be None in the beginning'
            ## Prediction and vanilla grad
            if args.if_pre == 1:
                logs = model(img)
                ps = F.softmax(logs, dim=1).cpu()
                logs[0, targ_class].backward()
                logs = logs.cpu()
            else:
                ps = model(img)
                ps[0, targ_class].backward()
                ps = ps.cpu()
            vanilla_grad = img.grad.cpu().numpy()
            vanilla_grad = np.rollaxis(vanilla_grad, 1, 4)  # [1, 224, 224, 3]
            vanilla_grad = np.mean(vanilla_grad[0], axis=-1)

            for samples in args.num_samples:
                print(f'No. of samples are {samples}')
                grad = np.mean(orig_grad[:samples], axis=0) #Since it is SmoothGrad, you get mean across sample dimesnion
                grad = np.mean(grad, axis=-1) #Mean across channel dimension for plotting

                if args.if_save_npy == 1:
                    np.save(os.path.join(out_dir,
                                         f'time_{f_time}_{img_name}_'
                                         f'heatmaps_num_samples_{samples}_{par_name}.npy'),
                            grad)

                ## Only saving the Madry results
                if args.if_save_plot == 1:
                    print('Saving the plots')
                    if model_name == 'pytorch':
                        orig_img = Image.open(img_path[0])
                        orig_img = orig_img.resize((299, 299))
                        orig_img = np.asarray(orig_img).astype(float)/255

                        grid = []
                        grid.append([orig_img, vanilla_grad, grad])
                        col_labels = ['Orig Image', 'Vanilla Grad', 'SmoothGrad']
                        row_labels_left = []
                        row_labels_right = []

                        eutils.zero_out_plot_multiple_patch(grid,
                                                            out_dir,
                                                            row_labels_left,
                                                            row_labels_right,
                                                            col_labels,
                                                            file_name=f'time_{f_time}_image_idx_{i:02d}_'
                                                                      f'image_name_{img_name}_'
                                                                      f'heatmaps_num_samples_{samples}_'
                                                                      f'{par_name}.jpeg',
                                                            dpi=299,
                                                            )

    ##########################################
    print(f'Time stamp is {f_time}')
    print(f'Time taken is {time.time() - s_time}')
########################################################################################################################
