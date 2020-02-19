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
from srblib import abs_path
import ipdb
import time
import os
from tqdm import tqdm
import utils as eutils


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

my_attacker = False
print(f'My Attacker is: {my_attacker}')

text_file = f'/home/naman/CS231n/heatmap_tests/' \
            f'Madri/Madri_New/robustness_applications/img_name_files/' \
            f'time_15669152608009198_seed_0_' \
            f'common_correct_imgs_model_names_madry_ressnet50_googlenet.txt'
img_name_list = []
with open(text_file, 'r') as f:
    for line in f:
        img_name_list.append(line.split('\n')[0])

end_idx = 2000


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

    parser.add_argument('-r_mn', '--rise_mask_num', type=int,
                       help='Number of random masks to be used by RISE', default=6000,
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

        if data_path == abs_path('~/CS231n/heatmap_tests/images/ILSVRC2012_img_val/'):
            self.img_filenames = []
            for file in glob.glob(os.path.join(data_path, "*.JPEG")):
                self.img_filenames.append(file)
            self.img_filenames.sort()
            # print(f'Taking first {end_idx} images from the selected list of images')
            # aa = img_name_list[0:end_idx]
            # self.img_filenames = [os.path.join(data_path, f'{ii}.JPEG') for ii in aa]

        else:
            self.img_filenames = []
            for file in glob.glob(os.path.join(data_path, "*.JPEG")):
                self.img_filenames.append(file)
            self.img_filenames.sort()

        # self.img_filenames = []
        # for file in glob.glob(os.path.join(data_path, "*.JPEG")):
        #     self.img_filenames.append(file)
        # self.img_filenames.sort()
        #
        # self.img_filenames = [self.img_filenames[i] for i in img_idxs]

        print(f'No. of images to be analyzed are {len(self.img_filenames)}')

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

    print(pytorch_preprocessFn)
    print(madry_preprocessFn)

    if my_attacker:
        madry_preprocessFn = pytorch_preprocessFn

    data = DataProcessing(img_dir, pytorch_preprocessFn, madry_preprocessFn, img_idxs=img_idxs)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers=8)
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
        'my_attacker': my_attacker,
        'parallel': False,
    }

    # ## GoogleNet
    # # Load model
    # model_kwargs = {
    #     'arch': 'googlenet',
    #     'dataset': dataset,
    #     'resume_path': f'./models/ImageNet_GoogleNet.pt.best',
    #     'my_attacker': my_attacker,
    #     'parallel': False,
    # }

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
        print('Using more than one gpus')
        model = nn.DataParallel(model, device_ids=dev_idx).to(device)
    else:
        model.to(device)

    return model

def load_orig_imagenet_model(dev_idx): #resnet50

    print(f'Loading ResNet')
    model = models.resnet50(pretrained=True)

    # print(f'Loading GoogleNet')
    # model = models.googlenet(pretrained=True)
    if args.if_pre == 1:
        pass
    else:
        model = nn.Sequential(model, nn.Softmax(dim=1))

    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    if torch.cuda.device_count() > 1:
        print('Using more than one gpus')
        model = nn.DataParallel(model, device_ids=dev_idx).to(device)
    else:
        model.to(device)

    return model



if __name__ == '__main__':
    s_time = time.time()
    f_time = ''.join(str(time.time()).split('.'))
    args = get_arguments()

    input_size = (224, 224)

    madry_model = load_madry_model(args.gpu)
    pytorch_model = load_orig_imagenet_model(args.gpu)
    print('Model Loaded')
    #########################################################
    #data
    np.random.seed(args.np_seed)
    img_idxs = np.random.choice(50000, args.num_imgs, replace=False).tolist()
    # img_idxs = np.random.choice(1000, args.num_imgs, replace=False).tolist()
    img_idxs = sorted(img_idxs)

    data_loader, tot_img_count = load_data(args.img_dir_path, batch_size=args.batch_size, img_idxs=img_idxs)
    print('Loader created')
    # ipdb.set_trace()

    # ipdb.set_trace()
    madry_correct = 0
    pytorch_correct  = 0

    top5_madry_correct = 0
    top5_pytorch_correct = 0

    img_count = 0
    incorrect_img_names = []
    for i, (pytorch_img, madry_img, targ_class, img_path) in enumerate(data_loader):

        print(f'Analysing batch: {i}')
        pytorch_img = pytorch_img.to(device)
        madry_img = madry_img.to(device)
        targ_class = targ_class.cpu().numpy()
        print('Images Loaded')
        img_names = [i.split('/')[-1].split('.')[0] for i in img_path]

        img_count += len(targ_class)

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

        ## Getting top 1 predictions
        madry_predicted_labels = torch.argmax(madry_probs.cpu(), dim=-1).numpy()
        pytorch_predicted_labels = torch.argmax(pytorch_probs.cpu(), dim=-1).numpy()

        madry_correct += np.size(np.where((targ_class - madry_predicted_labels) == 0))
        pytorch_correct += np.size(np.where((targ_class - pytorch_predicted_labels) == 0))


        ## Incorrect indices
        inc_idx = np.where((targ_class - madry_predicted_labels) != 0)[0]
        incorrect_img_names.extend([img_names[i] for i in inc_idx])


        ## Getting top 5 predictions
        _, madry_inds = torch.topk(madry_probs, 5, dim=-1)
        madry_inds = madry_inds.numpy()
        _, pytorch_inds = torch.topk(pytorch_probs, 5, dim=-1)
        pytorch_inds = pytorch_inds.numpy()

        top5_madry_correct += np.count_nonzero((madry_inds - np.expand_dims(targ_class, axis=-1)) == 0)
        top5_pytorch_correct += np.count_nonzero((pytorch_inds - np.expand_dims(targ_class, axis=-1)) == 0)

        # print(f'Number of correct images by madry is {madry_correct}/{img_count}')
        # print(f'Number of correct images by googlenet is {pytorch_correct}/{img_count}')

        print(f'Number of correct images by madry is {madry_correct}/{img_count}')
        print(f'Number of correct images by resnet is {pytorch_correct}/{img_count}')

        print(f'Top5 - Madry is: {top5_madry_correct}/{img_count}')
        print(f'Top5 - ResNet is: {top5_pytorch_correct}/{img_count}')

    print(f'Time taken is {time.time() - s_time}')
    print(f'Number of correct images by madry is {madry_correct}/{tot_img_count}')
    print(f'Number of correct images by ResNet is {pytorch_correct}/{tot_img_count}')

    print(f'Top5 - Madry is: {top5_madry_correct}/{tot_img_count}')
    print(f'Top5 - ResNet is: {top5_pytorch_correct}/{tot_img_count}')

    ipdb.set_trace()






