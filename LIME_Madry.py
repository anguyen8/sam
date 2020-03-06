################################################################################################################
## IN our implementation, we geenrate LIME heatmaps for the 4 models used in our paper
## i.e. we load all the four models in the memory in the very beginning.

## Make sure you have enough memory since this one (only this) is hard-coded to run for all the 4 models.
################################################################################################################

from __future__ import absolute_import
import warnings
warnings.simplefilter('ignore')
import ipdb, os, sys, json, glob, time, argparse

from skimage.io import imread

from srblib import abs_path
import numpy as np

from PIL import Image
import torch.nn as nn
import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

# from robustness import model_utils, datasets
from user_constants import DATA_PATH_DICT
import utils as eutils
import settings

import skimage
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries

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


def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Input paramters for LIME explanation of the images')

    parser.add_argument('-idp', '--img_dir_path', help='Path of the image directory', metavar='DIR')

    parser.add_argument('-op', '--out_path',
                        help='Path of the output directory where you want to save the results (Default is ./)')

    # parser.add_argument('-gpu', '--gpu', type=int,
    #                     help='GPU index', default=0,
    #                     )

    parser.add_argument('-ifp', '--if_pre', type=int, choices=range(2),
                        help='It is clear from name. Default: Post (0)', default=0,
                        )

    parser.add_argument('-if_sp', '--if_save_plot', type=int, choices=range(2),
                        help='Whether save the plots. Default: No (0)', default=0,
                        )

    parser.add_argument('-if_sn', '--if_save_npy', type=int, choices=range(2),
                        help='Whether save the plots. Default: Yes (1)', default=1,
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

    parser.add_argument('-l_bp', '--lime_background_pixel',
                        help=f'Background pixel for lime to be used for absence of super-pixel.'
                             f'Options - a number between (0-255), random, grey, none (mean of each superpixel) (default)',
                        )

    parser.add_argument('-l_sn', '--lime_superpixel_num', type=int,
                       help='Number of super pixels used by Lime. Default=50', default=50,
                       )

    parser.add_argument('-l_ns', '--lime_num_samples', type=int,
                        help='Number of samples used by Lime. Default=1000', default=1000,
                        )

    parser.add_argument('-l_ss', '--lime_sup_seed', type=int,
                        help=f'Seed for superpixel algorithm. '
                             f'It is only used by QuickShift algorithm.'
                             f'No effect on Slic algorithm (which is default for us).'
                             f'Default Value =0', #TODO: Implement QuickShift as well
                        default=0,
                        )

    parser.add_argument('-l_es', '--lime_explainer_seed', type=int,
                        help=f'Seed to creating Lime explainer (sampling of data points).'
                             f'Default=0',
                        default=0,
                        )

    parser.add_argument('-ifn', '--if_noise', type=int, choices=range(2),
                        help='Whether to add noise to the image or not. Default: No (0)', default=0,
                        )

    parser.add_argument('-mean', '--mean', type=float,
                        help='Mean of gaussian noise. Default: 0', default=0,
                        )

    parser.add_argument('-var', '--var', type=float,
                        help='Variance of gaussian noise. Default: 0.1', default=0.1,
                        )

    # Parse the arguments
    args = parser.parse_args()

    if args.lime_background_pixel is not None:
        if args.lime_background_pixel.lower() == 'random':
            args.lime_background_pixel = 'random'
        elif args.lime_background_pixel.lower() == 'grey':
            args.lime_background_pixel = 'grey'
        elif args.lime_background_pixel.lower() == 'none':
            args.lime_background_pixel = None
        elif args.lime_background_pixel.isdigit():
            args.lime_background_pixel = int(args.lime_background_pixel)
            if args.lime_background_pixel < 0 or args.lime_background_pixel > 255:
                print('Provide a valid option for background pixel.\nExiting')
                sys.exit(1)
        else:
            print('Please provide a valid option for background pixel. Exiting')
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
    def __init__(self, data_path, madry_transform, pytorch_transform,
                 img_idxs=[0, 1], idx_flag=1, if_noise=0, mean=0, var=0.1):
        self.data_path = data_path
        self.madry_transform = madry_transform
        self.pytorch_transform = pytorch_transform
        self.if_noise = if_noise
        self.mean = mean
        self.var = var

        if data_path == abs_path(settings.imagenet_val_path):
            aa = img_name_list[img_idxs[0]:img_idxs[1]]
            self.img_filenames = [os.path.join(data_path, f'{ii}.JPEG') for ii in aa]

            # self.img_filenames.sort()

        else:
            self.img_filenames = []
            for file in glob.glob(os.path.join(data_path, "*.JPEG")):
                self.img_filenames.append(file)
            self.img_filenames.sort()

        print(f'\nNo. of images to be analyzed are {len(self.img_filenames)}\n')

        if idx_flag == 1:
            print('Only prodicing results for 1 image')
            img_idxs = [0]
            self.img_filenames = [self.img_filenames[i] for i in img_idxs]



    def __getitem__(self, index):
        # ipdb.set_trace()
        img = Image.open(os.path.join(self.data_path, self.img_filenames[index])).convert('RGB')
        if self.if_noise == 1:
            print(f'Adding noise the image with mean: {self.mean} and var: {self.var}')
            img = np.asarray(img)
            img = skimage.util.random_noise(img.copy(), 'gaussian',
                                            mean=self.mean, var=self.var, seed=0)
            img = Image.fromarray(np.uint8(img * 255))

        y = self.get_image_class(os.path.join(self.data_path, self.img_filenames[index]))

        madry_img = self.madry_transform(img)
        pytorch_img = self.pytorch_transform(img)

        return madry_img, pytorch_img, y, os.path.join(self.data_path, self.img_filenames[index])

    def __len__(self):
        return len(self.img_filenames)

    def get_image_class(self, filepath):
        base_dir = '/home/naman/CS231n/heatmap_tests/'
        # ImageNet 2012 validation set images?
        with open(os.path.join(settings.imagenet_class_mappings, "ground_truth_val2012")) as f:
        # with open(os.path.join(base_dir, "imagenet_class_mappings", "ground_truth_val2012")) as f:
            ground_truth_val2012 = {x.split()[0]: int(x.split()[1])
                                    for x in f.readlines() if len(x.strip()) > 0}

        with open(os.path.join(settings.imagenet_class_mappings, "synset_id_to_class")) as f:
        # with open(os.path.join(base_dir, "imagenet_class_mappings", "synset_id_to_class")) as f:
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

def load_data(img_dir, batch_size=1, img_idxs=[0, 1], idx_flag=1, if_noise=0, mean=0, var=0.1):

    # 1. Define the appropriate image pre-processing function.
    madry_preprocessFn = transforms.Compose([transforms.Resize((256, 256)),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       ])

    pytorch_preprocessFn = transforms.Compose([transforms.Resize((256, 256)),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

    data = DataProcessing(img_dir, madry_preprocessFn, pytorch_preprocessFn,
                          img_idxs=img_idxs, idx_flag=idx_flag,
                          if_noise=if_noise, mean=mean, var=var)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    return test_loader, len(data)


def get_pytorch_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    return transf

def get_madry_preprocess_transform():
    transf = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transf

def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])
    return transf

def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

if __name__ == '__main__':
    s_time = time.time()
    f_time = ''.join(str(s_time).split('.'))
    print(f'Time stamp is {f_time}')
    args = get_arguments()

    ## #Label Mappings
    im_label_map = eutils.imagenet_label_mappings()

    ## #Models
    madry_model = eutils.load_madry_model(if_pre=args.if_pre)
    pytorch_model = eutils.load_orig_imagenet_model(arch='resnet50', if_pre=args.if_pre)
    gNet_model = eutils.load_orig_imagenet_model(arch='googlenet', if_pre=args.if_pre)
    gNet_R_model = eutils.load_madry_model(arch='madry_googlenet', if_pre=args.if_pre)

    ## #Explainer
    madry_explainer = lime_image.LimeImageExplainer(random_state=args.lime_explainer_seed)
    pytorch_explainer = lime_image.LimeImageExplainer(random_state=args.lime_explainer_seed)
    gNet_explainer = lime_image.LimeImageExplainer(random_state=args.lime_explainer_seed)
    gNet_R_explainer = lime_image.LimeImageExplainer(random_state=args.lime_explainer_seed)

    ## #Super-pixel algo
    slic_parameters = {'n_segments': args.lime_superpixel_num,
                       'compactness': 30,
                       'sigma': 3,
                       'random_seed':args.lime_sup_seed}
    segmenter = SegmentationAlgorithm('slic', **slic_parameters)

    pill_transf = get_pil_transform()
    #########################################################
    ## #Function to compute probabilities
    # Pytorch
    pytorch_preprocess_transform = get_pytorch_preprocess_transform()
    def pytorch_batch_predict(images):
        pytorch_model.eval()
        batch = torch.stack(tuple(pytorch_preprocess_transform(i) for i in images), dim=0)
        batch = batch.cuda()
        if args.if_pre == 1:
            logits = pytorch_model(batch)
            probs = F.softmax(logits, dim=1)
        else:
            probs = pytorch_model(batch)
        return probs.data.cpu().numpy()

    ## #GoogleNet
    gNet_preprocess_transform = get_pytorch_preprocess_transform()
    def gNet_batch_predict(images):
        gNet_model.eval()
        batch = torch.stack(tuple(gNet_preprocess_transform(i) for i in images), dim=0)
        batch = batch.cuda()
        if args.if_pre == 1:
            logits = gNet_model(batch)
            probs = F.softmax(logits, dim=1)
        else:
            probs = gNet_model(batch)
        return probs.data.cpu().numpy()

    # Madry
    madry_preprocess_transform = get_madry_preprocess_transform()
    def madry_batch_predict(images):
        madry_model.eval()
        batch = torch.stack(tuple(madry_preprocess_transform(i) for i in images), dim=0)
        batch = batch.cuda()
        if args.if_pre == 1:
            logits = madry_model(batch)
            probs = F.softmax(logits, dim=1)
        else:
            probs = madry_model(batch)
        return probs.data.cpu().numpy()

    # GoogleNet-R
    gNet_R_preprocess_transform = get_madry_preprocess_transform()
    def gNet_R_batch_predict(images):
        gNet_R_model.eval()
        batch = torch.stack(tuple(gNet_R_preprocess_transform(i) for i in images), dim=0)
        batch = batch.cuda()
        if args.if_pre == 1:
            logits = gNet_R_model(batch)
            probs = F.softmax(logits, dim=1)
        else:
            probs = gNet_R_model(batch)
        return probs.data.cpu().numpy()

    #########################################################
    #data
    data_loader, img_count = load_data(args.img_dir_path, batch_size=1, img_idxs=[args.start_idx, args.end_idx],
                                       idx_flag=args.idx_flag, if_noise=args.if_noise, mean=args.mean, var=args.var)
    ###################################################################

    # madry_correct = 0
    # pytorch_correct  = 0
    # gNet_correct = 0

    batch_size = 100
    print(f'Out path is {args.out_path}')
    for i, (madry_img, pytorch_img, targ_class, img_path) in enumerate(data_loader):

        print(f'Analysing batch: {i}')

        ## This image will be passed to Lime Explainer
        img = get_image(img_path[0])
        if args.if_noise == 1:
            # print(f'Adding noise the image with mean: {args.mean} and var: {args.var}')
            img = np.asarray(img)
            img = skimage.util.random_noise(img.copy(), 'gaussian',
                                            mean=args.mean, var=args.var, seed=0)

            img = Image.fromarray(np.uint8(img * 255))

        ########
        if use_cuda:
            pytorch_img = pytorch_img.cuda()
            madry_img = madry_img.cuda()

        gNet_img = pytorch_img.clone() #Since their preprocessing is all the same
        gNet_R_img = madry_img.clone()  # Since their preprocessing is all the same
        targ_class = targ_class.cpu()

        #Prob
        if args.if_pre == 1:
            print('Pre softmax analysis')
            pytorch_logits = pytorch_model(pytorch_img)
            pytorch_probs = F.softmax(pytorch_logits, dim=1).cpu()
            pytorch_logits = pytorch_logits.cpu()

            gNet_logits = gNet_model(gNet_img)
            gNet_probs = F.softmax(gNet_logits, dim=1).cpu()
            gNet_logits = gNet_logits.cpu()

            madry_logits = madry_model(madry_img)
            madry_probs = F.softmax(madry_logits, dim=1).cpu()
            madry_logits = madry_logits .cpu()

            gNet_R_logits = gNet_R_model(gNet_R_img)
            gNet_R_probs = F.softmax(gNet_R_logits, dim=1).cpu()
            gNet_R_logits = gNet_R_logits.cpu()
            softmax = 'pre'

        else:
            print('Post softmax analysis')
            pytorch_probs = pytorch_model(pytorch_img).cpu()
            gNet_probs = gNet_model(gNet_img).cpu()
            madry_probs = madry_model(madry_img).cpu()
            gNet_R_probs = gNet_R_model(gNet_R_img).cpu()
            softmax = 'post'

        madry_prediction = torch.argmax(madry_probs, dim=-1).cpu().item()
        pytorch_prediction = torch.argmax(pytorch_probs, dim=-1).cpu().item()
        gNet_prediction = torch.argmax(gNet_probs, dim=-1).cpu().item()
        gNet_R_prediction = torch.argmax(gNet_R_probs, dim=-1).cpu().item()
        true_class = targ_class.cpu().item()
        # if madry_prediction == true_class:
        #     madry_correct += 1
        #
        # if pytorch_prediction == true_class:
        #     pytorch_correct += 1
        #
        # if gNet_prediction == true_class:
        #     gNet_correct += 1

        # print(f'Madry Prediction: {madry_prediction}\nResNet Prediction: {pytorch_prediction}')
        # print(f'GoogleNet Prediction: {gNet_prediction}\nTrue class: {true_class}')

        # if madry_prediction == true_class and pytorch_prediction == true_class and gNet_prediction == true_class:
        #     print(f'Condition satisfied for image: {i}. Analyzing')
        # else:
        #     print(f'Condition did not satisfied for image: {i}. Trying for next image')
        #     continue

        #################################################
        # LIME analysis
        lime_img = np.array(pill_transf(img)) # Same image is used for all the explainers

        ## Madry
        print(f'Explaining Madry model')
        madry_lime_explanation = madry_explainer.explain_instance(lime_img,
                                                                  madry_batch_predict,
                                                                  batch_size=batch_size,
                                                                  segmentation_fn=segmenter,
                                                                  top_labels=None, #1000,
                                                                  labels=(true_class,),
                                                                  hide_color=args.lime_background_pixel,
                                                                  num_samples=args.lime_num_samples,
                                                                  )
        madry_segments = madry_lime_explanation.segments
        madry_heatmap = np.zeros(madry_segments.shape)
        local_exp = madry_lime_explanation.local_exp
        exp = local_exp[true_class]
        # exp = local_exp[madry_prediction]

        for i, (seg_idx, seg_val) in enumerate(exp):
            madry_heatmap[madry_segments == seg_idx] = seg_val

        ## PyTorch
        print(f'Explaining Pytorch model')
        pytorch_lime_explanation = pytorch_explainer.explain_instance(lime_img,
                                                                      pytorch_batch_predict,
                                                                      batch_size=batch_size,
                                                                      segmentation_fn=segmenter,
                                                                      top_labels=None, #1000,
                                                                      labels=(true_class,),
                                                                      hide_color=args.lime_background_pixel,
                                                                      num_samples=args.lime_num_samples,
                                                                      )
        pytorch_segments = pytorch_lime_explanation.segments
        pytorch_heatmap = np.zeros(pytorch_segments.shape)
        local_exp = pytorch_lime_explanation.local_exp
        exp = local_exp[true_class]
        # exp = local_exp[pytorch_prediction]

        for i, (seg_idx, seg_val) in enumerate(exp):
            pytorch_heatmap[pytorch_segments == seg_idx] = seg_val

        ## GoogleNet
        print(f'Explaining GoogleNet model')
        gNet_lime_explanation = gNet_explainer.explain_instance(lime_img,
                                                                gNet_batch_predict,
                                                                batch_size=batch_size,
                                                                segmentation_fn=segmenter,
                                                                top_labels=None, #1000,
                                                                labels=(true_class,),
                                                                hide_color=args.lime_background_pixel,
                                                                num_samples=args.lime_num_samples,
                                                                )
        gNet_segments = gNet_lime_explanation.segments
        gNet_heatmap = np.zeros(gNet_segments.shape)
        local_exp = gNet_lime_explanation.local_exp
        exp = local_exp[true_class]

        for i, (seg_idx, seg_val) in enumerate(exp):
            gNet_heatmap[gNet_segments == seg_idx] = seg_val

        ## GoogleNet-R
        print(f'Explaining GoogleNet model')
        gNet_R_lime_explanation = gNet_R_explainer.explain_instance(lime_img,
                                                                    gNet_R_batch_predict,
                                                                    batch_size=batch_size,
                                                                    segmentation_fn=segmenter,
                                                                    top_labels=None,  # 1000,
                                                                    labels=(true_class,),
                                                                    hide_color=args.lime_background_pixel,
                                                                    num_samples=args.lime_num_samples,
                                                                    )
        gNet_R_segments = gNet_R_lime_explanation.segments
        gNet_R_heatmap = np.zeros(gNet_R_segments.shape)
        local_exp = gNet_R_lime_explanation.local_exp
        exp = local_exp[true_class]

        for i, (seg_idx, seg_val) in enumerate(exp):
            gNet_R_heatmap[gNet_R_segments == seg_idx] = seg_val



        # ipdb.set_trace()
        # np.save(os.path.join(args.out_path, 'madry_heatmap.npy'), madry_heatmap)
        # np.save(os.path.join(args.out_path, 'pytorch_heatmap.npy'), pytorch_heatmap)
        # #################################################

        # ## Check
        # if np.sum(np.abs(madry_segments - pytorch_segments)) == 0 and np.sum(np.abs(madry_segments - gNet_segments)) == 0:
        #     pass
        # else:
        #     print(f'Something is wrong with the code\nSegments should be same.\nExiting')
        #     sys.exit(0)

        if isinstance(args.lime_background_pixel, int):
            temp_background_pixel = f'{args.lime_background_pixel:03d}'
        else:
            temp_background_pixel = f'{args.lime_background_pixel}'

        img_name = img_path[0].split('/')[-1].split('.')[0]
        par_name = f'sample_count_{args.lime_num_samples:05d}_' \
                   f'superpixel_seed_{args.lime_sup_seed}_' \
                   f'explainer_seed_{args.lime_explainer_seed}_' \
                   f'background_pixel_{temp_background_pixel}_' \
                   f'superpixel_count_{args.lime_superpixel_num:04d}_' \
                   f'softmax_{softmax}_' \
                   f'noise_{args.if_noise}_mean_{args.mean}_var_{args.var}'
        out_dir = os.path.join(args.out_path, img_name)
        eutils.mkdir_p(out_dir)

        print(f'Img name is {img_name}')

        if args.if_save_npy == 1:
            # np.save(os.path.join(out_dir, f'time_{f_time}_heatmaps_{img_name}_{par_name}_googlenet.npy'), gNet_heatmap)
            # np.save(os.path.join(out_dir, f'time_{f_time}_heatmaps_{img_name}_{par_name}_pytorch.npy'), pytorch_heatmap)
            # np.save(os.path.join(out_dir, f'time_{f_time}_heatmaps_{img_name}_{par_name}_madry.npy'), madry_heatmap)
            # np.save(os.path.join(out_dir, f'time_{f_time}_heatmaps_{img_name}_{par_name}_madry_googlenet.npy'),
            #         gNet_R_heatmap)

            np.save(os.path.join(out_dir, f'lime_{par_name}_model_name_googlenet.npy'), gNet_heatmap)
            np.save(os.path.join(out_dir, f'lime_{par_name}_model_name_pytorch.npy'), pytorch_heatmap)
            np.save(os.path.join(out_dir, f'lime_{par_name}_model_name_madry.npy'), madry_heatmap)
            np.save(os.path.join(out_dir, f'lime_{par_name}_model_name_madry_googlenet.npy'),
                    gNet_R_heatmap)



        if args.if_save_plot==1:
            madry_img = madry_img.cpu().data.numpy()
            orig_img = np.rollaxis(madry_img, 1, 4)
            for idx in range(len(targ_class)):
                ############################################
                # Saving the heatmaps
                img_name = img_path[idx].split('/')[-1].split('.')[0]
                par_name = f'sample_count_{args.lime_num_samples:05d}_' \
                           f'superpixel_seed_{args.lime_sup_seed}_' \
                           f'explainer_seed_{args.lime_explainer_seed}_' \
                           f'background_pixel_{temp_background_pixel}_' \
                           f'superpixel_count_{args.lime_superpixel_num:04d}_' \
                           f'softmax_{softmax}_' \
                           f'noise_{args.if_noise}_mean_{args.mean}_var_{args.var}'

                # out_dir = os.path.join(args.out_path,
                #                        f'exp_LIME_sample_count_{args.lime_num_samples:05d}_'
                #                        f'sample_seed_{args.lime_sup_seed}_'
                #                        f'background_pixel_{args.lime_background_pixel:03d}_'
                #                        f'superpixel_count_{args.lime_superpixel_num:04d}_'
                #                        f'softmax_{softmax}_time_{f_time}/{img_name}')

                out_dir = os.path.join(args.out_path, img_name)

                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                t_cls = targ_class[idx].item()

                # ipdb.set_trace()
                grid = [[orig_img[idx], gNet_heatmap, pytorch_heatmap, madry_heatmap]] #Since there is just heatmaps
                col_labels = ['Orig Image', 'GoogleNet', 'ResNet', 'Madry_ResNet']
                img_shape = grid[0][0].shape[0]

                ## For the orig image
                text = []
                text.append(("%.3f" % torch.max(madry_probs[idx, :]).item(),  # Madry prob (pL)
                             "%3d" % torch.argmax(madry_probs[idx, :]).item(),  # Madry Label (pL)
                             "%.3f" % torch.max(pytorch_probs[idx, :]).item(),  # pytorch_prob (pL)
                             "%3d" % torch.argmax(pytorch_probs[idx, :]).item(),  # Pytorch Label (pL)
                             "%.3f" % torch.max(gNet_probs[idx, :]).item(),  # pytorch_prob (pL)
                             "%3d" % torch.argmax(gNet_probs[idx, :]).item(),  # Pytorch Label (pL)
                             "%3d" % t_cls,  # label for given neuron (cNL)
                             ))

                madryProb, madryLabel, pytorchProb, pytorchLabel, gNetProb, gNetLabel, trueLabel = zip(*text)

                row_labels_left = [(f'Madry: Top-1:\n{im_label_map[int(madryLabel[i])]}: {madryProb[i]}\n',
                                    f'ResNet: Top-1:\n{im_label_map[int(pytorchLabel[i])]}: {pytorchProb[i]}\n',
                                    f'GoogleNet: Top-1:\n{im_label_map[int(gNetLabel[i])]}: {gNetProb[i]}\n',
                                    f'Target:  {int(trueLabel[i])}\n{im_label_map[int(trueLabel[i])]}')
                                   for i in range(len(madryProb))]

                row_labels_right = []

                eutils.zero_out_plot_multiple_patch(grid,
                                             out_dir,
                                             row_labels_left,
                                             row_labels_right,
                                             col_labels,
                                             file_name=f'LIME_heatmap_{par_name}_time_{f_time}.jpeg',
                                             dpi=img_shape,
                                             )
        else:
            print('Not saving the plots')

########################################
    print(f'Par_name is {par_name}')
    print(f'Time stamp is {f_time}')
    print(f'Time taken is {time.time() - s_time}')
    # # if pytorch_img.shape[0] == 1:
    # #     print(f'Madry correct count is: {madry_correct}')
    # #     print(f'ResNet correct count is: {pytorch_correct}')
    # #     print(f'GoogleNet correct count is: {gNet_correct}')
    # # ipdb.set_trace()
    # aa = 1

