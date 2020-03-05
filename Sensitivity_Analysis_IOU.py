import argparse, time, os, sys, glob, warnings, ipdb, math
from RISE_evaluation import CausalMetric, auc, gkern
from skimage.measure import compare_ssim as ssim
from scipy.stats import spearmanr, pearsonr
from torchvision.transforms import transforms
from skimage.transform import resize
import xml.etree.ElementTree as ET
from itertools import combinations
from skimage.feature import hog
from srblib import abs_path
from copy import deepcopy
from RISE_utils import *
import utils as eutils
from PIL import Image
import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
import settings

def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Paramters for sensitivity analysis of heatmaps')

    parser.add_argument('-idp', '--input_dir_path', help='Path of the input directory', metavar='DIR')

    parser.add_argument('-op', '--out_path',
                        help='Path of the output directory where you want to save the results (Default is ./img_name/)')

    parser.add_argument('-mn', '--method_name',
                        choices=['occlusion', 'ig', 'sg', 'grad', 'lime', 'mp', 'inpgrad'],
                        help='Method you are analysing')

    # parser.add_argument('--metric_name', choices=['iou'],
    #                     help='Metric to be computed')

    # parser.add_argument('--num_variations', type=int,
    #                     help='Number of variations for a particular method.')

    # parser.add_argument('--no_img_name_dir_flag', action='store_false', default=True,
    #                     help=f'Flag to say that image name is stored as seperate directory in the input path.'
    #                          f'Default=True')
    #
    # parser.add_argument('--no_model_name_dir_flag', action='store_false', default=True,
    #                     help=f'Flag to say that model name is stored as seperate directory in the input path. '
    #                          f'Default=True')

    parser.add_argument('--idx_flag', type=int,
                        help=f'Flag whether to use some images in the folder (1) or all (0). '
                             f'This is just for testing purposes. '
                             f'Default=0', default=0,
                        )

    parser.add_argument('-s_idx', '--start_idx', type=int,
                        help='Start index for selecting images. Default: 0', default=0,
                        )

    parser.add_argument('-e_idx', '--end_idx', type=int,
                        help='End index for selecting images. Default: 2K', default=1735,
                        )

    # parser.add_argument('--if_random', action='store_true', default=False,
    #                     help=f'Flag to say you want to compute results for baseline'
    #                          f'Default=False')

    # Parse the arguments
    args = parser.parse_args()
    args.metric_name = 'iou'
    args.no_model_name_dir_flag = False
    args.if_random = False
    args.no_img_name_dir_flag = True
    # args.start_idx = 0
    # args.end_idx = 2000

    if args.if_random:
        args.random_state = np.random.RandomState(0)

    # if args.num_variations is None:
    #     print('Please provide this number.\nExiting')
    #     sys.exit(0)
    # elif args.num_variations < 2:
    #     print('This number cant be less than 2.\nExiting')
    #     sys.exit(0)

    if args.method_name is None:
        print('Please provide the name of the method.\nExiting')
        sys.exit(0)

    if args.metric_name is None:
        print('Please provide the name of the metric.\nExiting')
        sys.exit(0)

    if args.input_dir_path is None:
        print('Please provide image dir path. Exiting')
        sys.exit(1)
    args.input_dir_path = abs_path(args.input_dir_path)

    if args.out_path is None:
        args.out_path = './'
    args.out_path = os.path.abspath(args.out_path)

    return args


########################################################################################################################
def preprocess_gt_bb(img, size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])

    preprocessed_img_tensor = transform(np.uint8(255 * img)).numpy()
    return preprocessed_img_tensor[0, :, :]


########################################################################################################################
def get_true_bbox(img_path, base_xml_dir=abs_path(settings.imagenet_val_xml_path)):
    # parse the xml for bounding box coordinates
    temp_img = Image.open(img_path)
    sz = temp_img.size # width x height
    im_name = img_path.split('/')[-1].split('.')[0]
    tree = ET.parse(os.path.join(abs_path(base_xml_dir), f'{im_name}.xml'))

    root = tree.getroot()
    # Get Ground Truth ImageNet masks

    # temp_area = 0
    gt_mask = np.zeros((sz[1], sz[0]))  # because we want rox x col
    for iIdx, type_tag in enumerate(root.findall('object/bndbox')):
        xmin = int(type_tag[0].text)
        ymin = int(type_tag[1].text)
        xmax = int(type_tag[2].text)
        ymax = int(type_tag[3].text)
        # if (ymax - ymin)*(xmax - xmin) > temp_area:
            # temp_area = (ymax - ymin)*(xmax - xmin)
        gt_mask[ymin:ymax, xmin:xmax] = 1

    gt = preprocess_gt_bb(gt_mask, 224)
    gt = (gt >= 0.5).astype(float) #binarize after resize

    return gt


########################################################################################################################
def getbb_from_heatmap_cam(heatmap, size=None, thresh_val=0.5, thres_first=True):

    heatmap[heatmap < thresh_val] = 0
    if thres_first and size is not None:
        heatmap = resize(heatmap, size)

    bb_from_heatmap = np.zeros(heatmap.shape)

    if (heatmap == 0).all():
        if size is not None:
            bb_from_heatmap[1:size[1], 1:size[0]] = 0
            return bb_from_heatmap
        else:
            bb_from_heatmap[1:heatmap.shape[1], 1:heatmap.shape[0]] = 0
            return bb_from_heatmap

    x = np.where(heatmap.sum(0) > 0)[0] + 1
    y = np.where(heatmap.sum(1) > 0)[0] + 1
    bb_from_heatmap[y[0]:y[-1], x[0]:x[-1]] = 1
    return bb_from_heatmap


########################################################################################################################
def compute_score(heat, metric_name, **kwargs):

    if metric_name.lower() == 'iou':

        # heat = np.random.rand(heat.shape[0], heat.shape[1])

        heat[heat < 0] = 0 ##Removing the negative contributions, Min_Val = 0
        img_path = kwargs['img_path']
        gt_mask = get_true_bbox(img_path)
        max_val = heat.max()
        method_name = kwargs['method_name']

        size = (224, 224)

        thres_vals = np.arange(0.1, 0.51, 0.05)
        num_thres = len(thres_vals)

        out = []

        if max_val <= 0:
            # pred_mask = getbb_from_heatmap_cam(heat, size=size, thresh_val=0)
            # ## Since the masks has been binarized
            # mask_intersection = np.bitwise_and(gt_mask.astype(int), pred_mask.astype(int))
            # mask_union = np.bitwise_or(gt_mask.astype(int), pred_mask.astype(int))
            # IOU = np.sum(mask_intersection) / np.sum(mask_union)

            ## Since none of the pixels are important in the image
            IOU = 0
            out = [0.0]*num_thres #IOU is zero at all the sampling thresholds
        else:

            for thres in thres_vals:
                pred_mask = getbb_from_heatmap_cam(heat.copy(), size=size, thresh_val=thres*max_val)
                ## Since the masks has been binarized
                mask_intersection = np.bitwise_and(gt_mask.astype(int), pred_mask.astype(int))
                mask_union = np.bitwise_or(gt_mask.astype(int), pred_mask.astype(int))
                IOU = np.sum(mask_intersection) / np.sum(mask_union)
                out.append(IOU)

    else:
        print(f'This metric  has still not been implemented.\nExiting')
        sys.exit(1)

    return out


########################################################################################################################
if __name__ == '__main__':
    base_img_dir = abs_path(settings.imagenet_val_path)
    # base_img_dir = '/home/naman/CS231n/heatmap_tests/images/ILSVRC2012_img_val'
    # text_file = f'/home/naman/CS231n/heatmap_tests/' \
    #             f'Madri/Madri_New/robustness_applications/img_name_files/' \
    #             f'time_15669152608009198_seed_0_' \
    #             f'common_correct_imgs_model_names_madry_ressnet50_googlenet.txt'
    s_time = time.time()
    f_time = ''.join(str(s_time).split('.'))
    args = get_arguments()
    im_label_map = eutils.imagenet_label_mappings()
    my_attacker = True
    eutils.mkdir_p(args.out_path)
    img_filenames = os.listdir(args.input_dir_path)
    img_filenames = [i for i in img_filenames if 'ILSVRC2012_val_000' in i and int(i.split('_')[-1]) in range(1, 50001)]
    if args.idx_flag == 1:
        img_filenames = img_filenames[0]

    # incorrect_img_list = np.load('/home/naman/CS231n/heatmap_tests/Madri/Madri_New/'
    #                              'robustness_applications/img_name_files/'
    #                              'incorrect_img_names.npy').tolist()

    #############################################
    ## #Inits

    model_names = []
    model_names.append('googlenet')
    model_names.append('pytorch')
    model_names.append('madry')
    model_names.append('madry_googlenet')
    print(model_names)

    mean_dict = {'pytorch': [],
                 'googlenet': [],
                 'madry': [],
                 'madry_googlenet': []}

    var_dict = deepcopy(mean_dict)

    error_mean_dict = deepcopy(mean_dict)
    error_var_dict = deepcopy(mean_dict)

    output = deepcopy(mean_dict)
    error_output = deepcopy(mean_dict)

    load_model_fns = {'pytorch': eval('eutils.load_orig_imagenet_model'),
                      'madry': eval('eutils.load_madry_model'),
                      'googlenet': eval('eutils.load_orig_imagenet_model')}
    load_model_args = {'pytorch': 'resnet50', 'madry': 'madry', 'googlenet': 'googlenet'}

    method_dict = {'grad': 'Grad',
                   'inpgrad': 'InpGrad',
                   'ig': 'IG',
                   'lime': 'Lime',
                   'mp': 'MP',
                   'occlusion': 'Occlusion',
                   'sg': 'SmoothGrad',
                   }

    method_name = method_dict[args.method_name]
    metric_name = args.metric_name

    if method_name.lower() in ['occlusion', 'mp']:
        rescale_flag = False
    elif method_name.lower() in ['grad', 'inpgrad', 'ig', 'lime', 'smoothgrad']:
        rescale_flag = True

    print(f'Rescale flag is {rescale_flag}')
    #############################################

    # img_filenames = []
    # with open(text_file, 'r') as f:
    #     img_filenames = f.read().splitlines()
    #     img_filenames = img_filenames[args.start_idx:args.end_idx]
    # if args.idx_flag == 1:
    #     img_filenames = img_filenames[0]

    # img_file_numbers = np.array([int(imName.split('_')[-1]) for imName in img_filenames], dtype=int)
    img_file_numbers = np.array([int(imName.split('_')[-1]) for imName in img_filenames],
                                dtype=int)
    print(f'No of images are {len(img_filenames)}')

    print(f'Metric: {metric_name}')
    print(f'Method: {method_name}')

    # num_inc_imgs = len([i for i in img_filenames if i in incorrect_img_list])

    kwargs = {}
    kwargs['method_name'] = method_name.lower()
    ## Only when your are dealing with insertion and deletion metrics
    if metric_name.lower() == 'insertion':
        klen = 11
        ksig = 5
        kern = gkern(klen, ksig) #It has to be on CPU
        blur = lambda x: nn.functional.conv2d(x, kern, padding=klen // 2)
        # substrate_fn = blur
        substrate_fn = torch.zeros_like
    elif metric_name.lower() == 'deletion':
        substrate_fn = torch.zeros_like


    for modelIdx, model_name in enumerate(model_names):
        print(f'Analyzing for model {model_name}')
        temp_mean = np.zeros((len(img_filenames), 9), dtype=float) ##because there are 9 thresholds value
        temp_std = np.zeros_like(temp_mean)

        error_temp_mean = np.zeros_like(temp_mean)
        error_temp_std = np.zeros_like(temp_mean)

        modelTime = time.time()
        kwargs['model_name'] = model_name
        print(f'Calculation variation for model: {model_name}')

        if my_attacker:
            preprocessFn = eutils.return_transform('pytorch')
        else:
            preprocessFn = eutils.return_transform(model_name)

        idx = -1
        for _, img_name in enumerate(img_filenames):
            idx += 1

            img_path = os.path.join(base_img_dir, f'{img_name}.JPEG')
            kwargs['img_path'] = img_path

            targ_class = eutils.get_image_class(img_path)

            if len(img_filenames) < 2000:
                print(f'Calculation variation across img: {img_name}, img_number: {idx:04d}')
                # print(f'Target class is {targ_class}')

            if args.no_model_name_dir_flag:
                dir_name = os.path.join(args.input_dir_path,
                                        f"{method_name}_{model_name}")
            else:
                dir_name = args.input_dir_path

            if method_name.lower() == 'lime':
                npy_str_lists = glob.glob(os.path.join(dir_name,
                                                       f'{img_name}/*{model_name}.npy'))

            elif method_name.lower() == 'mp':
                im_num = int(img_name.split('_')[-1])
                npy_str_lists = glob.glob(os.path.join(dir_name,
                                                       f'{img_name}/*{model_name}.npy'))

            else:
                npy_str_lists = glob.glob(os.path.join(dir_name,
                                                       f'{img_name}/*{model_name}*.npy'))

            ##Filtering the correct cases
            if model_name == 'googlenet':
                npy_str_lists = [i for i in npy_str_lists if 'madry_googlenet' not in i]

            if model_name == 'madry':
                npy_str_lists = [i for i in npy_str_lists if 'madry_googlenet' not in i]

            npy_str_lists.sort()
            num_var = len(npy_str_lists)
            assert num_var >= 2, 'Num of variations should be greater than 2'

            # print(f'Loading the results')
            heatmap_list = [np.load(i) for i in npy_str_lists]

            # ## We do not need to rescale for IOU, Insertion and Deletion Metrics
            # if rescale_flag:
            #     ## Rescale the heatmaps to the original shape
            #     pass


            if method_name.lower() == 'occlusion' and metric_name != 'iou':
                ## Because for iou you threshold first and then resize
                ## I can resize the input to the image size regardless of the input shape
                ## If input size is 224, there would not be any chnage in the output
                heatmap_list = [resize(i, (224, 224), order=5) for i in heatmap_list]
            else:
                pass

            if args.if_random:
                # if method_name.lower() == 'mp':
                #     heatmap_list = [args.random_state.uniform(low=0, high=1, size=(i.shape)).astype(i.dtype)
                #                     for i in heatmap_list]
                # else:
                #     heatmap_list = [args.random_state.uniform(low=-1, high=1, size=(i.shape)).astype(i.dtype)
                #                     for i in heatmap_list]

                heatmap_list = [np.ones_like(i) for i in heatmap_list]


            # req_comb = list(combinations(range(num_var),2))
            scores = []
            for hIdx, heatmap in enumerate(heatmap_list):
                scores.append(compute_score(heatmap, metric_name, **kwargs))

            # I do not think nanis ever going to happen in IOU calculation
            if np.isnan(scores).any():
                temp_mean[idx] = math.nan
                temp_std[idx] = math.nan
                continue

            scores = np.array(scores) #[3x9]
            ## Localization error
            error_mat = np.where(scores > 0.5, 0, 1).astype(float)

            temp_mean[idx, :] = np.mean(scores, axis=0)
            temp_std[idx, :] = np.std(scores, axis=0)

            error_temp_mean[idx, :] = np.mean(error_mat, axis=0)
            error_temp_std[idx, :] = np.std(error_mat, axis=0)

            mean_dict[model_name].append(np.mean(scores, axis=0))
            var_dict[model_name].append(np.var(scores, axis=0))

            error_mean_dict[model_name].append(np.mean(error_mat, axis=0))
            error_var_dict[model_name].append(np.var(error_mat, axis=0))

        print(f'Len of samples considered is {len(mean_dict[model_name])}')
        output[model_name].append(np.mean(np.array(mean_dict[model_name]), axis=0))
        output[model_name].append(np.sqrt(np.mean(np.array(var_dict[model_name]),
                                                  axis=0)
                                          )
                                  )

        error_output[model_name].append(np.mean(np.array(error_mean_dict[model_name]), axis=0))
        error_output[model_name].append(np.sqrt(np.mean(np.array(error_var_dict[model_name]),
                                                        axis=0)
                                                )
                                        )

        print(f'Mean is {output[model_name][0]}, \nStd is {output[model_name][1]}')
        print(f'Error Mean is {error_output[model_name][0]}, \nError Std is {error_output[model_name][1]}')

        print(f'Time taken to evaluate {metric_name} metric for '
              f'model {model_name} on method {method_name} is {time.time() - modelTime}')

        if len(img_filenames) >= 2:
            print(f'Saving the text files')
            path = os.path.join(args.out_path, f'Method_{method_name}_Metric_{metric_name}/Model_{model_name}/')
            eutils.mkdir_p(path)
            for thresIdx in range(9):
                ##Save the results to the text file
                ## IOU
                if args.if_random:
                    path = os.path.join(args.out_path,
                                        f'Method_{method_name}_Metric_{metric_name}')
                    fName = os.path.join(path, f'time_{f_time}_IOU_FULL_IMAGE_'
                                               f'{method_name}_{metric_name}_thresholdIdx_{thresIdx}.txt')

                else:
                    fName = os.path.join(path, f'time_{f_time}_IOU_Model_{model_name}_'
                                               f'{method_name}_{metric_name}_thresholdIdx_{thresIdx}.txt')

                file_handle = open(fName, 'ab')
                temp_arr = np.concatenate((np.expand_dims(img_file_numbers, axis=-1),
                                           np.expand_dims(temp_mean[:, thresIdx], axis=-1),
                                           np.expand_dims(temp_std[:, thresIdx], axis=-1),
                                           ), axis=-1)

                np.savetxt(file_handle, temp_arr, fmt='%05d,  %.16f,  %.16f',
                           header='ImNum,  Mean              ,  Std', footer='\nCumulative Results', comments='',
                           )
                temp_arr = np.concatenate((np.array([[len(mean_dict[model_name])]]),
                                           np.array([[output[model_name][0][thresIdx]]]),
                                           np.array([[output[model_name][1][thresIdx]]]),
                                           ), axis=-1)
                np.savetxt(file_handle, temp_arr,
                           fmt='%05d,  %.16f,  %.16f',
                           header='ImCou,  Mean              ,  Std', comments='', )
                file_handle.close()

                ##Localizaion error
                if args.if_random:
                    path = os.path.join(args.out_path,
                                        f'Method_{method_name}_Metric_{metric_name}')
                    fName = os.path.join(path, f'time_{f_time}_Error_Random_'
                                               f'{method_name}_{metric_name}_thresholdIdx_{thresIdx}.txt')
                else:
                    fName = os.path.join(path, f'time_{f_time}_Error_Model_{model_name}_'
                                               f'{method_name}_{metric_name}_thresholdIdx_{thresIdx}.txt')
                file_handle = open(fName, 'ab')
                temp_arr = np.concatenate((np.expand_dims(img_file_numbers, axis=-1),
                                           np.expand_dims(error_temp_mean[:, thresIdx], axis=-1),
                                           np.expand_dims(error_temp_std[:, thresIdx], axis=-1),
                                           ), axis=-1)

                np.savetxt(file_handle, temp_arr, fmt='%05d,  %.16f,  %.16f',
                           header='ImNum,  Mean              ,  Std', footer='\nCumulative Results', comments='',
                           )
                temp_arr = np.concatenate((np.array([[len(mean_dict[model_name])]]),
                                           np.array([[error_output[model_name][0][thresIdx]]]),
                                           np.array([[error_output[model_name][1][thresIdx]]]),
                                           ), axis=-1)
                np.savetxt(file_handle, temp_arr,
                           fmt='%05d,  %.16f,  %.16f',
                           header='ImCou,  Mean              ,  Std', comments='', )
                file_handle.close()

            ##cumulative results
            if len(model_names) == 4:
                path = os.path.join(args.out_path, f'Method_{method_name}_Metric_{metric_name}')
                ## IOU
                fName = os.path.join(path, f'time_{f_time}_Cumulative_IOU_Model_{model_name}_'
                                           f'{method_name}_{metric_name}.txt')
                file_handle = open(fName, 'ab')
                temp_arr = np.concatenate((np.expand_dims(np.arange(0.1, 0.51, 0.05), axis=-1),  # Idx
                                           np.expand_dims(output[model_name][0], axis=-1),  # Mean
                                           np.expand_dims(output[model_name][1], axis=-1),  # Std
                                           ), axis=-1)
                np.savetxt(file_handle, temp_arr, fmt='%.3f,  %.16f,  %.16f',
                           header='Alpha,  Mean              ,  Std', comments='',
                           )
                file_handle.close()

                ## Error
                fName = os.path.join(path, f'time_{f_time}_Cumulative_Error_Model_{model_name}_'
                                           f'{method_name}_{metric_name}.txt')
                file_handle = open(fName, 'ab')
                temp_arr = np.concatenate((np.expand_dims(np.arange(0.1, 0.51, 0.05), axis=-1),  # Idx
                                           np.expand_dims(error_output[model_name][0], axis=-1),  # Mean
                                           np.expand_dims(error_output[model_name][1], axis=-1),  # Var
                                           ), axis=-1)
                np.savetxt(file_handle, temp_arr, fmt='%.3f,  %.16f,  %.16f',
                           header='Alpha,  Mean              ,  Var', comments='',
                           )
                file_handle.close()
    ##############

    print(f'Time taken is {time.time() - s_time}')
    print(f'Time stamp is {f_time}')



