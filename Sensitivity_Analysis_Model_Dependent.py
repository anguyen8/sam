import argparse, time, os, sys, glob, warnings, ipdb, math
from RISE_evaluation import CausalMetric, auc, gkern
from skimage.measure import compare_ssim as ssim
from scipy.stats import spearmanr, pearsonr
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
import settings

def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Paramters for sensitivity analysis of heatmaps')

    parser.add_argument('-idp', '--input_dir_path', help='Path of the input directory', metavar='DIR')

    parser.add_argument('-op', '--out_path',
                        help='Path of the output directory where you want to save the results (Default is ./img_name/)')

    parser.add_argument('-mn', '--method_name', choices=['occlusion', 'ig', 'sg', 'grad', 'lime', 'mp', 'inpgrad'],
                        help='Method you are analysing')

    parser.add_argument('--metric_name', choices=['insertion', 'deletion'],
                        help='Metric to be computed')

    # parser.add_argument('--num_variations', type=int,
    #                     help='Number of variations for a particular method.')

    # parser.add_argument('--no_img_name_dir_flag', action='store_false', default=True,
    #                     help=f'Flag to say that image name is stored as seperate directory in the input path.'
    #                          f'Default=True')
    #
    # parser.add_argument('--no_model_name_dir_flag', action='store_false', default=True,
    #                     help=f'Flag to say that model name is stored as seperate directory in the input path. '
    #                          f'Default=True')

    parser.add_argument('--idx_flag', type=int, choices=range(2),
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

    parser.add_argument('--if_noise', type=int, choices=range(2),
                        help='Flag whether noise was present in the input while calculating the heatmaps. Default: False (0)',
                        default=0,
                        )

    # parser.add_argument('--if_random', action='store_true', default=False,
    #                     help=f'Flag to say you want to compute results for baseline'
    #                          f'Default=False')

    # Parse the arguments
    args = parser.parse_args()
    args.no_model_name_dir_flag = False
    args.if_random = False
    args.no_img_name_dir_flag = True
    # args.start_idx = 0
    # args.end_idx = 2000

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
def compute_score(heat, metric_name, **kwargs):

    if metric_name.lower() in ['insertion', 'deletion']:
        img = kwargs['img']
        if_noise = kwargs['if_noise']
        if if_noise == 1:
            hIdx = kwargs['heatmap_idx']
            noise_flag_list = kwargs['noise_flag_list']
            if noise_flag_list[hIdx] == 1:
                img = kwargs['noisy_img']

        metricObj = kwargs['metricObj']
        model_name = kwargs['model_name']
        # dir = abs_path(f'./temp_results/Metric_{metric_name}/Model_{model_name}')
        # eutils.mkdir_p(dir)
        ## We are computing percentage AUC
        aa = metricObj.single_run(img, heat) #, verbose=1, save_to=dir)
        out = auc(aa)

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
    print(f'My_Attacker is: {my_attacker}')

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
    model_names.append('pytorch')
    model_names.append('madry')
    model_names.append('googlenet')
    model_names.append('madry_googlenet')

    print(model_names)
    print(f'If random is : {args.if_random}')

    mean_dict = {'pytorch': [],
                 'googlenet': [],
                 'madry': [],
                 'madry_googlenet': []}

    var_dict = deepcopy(mean_dict)
    output = deepcopy(mean_dict)

    load_model_fns = {'pytorch': eval('eutils.load_orig_imagenet_model'),
                      'madry': eval('eutils.load_madry_model'),
                      'googlenet': eval('eutils.load_orig_imagenet_model'),
                      'madry_googlenet': eval('eutils.load_madry_model'),
                      }
    load_model_args = {'pytorch': 'resnet50',
                       'madry': 'madry',
                       'madry_googlenet': 'madry_googlenet',
                       'googlenet': 'googlenet'}

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

    img_file_numbers = np.array([int(imName.split('_')[-1]) for imName in img_filenames],
                                dtype=int)

    print(f'Metric: {metric_name}')
    print(f'Method: {method_name}')

    # num_inc_imgs = len([i for i in img_filenames if i in incorrect_img_list])

    kwargs = {}
    kwargs['method_name'] = method_name.lower()
    ## Only when your are dealing with insertion and deletion metrics
    if metric_name.lower() == 'insertion':
        # klen = 11
        # ksig = 5
        # kern = gkern(klen, ksig) #It has to be on CPU
        # blur = lambda x: nn.functional.conv2d(x, kern, padding=klen // 2)
        # substrate_fn = blur
        substrate_fn = torch.zeros_like
    elif metric_name.lower() == 'deletion':
        substrate_fn = torch.zeros_like


    for model_name in model_names:

        temp_mean = np.zeros(len(img_filenames), dtype=float)
        temp_std = np.zeros(len(img_filenames), dtype=float)
        modelTime = time.time()

        kwargs['model_name'] = model_name
        print(f'Calculation variation for model: {model_name}')

        assert my_attacker == True, \
            'In this case, you have to process robust models just like normal pytorch models'
        if my_attacker:
            preprocessFn = eutils.return_transform('pytorch')
        else:
            preprocessFn = eutils.return_transform(model_name)

        # print(f'My Attacker is {my_attacker} and transform is {preprocessFn}')
        if metric_name.lower() in ['insertion', 'deletion']:
            load_model = load_model_fns[model_name]
            model_arg = load_model_args[model_name]
            model = load_model(arch=model_arg, if_pre=0, my_attacker=my_attacker) #Returns prob #on cuda
            # kwargs['model'] = model
            metricObj = CausalMetric(model, metric_name[:3], 224*8, substrate_fn=substrate_fn)
            kwargs['metricObj'] = metricObj

        idx = -1
        for _, img_name in enumerate(img_filenames):
            idx += 1

            img_path = os.path.join(base_img_dir, f'{img_name}.JPEG')
            kwargs['img_path'] = img_path

            if metric_name.lower() in ['insertion', 'deletion']:
                img = read_tensor(img_path, preprocessFn) #Image has to be on CPU
                kwargs['img'] = img

                kwargs['if_noise'] = args.if_noise
                if args.if_noise == 1:
                    noisy_img = read_tensor(img_path, preprocessFn, if_noise=args.if_noise) #Image has to be on CPU
                    kwargs['noisy_img'] = noisy_img

            # preds = model(img.cuda())
            # # print(f'Pred class is {torch.argmax(preds)} and the prob is {torch.max(preds)}')

            targ_class = eutils.get_image_class(img_path)

            if len(img_filenames) < 2000:
                print(f'Calculation variation across img: {img_name}, img_number: {idx:04d}')
                print(f'Target class is {targ_class}')

            if args.no_model_name_dir_flag:
                dir_name = os.path.join(args.input_dir_path,
                                        f"{method_name}_{model_name}")
            else:
                dir_name = args.input_dir_path

            if method_name.lower() == 'lime':
                npy_str_lists = glob.glob(os.path.join(dir_name,
                                                       f'{img_name}/{model_name}.npy'))

                ##Filtering the correct cases
                if model_name == 'googlenet':
                    npy_str_lists = [i for i in npy_str_lists if 'madry_googlenet' not in i]

                if model_name == 'madry':
                    npy_str_lists = [i for i in npy_str_lists if 'madry_googlenet' not in i]

                npy_str_lists.sort()
                if args.if_noise == 1:
                    noise_flag_list = [int(aa.split('_noise_')[1].split('_')[0]) for aa in npy_str_lists]
                    kwargs['noise_flag_list'] = noise_flag_list

            elif method_name.lower() == 'mp':
                im_num = int(img_name.split('_')[-1])
                npy_str_lists = glob.glob(os.path.join(dir_name,
                                                      f'{img_name}/*{model_name}.npy'))

                ##Filtering the correct cases
                if model_name == 'googlenet':
                    npy_str_lists = [i for i in npy_str_lists if 'madry_googlenet' not in i]

                if model_name == 'madry':
                    npy_str_lists = [i for i in npy_str_lists if 'madry_googlenet' not in i]

                npy_str_lists.sort()

            else:
                npy_str_lists = glob.glob(os.path.join(dir_name,
                                                      f'{img_name}/*{model_name}*.npy'))

                ##Filtering the correct cases
                if model_name == 'googlenet':
                    npy_str_lists = [i for i in npy_str_lists if 'madry_googlenet' not in i]

                if model_name == 'madry':
                    npy_str_lists = [i for i in npy_str_lists if 'madry_googlenet' not in i]

                npy_str_lists.sort()
                if args.if_noise == 1:
                    noise_flag_list = [int(aa.split('_noise_')[1].split('_')[0]) for aa in npy_str_lists]
                    kwargs['noise_flag_list'] = noise_flag_list


            num_var = len(npy_str_lists)
            assert num_var >= 2, 'Num of variations should be greater than 2'

            # print(num_var)
            # print(npy_str_lists)

            # print(f'Loading the results')
            heatmap_list = [np.load(i) for i in npy_str_lists]

            # ## We do not need to rescale for IOU, Insertion and Deletion Metrics
            # if rescale_flag:
            #     ## Rescale the heatmaps to the original shape
            #     pass


            if method_name.lower() == 'occlusion' and metric_name.lower() != 'iou':
                ## Because for iou you threshold first and then resize
                ## I can resize the input to the image size regardless of the input shape
                ## If input size is 224, there would not be any chnage in the output
                heatmap_list = [resize(i, (224, 224), order=5) for i in heatmap_list]
            else:
                pass

            # if args.if_random:
            #     if method_name.lower() == 'mp':
            #         heatmap_list = [args.random_state.uniform(low=0, high=1, size=(i.shape)).astype(i.dtype)
            #                         for i in heatmap_list]
            #     else:
            #         heatmap_list = [args.random_state.uniform(low=-1, high=1, size=(i.shape)).astype(i.dtype)
            #                         for i in heatmap_list]

            scores = []
            for hIdx, heatmap in enumerate(heatmap_list):
                kwargs['heatmap_idx'] = hIdx
                scores.append(compute_score(heatmap, metric_name, **kwargs))

            if np.isnan(scores).any():
                temp_mean[idx] = math.nan
                temp_std[idx] = math.nan
                continue

            scores = np.array(scores) # scores is like 1D array
            temp_mean[idx] = np.mean(scores)
            temp_std[idx] = np.std(scores)

            mean_dict[model_name].append(np.mean(scores, axis=0)) #Here providing axis or not does not matter
            var_dict[model_name].append(np.var(scores, axis=0))

        print(f'Len of samples considered is {len(mean_dict[model_name])}')
        output[model_name].append(np.mean(np.array(mean_dict[model_name]), axis=0))
        output[model_name].append(np.sqrt(np.mean(np.array(var_dict[model_name]),
                                                  axis=0,
                                                  )
                                          )
                                  )

        print(f'Mean is {output[model_name][0]}, std is {output[model_name][1]}')

        print(f'Time taken to evaluate {metric_name} metric for '
              f'model {model_name} on method {method_name} is {time.time() - modelTime}')

        if len(img_filenames) >= 1:
            print(f'Saving to the text file')
            ##Save the results to the text file
            path = os.path.join(args.out_path,
                                f'Method_{method_name}_Metric_{metric_name}')
            eutils.mkdir_p(path)

            if args.if_random:
                fName = os.path.join(path, f'time_{f_time}_'
                                           f'Random_Model_{model_name}_{method_name}_{metric_name}.txt')
            else:
                fName = os.path.join(path, f'time_{f_time}_'
                                           f'Model_{model_name}_{method_name}_{metric_name}.txt')

            file_handle = open(fName, 'ab')
            temp_arr = np.concatenate((np.expand_dims(img_file_numbers, axis=-1),
                                       np.expand_dims(temp_mean, axis=-1),
                                       np.expand_dims(temp_std, axis=-1),
                                       ), axis=-1)

            np.savetxt(file_handle, temp_arr, fmt='%05d,  %.16f,  %.16f',
                       header='ImNum,  Mean              ,  Std', footer='\nCumulative Results', comments='',
                       )
            temp_arr = np.concatenate((np.array([[len(mean_dict[model_name])]]),
                                       np.array([[output[model_name][0]]]),
                                       np.array([[output[model_name][1]]]),
                                       ), axis=-1)
            np.savetxt(file_handle, temp_arr,
                       fmt='%05d,  %.16f,  %.16f',
                       header='ImCou,  Mean              ,  Std', comments='', )
            file_handle.close()

    if len(model_names) == 3:
        if len(img_filenames) >= 10:
            path = os.path.join(args.out_path,
                                f'Method_{method_name}_Metric_{metric_name}')
            eutils.mkdir_p(path)

            if args.if_random:
                fName = os.path.join(path, f'time_{f_time}_'
                                           f'Random_cumulative_results.txt')
            else:
                fName = os.path.join(path, f'time_{f_time}_'
                                           f'cumulative_results.txt')
            file_handle = open(fName, 'ab')


            temp_arr = np.concatenate((np.asarray((output['googlenet'])).reshape(1, -1),
                                       np.asarray((output['pytorch'])).reshape(1, -1),
                                       np.asarray((output['madry'])).reshape(1, -1),
                                       ), axis=0)
            np.savetxt(file_handle, temp_arr,
                       fmt='%.16f,  %.16f',
                       header='Mean              ,  Var', comments='', )
            file_handle.close()

    print(f'Time taken is {time.time() - s_time}')
    print(f'Time stamp is {f_time}')



