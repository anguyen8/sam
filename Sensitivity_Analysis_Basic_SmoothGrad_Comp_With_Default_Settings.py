import argparse, time, os, sys, glob, warnings, ipdb, math
from RISE_evaluation import CausalMetric, auc, gkern
from skimage.measure import compare_ssim as ssim
from scipy.stats import spearmanr, pearsonr
from skimage.transform import resize
from itertools import combinations
from skimage.feature import hog
from srblib import abs_path
from copy import deepcopy
from RISE_utils import *
import utils as eutils
import torch.nn as nn
import numpy as np
import torch
import settings

def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Paramters for sensitivity analysis of heatmaps')

    parser.add_argument('-idp', '--input_dir_path', help='Path of the input directory', metavar='DIR')

    parser.add_argument('-op', '--out_path',
                        help='Path of the output directory where you want to save the text files (Default is ./)')

    parser.add_argument('-mn', '--method_name', choices=['sg'],
                        #['occlusion', 'ig', 'sg', 'grad', 'lime', 'mp', 'inpgrad'],
                        help='Method you are analysing')

    parser.add_argument('--exp_num', choices=['a30', 'a31', ],
                        help='Which experiment of SmoothGrad')

    parser.add_argument('--metric_name', choices=['ssim', 'hog', 'spearman'],
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
    args.no_model_name_dir_flag = False
    args.if_random = False
    args.no_img_name_dir_flag = True

    # if args.if_random:
    #     np.random.seed(0)

    # if args.num_variations is None:
    #     print('Please provide this number.\nExiting')
    #     sys.exit(0)
    # elif args.num_variations < 2:
    #     print('This number cant be less than 2.\nExiting')
    #     sys.exit(0)

    if args.method_name is None:
        print('Please provide the name of the method.\nExiting')
        sys.exit(0)

    if args.exp_num is None:
        print('Please provide the experiment number.\nExiting')
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
def compute_score(h1, h2, method, metric_name):

    if metric_name.lower() == 'ssim':
        if method.lower() == 'mp':
            data_range = 1
        else:
            data_range = 2
        out = ssim(h1, h2, data_range=data_range, win_size=5) #to match the implementation of Been KIM

    elif metric_name.lower() == 'hog':
        hog1 = hog(h1, pixels_per_cell=(16, 16))
        hog2 = hog(h2, pixels_per_cell=(16, 16))
        out, _ = pearsonr(hog1, hog2)


    elif metric_name.lower() == 'spearman':
        out, _ = spearmanr(h1, h2, axis=None)

    else:
        print(f'Still not implemented.\nExiting')
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
    eutils.mkdir_p(args.out_path)

    img_filenames = os.listdir(args.input_dir_path)
    img_filenames = [i for i in img_filenames if 'ILSVRC2012_val_000' in i and int(i.split('_')[-1]) in range(1, 50001)]
    if args.idx_flag == 1:
        img_filenames = img_filenames[0]

    # ## TODO: Chnages here
    # incorrect_img_list = np.load('/home/naman/CS231n/heatmap_tests/Madri/Madri_New/'
    #                              'robustness_applications/img_name_files/incorrect_img_names.npy').tolist()

    ##############################################################
    model_names = []
    model_names.append('madry_googlenet')
    model_names.append('googlenet')
    model_names.append('pytorch')
    model_names.append('madry')
    print(model_names)


    mean_dict = {'pytorch': [],
                 'googlenet': [],
                 'madry': [],
                 'madry_googlenet': []} ## TODO: Chnages here

    var_dict = deepcopy(mean_dict)
    output = deepcopy(mean_dict)

    # load_model_fns = {'pytorch': eval('eutils.load_orig_imagenet_model'),
    #                   'madry': eval('eutils.load_madry_model'),
    #                   'googlenet': eval('eutils.load_orig_imagenet_model')}
    # load_model_args = {'pytorch': 'resnet50', 'madry': 'madry', 'googlenet': 'googlenet'}

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

    # if method_name.lower() in ['occlusion', 'mp', 'lime']:
    #     rescale_flag = False
    # elif method_name.lower() in ['grad', 'inpgrad', 'ig', 'smoothgrad']:
    #     rescale_flag = True

    print(f'Rescale flag is {rescale_flag}')

    # img_filenames = []
    # with open(text_file, 'r') as f:
    #     img_filenames = f.read().splitlines()
    #     img_filenames = img_filenames[args.start_idx:args.end_idx]
    # if args.idx_flag == 1:
    #     img_filenames = img_filenames[0]

    ## TODO: Chnages here
    img_file_numbers = np.array([int(imName.split('_')[-1]) for imName in img_filenames],
                                dtype=int)

    print(f'Metric: {metric_name}')
    print(f'Method: {method_name}')

    # ## TODO: Chnages here
    # num_inc_imgs = len([i for i in img_filenames if i in incorrect_img_list])

    for modelIdx, model_name in enumerate(model_names):
        ## TODO: Chnages here
        temp_mean = np.zeros(len(img_filenames), dtype=float)
        temp_std = np.zeros(len(img_filenames), dtype=float) ## TODO: Chnages here (Name chnages)
        modelTime = time.time()
        print(f'Calculation variation for model: {model_name}')

        ## TODO: Chnages here
        idx = -1
        for _, img_name in enumerate(img_filenames):
            idx += 1

            if len(img_filenames) < 2000:
                print(f'Calculation variation across img: {img_name}, img_number: {idx:04d}')

            if args.no_model_name_dir_flag:
                dir_name = os.path.join(args.input_dir_path,
                                        f"{method_name}_{model_name}")
            else:
                dir_name = args.input_dir_path


            if method_name.lower() == 'lime':
                npy_str_lists = glob.glob(os.path.join(dir_name,
                                                       f'{img_name}/time_*var_0.1_{model_name}.npy'))

            elif method_name.lower() == 'mp':
                im_num = int(img_name.split('_')[-1])
                npy_str_lists = glob.glob(os.path.join(dir_name,
                                                      f'{img_name}/*{im_num:05d}_*{model_name}.npy'))

                if model_name.lower() == 'googlenet':
                    npy_str_lists = [i for i in npy_str_lists if 'madry_googlenet' not in i]

                num_var = len(npy_str_lists)

                npy_str_lists.sort()

            elif method_name.lower() == 'smoothgrad':
                npy_str_lists = glob.glob(os.path.join(dir_name,
                                                       f'{img_name}/*var_0.1*{model_name}.npy'))

                if model_name.lower() == 'googlenet':
                    npy_str_lists = [i for i in npy_str_lists if 'madry_googlenet' not in i]
                elif model_name.lower() == 'madry':
                    npy_str_lists = [i for i in npy_str_lists if 'madry_googlenet' not in i]

                num_var = len(npy_str_lists)

                if args.exp_num.lower() == 'a30':
                    baseline = [i for i in npy_str_lists if '_num_samples_50_' in i]
                    npy_str_lists = [i for i in npy_str_lists if '_num_samples_50_' not in i]
                    assert num_var - 1 == 4, 'Incorrect input path. Check your code'
                elif args.exp_num.lower() == 'a31':
                    baseline = [i for i in npy_str_lists if '_stdev_spread_0.2' in i]
                    npy_str_lists = [i for i in npy_str_lists if '_stdev_spread_0.2' not in i]
                    assert num_var - 1 == 2, 'Incorrect input path. Check your code'

                npy_str_lists.sort()


            else:
                npy_str_lists = glob.glob(os.path.join(dir_name,
                                                      f'{img_name}/*_{img_name}_*model_name_'
                                                      f'{model_name}_batch_idx*.npy'))

            assert len(baseline) == 1, 'Somthing is wrong with the baseline file'


            # print(f'Loading the results')
            heatmap_list = [np.load(i) for i in npy_str_lists]
            baseline_heatmap = [np.load(i) for i in baseline]

            if rescale_flag:
                ## Rescale the heatmaps to the original shape
                mVals = [max(np.abs(hMap.max()), np.abs(hMap.min())) for hMap in heatmap_list]
                heatmap_list = [hMap/mVals[hI] for hI, hMap in enumerate(heatmap_list)]

                mVals = [np.max(np.abs(hMap)) for hMap in baseline_heatmap]
                baseline_heatmap = [hMap / mVals[hI] for hI, hMap in enumerate(baseline_heatmap)]


            if method_name.lower() == 'occlusion':
                ## I can resize the input to the image size regardless of the input shape
                ## If input size is 224, there would not be any chnage in the output
                heatmap_list = [resize(i, (224, 224), order=5) for i in heatmap_list]
                baseline_heatmap = [resize(i, (224, 224), order=5) for i in baseline_heatmap]

            # if args.if_random:
            #     if method_name.lower() == 'mp':
            #         heatmap_list = [np.random.uniform(low=0, high=1, size=(i.shape)).astype(i.dtype) for i in heatmap_list]
            #     else:
            #         heatmap_list = [np.random.uniform(low=-1, high=1, size=(i.shape)).astype(i.dtype) for i in heatmap_list]

            # req_comb = list(combinations(range(num_var),2))
            scores = []
            for _, hMap in enumerate(heatmap_list):
            # for i1, i2 in req_comb:
                try:
                    scores.append(compute_score(baseline_heatmap[0], hMap, method_name, metric_name))
                except:
                    scores = [math.nan]

            if np.isnan(scores).any():
                temp_mean[idx] = math.nan
                temp_std[idx] = math.nan
                continue

            # ipdb.set_trace()
            mean_dict[model_name].append(np.mean(scores))
            var_dict[model_name].append(np.var(scores))

            temp_mean[idx] = np.mean(scores)
            temp_std[idx] = np.std(scores) ## TODO: Chnages here

        print(f'Len of samples considered is {len(mean_dict[model_name])}')
        output[model_name].append(np.mean(mean_dict[model_name]))
        output[model_name].append(np.sqrt(np.mean(var_dict[model_name]))) ## TODO: Chnages here

        print(f'Mean is {output[model_name][0]}, std is {output[model_name][1]}') ## TODO: Chnages here
        print(f'Time taken to evaluate {metric_name} metric for '
              f'model {model_name} on method {method_name} is {time.time() - modelTime}')


        if len(img_filenames) >= 1:
            ##Save the results to the text file
            path = os.path.join(args.out_path,
                                f'Method_{method_name}_Metric_{metric_name}')
            eutils.mkdir_p(path)

            if args.if_random:
                fName = os.path.join(path, f'time_{f_time}_'
                                           f'Random_Baseline_{method_name}_{metric_name}.txt')
            else:
                fName = os.path.join(path, f'time_{f_time}_'
                                           f'Model_{model_name}_{method_name}_{metric_name}.txt')

            file_handle = open(fName, 'ab')
            temp_arr = np.concatenate((np.expand_dims(img_file_numbers, axis=-1),
                                       np.expand_dims(temp_mean, axis=-1),
                                       np.expand_dims(temp_std, axis=-1),
                                       ), axis=-1)

            ## TODO: Chnages here
            np.savetxt(file_handle, temp_arr, fmt='%05d,  %.16f,  %.16f',
                       header='ImNum,  Mean              ,  Std', footer='\nCumulative Results', comments='', ## TODO: Chnages here
                       )
            temp_arr = np.concatenate((np.array([[len(mean_dict[model_name])]]),
                                       np.array([[output[model_name][0]]]),
                                       np.array([[output[model_name][1]]]),
                                       ), axis=-1)
            np.savetxt(file_handle, temp_arr,
                       fmt='%05d,  %.16f,  %.16f',
                       header='ImCou,  Mean              ,  Std', comments='',) ## TODO: Chnages here
            file_handle.close()

    ## TODO: Chnages here
    ## Saving the cumulative results
    if len(model_names) == 3:
        if len(img_filenames) >= 10:
            path = os.path.join(args.out_path,
                                f'Method_{method_name}_Metric_{metric_name}')
            eutils.mkdir_p(path)
            fName = os.path.join(path, f'time_{f_time}_'
                                       f'cumulative_results.txt')
            file_handle = open(fName, 'ab')

            ## TODO: Chnages here
            gNet_row = np.asarray((['GoogleNet'] + [len(mean_dict['googlenet'])] + output['googlenet']), dtype=object).reshape(1, -1)
            rNet_row = np.asarray((['ResNet50'] + [len(mean_dict['pytorch'])] + output['pytorch']), dtype=object).reshape(1, -1)
            mNet_row = np.asarray((['Madry (Res)'] + [len(mean_dict['madry'])] + output['madry']), dtype=object).reshape(1, -1)
            temp_arr = np.concatenate((gNet_row,
                                       rNet_row,
                                       mNet_row,
                                       ),
                                      axis=0)
            np.savetxt(file_handle, temp_arr,
                       fmt='%-11s,  %05d,  %.16f,  %.16f',
                       header='Network    ,  ImCou,  Mean              ,  Std', comments='', ) ## TODO: Chnages here
            file_handle.close()
            ## TODO: Till here

    print(f'Time taken is {time.time() - s_time}')
    print(f'Time stamp is {f_time}')



