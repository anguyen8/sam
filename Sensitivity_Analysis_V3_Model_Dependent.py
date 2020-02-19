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

def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Paramters for sensitivity analysis of heatmaps')

    parser.add_argument('-idp', '--input_dir_path', help='Path of the input directory', metavar='DIR')

    parser.add_argument('-op', '--out_path',
                        help='Path of the output directory where you want to save the results (Default is ./img_name/)')

    parser.add_argument('-mn', '--method_name',
                        choices=['occlusion', 'ig', 'sg', 'grad', 'lime', 'mp', 'inpgrad'],
                        help='Method you are analysing')

    parser.add_argument('--occlusion_exp_num', choices=['a01', 'a02', 'a03'],
                        help=f'Which experiment of Occlusion'
                             f'a01 - Patch 5, 6, 7'
                             f'a02 - Patch - 5, 17, 29, 41, 53'
                             f'a03 - Patch - 52, 53, 54')

    parser.add_argument('--lime_exp_num', choices=['a01', 'a02', 'a04'],
                        help='Which experiment of LIME')

    parser.add_argument('--sg_exp_num', choices=['a30', 'a31'],
                        help='Experiment number of SG',
                        )

    parser.add_argument('--mp_exp_num', choices=['a20', 'a21', 'a22', 'a23'],
                        help='Which experiment of MP')

    parser.add_argument('--metric_name', choices=['insertion', 'deletion'],
                        help='Metric to be computed')

    # parser.add_argument('--num_variations', type=int,
    #                     help='Number of variations for a particular method.')

    parser.add_argument('--no_img_name_dir_flag', action='store_false', default=True,
                        help=f'Flag to say that image name is stored as seperate directory in the input path.'
                             f'Default=True')

    parser.add_argument('--no_model_name_dir_flag', action='store_false', default=True,
                        help=f'Flag to say that model name is stored as seperate directory in the input path. '
                             f'Default=True')

    parser.add_argument('--idx_flag', type=int, choices=range(2),
                        help=f'Flag whether to use some images in the folder (1) or all (0). '
                             f'This is just for testing purposes. '
                             f'Default=0', default=0,
                        )

    parser.add_argument('-s_idx', '--start_idx', type=int,
                        help='Start index for selecting images. Default: 0', default=0,
                        )

    parser.add_argument('-e_idx', '--end_idx', type=int,
                        help='End index for selecting images. Default: 2K', default=2000,
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
    args.if_random = False
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
    elif args.method_name == 'occlusion':
        args.exp_num = args.occlusion_exp_num
    elif args.method_name == 'lime':
        args.exp_num = args.lime_exp_num
    elif args.method_name == 'sg':
        args.exp_num = args.sg_exp_num
    elif args.method_name == 'mp':
        args.exp_num = args.mp_exp_num
    else:
        print(f'Somethis is wrong in the input arguments.\nExiting')
        sys.exit(0)

    if args.exp_num is None:
        print(f'Please provide exp_num\nExiting')
        sys.exit(0)

    if args.metric_name is None:
        print('Please provide the name of the metric.\nExiting')
        sys.exit(0)

    if args.input_dir_path is None:
        print('Please provide image dir path. Exiting')
        sys.exit(0)
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
        dir = abs_path(f'./temp_results/Metric_{metric_name}/Model_{model_name}')
        eutils.mkdir_p(dir)
        ## We are computing percentage AUC
        # if kwargs['heatmap_idx'] == 2:
        aa = metricObj.single_run(img, heat, verbose=3, save_to=kwargs['save_path']) #, verbose=1, save_to=dir)
        # else:
        #     aa = metricObj.single_run(img, heat) #, verbose=1, save_to=dir)

        out = auc(aa)

        # if kwargs['heatmap_idx'] == 2:
        #     ipdb.set_trace()

    else:
        print(f'This metric  has still not been implemented.\nExiting')
        sys.exit(0)

    return out


########################################################################################################################
def sort_fun_selection(method, exp_name):
    #################################
    if method.lower() == 'occlusion':
        def sort_fun(elem):
            return int(elem.split('/')[-1].split('_patch_size_')[-1].split('_')[0])
    #################################
    elif method.lower() == 'lime':
        if exp_name == 'a01':
            def sort_fun(elem):
                return int(elem.split('/')[-1].split('_sample_count_')[-1].split('_')[0])
        elif exp_name == 'a02':
            def sort_fun(elem):
                return int(elem.split('/')[-1].split('_explainer_seed_')[-1].split('_')[0])
        elif exp_name == 'a04':
            def sort_fun(elem):
                return int(elem.split('/')[-1].split('_explainer_seed_')[-1].split('_')[0])
    #################################
    elif method.lower() == 'smoothgrad':
        if exp_name.lower() == 'a30':
            def sort_fun(elem):
                try:
                    key_val = int(elem.split('/')[-1].split('_num_samples_')[-1].split('_')[0])
                except:
                    key_val = 0
                return key_val
        elif exp_name.lower() == 'a31':
            def sort_fun(elem):
                try:
                    key_val = float(elem.split('/')[-1].split('_stdev_spread_')[-1].split('_')[0])
                except:
                    key_val = 0
                return key_val
    #################################
    elif method.lower() == 'mp':
        if exp_name.lower() == 'a20':
            def sort_fun(elem):
                key_val = elem.split('/')[-1].split('_mT_')[-1].split('_')[0]
                if key_val.lower() == 'ones':
                    out = 1
                elif key_val.lower() == 'circ':
                    out = 2
                elif key_val.lower() == 'rand':
                    out = 3
                else:
                    print(f'Key not implemented.\nExiting')
                    sys.exit(0)
                return out

        elif exp_name.lower() == 'a21':
            def sort_fun(elem):
                return int(elem.split('/')[-1].split('_iter_')[-1].split('_')[0])

        elif exp_name.lower() == 'a22':
            def sort_fun(elem):
                return elem.split('/')[-1].split('_blurR_')[-1].split('_')[0]
        elif exp_name.lower() == 'a23':
            def sort_fun(elem):
                return elem.split('/')[-1].split('_seed_')[-1].split('_')[0]
    #################################
    else:
        print(f'Sorting function not implemented.\nExiting')
        sys.exit(0)

    return sort_fun


########################################################################################################################
if __name__ == '__main__':
    base_img_dir = '/home/naman/CS231n/heatmap_tests/images/ILSVRC2012_img_val'
    text_file = f'/home/naman/CS231n/heatmap_tests/' \
                f'Madri/Madri_New/robustness_applications/img_name_files/' \
                f'time_15669152608009198_seed_0_' \
                f'common_correct_imgs_model_names_madry_ressnet50_googlenet.txt'
    s_time = time.time()
    f_time = ''.join(str(s_time).split('.'))
    args = get_arguments()
    im_label_map = eutils.imagenet_label_mappings()
    my_attacker = True
    print(f'My_Attacker is: {my_attacker}')

    eutils.mkdir_p(args.out_path)

    incorrect_img_list = np.load('/home/naman/CS231n/heatmap_tests/Madri/Madri_New/'
                                 'robustness_applications/img_name_files/'
                                 'incorrect_img_names.npy').tolist()

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

    #############################################
    ## #Inits

    model_names = []
    model_names.append('pytorch')
    model_names.append('madry')
    model_names.append('googlenet')
    model_names.append('madry_googlenet')

    print(model_names)
    print(f'If random is : {args.if_random}')

    sort_fun = sort_fun_selection(method_name.lower(), args.exp_num)

    data_dict = {'pytorch': [],
                 'googlenet': [],
                 'madry': [],
                 'madry_googlenet': []}

    value_dict = {'pytorch': 0,
                 'googlenet': 0,
                 'madry': 0,
                 'madry_googlenet': 0}

    if method_name.lower() == 'occlusion':
        if args.exp_num == 'a01':
            num_settings = 3
        elif args.exp_num == 'a02':
            num_settings = 5
        elif args.exp_num == 'a03':
            num_settings = 3
    elif method_name.lower() == 'lime':
        if args.exp_num == 'a01':
            num_settings = 2
        elif args.exp_num == 'a02':
            num_settings = 5
        elif args.exp_num == 'a04':
            num_settings = 1 ## Depending on the number of seeds you want
    elif method_name.lower() == 'smoothgrad':
        if args.exp_num == 'a30':
            num_settings = 5
        elif args.exp_num == 'a31':
            num_settings = 3
    elif method_name.lower() == 'mp':
        if args.exp_num == 'a20':
            num_settings = 3
        elif args.exp_num == 'a21':
            num_settings = 4
        elif args.exp_num == 'a22':
            num_settings = 3
        elif args.exp_num == 'a23':
            num_settings = 5
    else:
        print(f'Not implemented.\nExiting')
        sys.exit(0)


    metric_val_list = [deepcopy(data_dict) for i in range(num_settings)]
    mean_metric_val_list = [deepcopy(value_dict) for i in range(num_settings)]

    load_model_fns = {'pytorch': eval('eutils.load_orig_imagenet_model'),
                      'madry': eval('eutils.load_madry_model'),
                      'googlenet': eval('eutils.load_orig_imagenet_model'),
                      'madry_googlenet': eval('eutils.load_madry_model'),
                      }
    load_model_args = {'pytorch': 'resnet50',
                       'madry': 'madry',
                       'madry_googlenet': 'madry_googlenet',
                       'googlenet': 'googlenet'}



    if method_name.lower() in ['occlusion', 'mp']:
        rescale_flag = False
    elif method_name.lower() in ['grad', 'inpgrad', 'ig', 'lime', 'smoothgrad']:
        rescale_flag = True

    print(f'Rescale flag is {rescale_flag}')

    #############################################

    img_filenames = []
    with open(text_file, 'r') as f:
        img_filenames = f.read().splitlines()
        img_filenames = img_filenames[args.start_idx:args.end_idx]
    if args.idx_flag == 1:
        img_filenames = img_filenames[0]
    # img_filenames.sort()
    img_file_numbers = np.array([int(imName.split('_')[-1]) for imName in img_filenames
                                 if imName not in incorrect_img_list],
                                dtype=int)
    # img_file_numbers = np.array([int(imName.split('_')[-1]) for imName in img_filenames], dtype=int)

    print(f'Metric: {metric_name}')
    print(f'Method: {method_name}')

    num_inc_imgs = len([i for i in img_filenames if i in incorrect_img_list])

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

        if args.if_random:
            args.random_state = np.random.RandomState(0)

        # temp_mean = np.zeros(len(img_filenames) - num_inc_imgs, dtype=float)
        # temp_std = np.zeros(len(img_filenames) - num_inc_imgs, dtype=float)
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
            pixels_removed = 224*8
            print(f'Pixels removed are: {pixels_removed}')
            metricObj = CausalMetric(model, metric_name[:3], pixels_removed, substrate_fn=substrate_fn)
            kwargs['metricObj'] = metricObj

        idx = -1
        for _, img_name in enumerate(img_filenames):
            idx += 1

            if img_name in incorrect_img_list:
                idx -= 1
                continue

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

            #############################
            if method_name.lower() == 'lime':
                npy_str_lists = glob.glob(os.path.join(dir_name,
                                                       f'{img_name}/time_*{model_name}.npy'))

                ##Filtering the correct cases
                if model_name == 'googlenet':
                    npy_str_lists = [i for i in npy_str_lists if 'madry_googlenet' not in i]

                if model_name == 'madry':
                    npy_str_lists = [i for i in npy_str_lists if 'madry_googlenet' not in i]

                npy_str_lists.sort(key=sort_fun)
                if args.if_noise == 1:
                    noise_flag_list = [int(aa.split('_noise_')[1].split('_')[0]) for aa in npy_str_lists]
                    kwargs['noise_flag_list'] = noise_flag_list

                #############
                if args.exp_num == 'a04':
                    npy_str_lists = [i for i in npy_str_lists if 'explainer_seed_0' in i]
                ######
                num_var = len(npy_str_lists)

                if args.exp_num == 'a01':
                    assert num_var == 2, 'Num of variations has to be 2'
                elif args.exp_num == 'a02':
                    assert num_var == 5, 'Num of variations has to be 5'
                elif args.exp_num == 'a04':
                    assert num_var == 1, 'Num of variations has to be 1'


            elif method_name.lower() == 'mp':
                im_num = int(img_name.split('_')[-1])
                npy_str_lists = glob.glob(os.path.join(dir_name,
                                                      f'{img_name}/*{im_num:05d}_*{model_name}.npy'))

                ##Filtering the correct cases
                if model_name == 'googlenet':
                    npy_str_lists = [i for i in npy_str_lists if 'madry_googlenet' not in i]

                if model_name == 'madry':
                    npy_str_lists = [i for i in npy_str_lists if 'madry_googlenet' not in i]

                npy_str_lists.sort(key=sort_fun)
                num_var = len(npy_str_lists)

                if args.exp_num == 'a20':
                    assert num_var == 3, 'Num of variations has to be 3'
                elif args.exp_num == 'a21':
                    assert num_var == 4, 'Num of variations has to be 4'
                elif args.exp_num == 'a22':
                    assert num_var == 3, 'Num of variations has to be 3'
                elif args.exp_num == 'a23':
                    assert num_var == 5, 'Num of variations has to be 5'

                # ipdb.set_trace()

            elif method_name.lower() == 'occlusion':
                npy_str_lists = glob.glob(os.path.join(dir_name,
                                                       f'{img_name}/*_{img_name}_*{model_name}*.npy'))
                ##Filtering the correct cases
                if model_name == 'googlenet':
                    npy_str_lists = [i for i in npy_str_lists if 'madry_googlenet' not in i]

                if model_name == 'madry':
                    npy_str_lists = [i for i in npy_str_lists if 'madry_googlenet' not in i]

                npy_str_lists.sort(key=sort_fun)
                num_var = len(npy_str_lists)

                if args.exp_num == 'a01':
                    assert num_var == 3, 'Num of variations has to be 3'
                elif args.exp_num == 'a02':
                    assert num_var == 5, 'Num of variations has to be 5'
                elif args.exp_num == 'a03':
                    assert num_var == 3, 'Num of variations has to be 3'

            elif method_name.lower() == 'smoothgrad':
                npy_str_lists = glob.glob(os.path.join(dir_name,
                                                       f'{img_name}/*_{img_name}_*{model_name}*.npy'))
                ##Filtering the correct cases
                if model_name == 'googlenet':
                    npy_str_lists = [i for i in npy_str_lists if 'madry_googlenet' not in i]

                if model_name == 'madry':
                    npy_str_lists = [i for i in npy_str_lists if 'madry_googlenet' not in i]

                npy_str_lists.sort(key=sort_fun)
                num_var = len(npy_str_lists)

                if args.exp_num == 'a30':
                    assert num_var == 5, 'Num of variations has to be 5'
                elif args.exp_num == 'a31':
                    assert num_var == 3, 'Num of variations has to be 3'

            else:
                print(f'Not implemented.\nExiting')
                sys.exit(0)
                # npy_str_lists = glob.glob(os.path.join(dir_name,
                #                                       f'{img_name}/*_{img_name}_*{model_name}*.npy'))
                #
                # ##Filtering the correct cases
                # if model_name == 'googlenet':
                #     npy_str_lists = [i for i in npy_str_lists if 'madry_googlenet' not in i]
                #
                # if model_name == 'madry':
                #     npy_str_lists = [i for i in npy_str_lists if 'madry_googlenet' not in i]
                #
                # npy_str_lists.sort()
                # num_var = len(npy_str_lists)
                # if args.if_noise == 1:
                #     noise_flag_list = [int(aa.split('_noise_')[1].split('_')[0]) for aa in npy_str_lists]
                #     kwargs['noise_flag_list'] = noise_flag_list


            # assert num_var >= 2, 'Num of variations should be greater than 2'


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

            if args.if_random:
                if method_name.lower() == 'mp':
                    heatmap_list = [args.random_state.uniform(low=0, high=1, size=(i.shape)).astype(i.dtype)
                                    for i in heatmap_list]
                else:
                    heatmap_list = [args.random_state.uniform(low=-1, high=1, size=(i.shape)).astype(i.dtype)
                                    for i in heatmap_list]

            scores = []
            for hIdx, heatmap in enumerate(heatmap_list):
                kwargs['heatmap_idx'] = hIdx
                kwargs['save_path'] = os.path.abspath(f'./temp_results/Intermediate_Plots/{metric_name}/Setting_Idx_{hIdx}/{model_name}')
                scores.append(compute_score(heatmap, metric_name, **kwargs))

            assert np.isnan(scores).any() == False, 'There should nto be any NaNs'

            for scIdx, score in enumerate(scores):
                metric_val_list[scIdx][model_name].append(score)


        print(f'Len of samples considered is {len(metric_val_list[0][model_name])}')

        for setting in range(num_settings):
            mean_metric_val_list[setting][model_name] =  np.mean(metric_val_list[setting][model_name])

        print(f'For model {model_name}, mean value of metric across settings is {[i[model_name] for i in mean_metric_val_list]}')

        print(f'Time taken to evaluate {metric_name} metric for '
              f'model {model_name} on method {method_name} is {time.time() - modelTime}')


        if len(img_filenames) >= 10:
            print(f'Saving to the text file')
            ##Save the results to the text file
            for settingIdx in range(num_settings):
                setting_name = f'Setting_Idx_{settingIdx}' #+ '_'.join(npy_str_lists[settingIdx].split('/')[-1].split('.npy')[0].split('_')[2:])

                path = os.path.join(args.out_path,
                                    f'Method_{method_name}_Exp_Num_{args.exp_num}_Metric_{metric_name}/{setting_name}')
                eutils.mkdir_p(path)

                if args.if_random:
                    fName = os.path.join(path, f'time_{f_time}_'
                                               f'Random_Model_{model_name}_{method_name}_{metric_name}.txt')
                else:
                    fName = os.path.join(path, f'time_{f_time}_'
                                               f'Model_{model_name}_{method_name}_{metric_name}.txt')

                file_handle = open(fName, 'ab')

                temp_arr = np.concatenate((np.expand_dims(img_file_numbers, axis=-1),
                                           np.expand_dims(metric_val_list[settingIdx][model_name], axis=-1),
                                           ), axis=-1)

                np.savetxt(file_handle, temp_arr, fmt='%05d,  %.16f',
                           header='ImNum,  Mean              ', footer='\nCumulative Results', comments='',
                       )
                temp_arr = np.concatenate((np.array([[len(metric_val_list[settingIdx][model_name])]]),
                                           np.array([[mean_metric_val_list[settingIdx][model_name]]]),
                                           ), axis=-1)
                np.savetxt(file_handle, temp_arr,
                           fmt='%05d,  %.16f',
                           header='ImCou,  Mean              ', comments='', )
                file_handle.close()

    print(f'Time taken is {time.time() - s_time}')
    print(f'Time stamp is {f_time}')



