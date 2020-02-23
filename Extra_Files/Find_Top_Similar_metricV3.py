import ipdb, os, time, argparse, sys, glob, skimage
from srblib import abs_path

import sys; sys.path.append("..")
import utils as eutils

import numpy as np
from copy import deepcopy
from functools import reduce
from PIL import Image
from torchvision.transforms import transforms

from skimage.measure import compare_ssim as ssim

import warnings
warnings.filterwarnings("ignore")


def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Paramters for sensitivity analysis of heatmaps')

    # parser.add_argument('-idp', '--input_dir_path', help='Path of the image directory', metavar='DIR')

    parser.add_argument('-op', '--out_path',
                        help='Path of the output directory where you want to save the results (Default is ./)')

    # parser.add_argument('-mn', '--method_name', choices=['occlusion', 'ig', 'sg', 'grad', 'lime', 'mp', 'inpgrad'],
    #                     help='Method you are analysing')

    parser.add_argument('--save', action='store_true', default=False,
                        help=f'Flag to say that plot need to be saved. '
                             f'Default=False')

    parser.add_argument('-s_idx', '--start_idx', type=int,
                        help='Start index for selecting images. Default: 0', default=0,
                        )

    parser.add_argument('-e_idx', '--end_idx', type=int,
                        help='End index for selecting images. Default: 2K', default=2000,
                        )

    # parser.add_argument('--if_noise', action='store_true',
    #                     help='Flag whether noise was present in the input while calculating the heatmaps. Default: False',
    #                     default=False,
    #                     )

    # Parse the arguments
    args = parser.parse_args()

    args.method_name = 'occlusion'
    # # args.model_name_dir_flag = False
    # args.img_name_dir_flag = True
    # args.if_noise = False

    # if args.input_dir_path is None:
    #     print('Please provide image dir path. Exiting')
    #     sys.exit(1)
    # args.input_dir_path = abs_path(args.input_dir_path)

    if args.out_path is None:
        args.out_path = './'
    args.out_path = os.path.abspath(args.out_path)

    if args.method_name is None:
        print('Please provide the name of the method.\nExiting')
        sys.exit(0)

    return args


########################################################################################################################
def sort_fun(elem):
    key_val = elem.split(',')[1].strip()
    try:
        aa = float(key_val)
        return aa
    except:
        print(f'Something is wrong. May be because of Nan values.\nExiting')
        sys.exit(1)


########################################################################################################################
if __name__ == '__main__':
    s_time = time.time()
    f_time = ''.join(str(s_time).split('.'))
    args = get_arguments()
    im_label_map = eutils.imagenet_label_mappings()

    base_img_dir = '/home/naman/CS231n/heatmap_tests/images/ILSVRC2012_img_val'
    img_text_file = f'/home/naman/CS231n/heatmap_tests/' \
                f'Madri/Madri_New/robustness_applications/img_name_files/' \
                f'time_15669152608009198_seed_0_' \
                f'common_correct_imgs_model_names_madry_ressnet50_googlenet.txt'

    img_filenames = []
    with open(img_text_file, 'r') as f:
        img_filenames = f.read().splitlines()
        img_filenames = img_filenames[args.start_idx:args.end_idx]


    empty_dict = {'pytorch': [],
                  'googlenet': [],
                  'madry': [],
                  'madry_googlenet': [],
                  }

    num_settings = 5

    insertion_dicts = {'Setting_Idx_0': {'img_dict': deepcopy(empty_dict), 'mean_dict': deepcopy(empty_dict)},
                       'Setting_Idx_1': {'img_dict': deepcopy(empty_dict), 'mean_dict': deepcopy(empty_dict)},
                       'Setting_Idx_2': {'img_dict': deepcopy(empty_dict), 'mean_dict': deepcopy(empty_dict)},
                       'Setting_Idx_3': {'img_dict': deepcopy(empty_dict), 'mean_dict': deepcopy(empty_dict)},
                       'Setting_Idx_4': {'img_dict': deepcopy(empty_dict), 'mean_dict': deepcopy(empty_dict)},
                       }

    deletion_dicts = {'Setting_Idx_0': {'img_dict': deepcopy(empty_dict), 'mean_dict': deepcopy(empty_dict)},
                      'Setting_Idx_1': {'img_dict': deepcopy(empty_dict), 'mean_dict': deepcopy(empty_dict)},
                      'Setting_Idx_2': {'img_dict': deepcopy(empty_dict), 'mean_dict': deepcopy(empty_dict)},
                      'Setting_Idx_3': {'img_dict': deepcopy(empty_dict), 'mean_dict': deepcopy(empty_dict)},
                      'Setting_Idx_4': {'img_dict': deepcopy(empty_dict), 'mean_dict': deepcopy(empty_dict)},
                      }


    ###############################################
    model_names = []
    model_names.append('googlenet')
    model_names.append('pytorch')
    model_names.append('madry')
    model_names.append('madry_googlenet')

    method_dict = {'grad': 'Grad',
                   'inpgrad': 'InpGrad',
                   'ig': 'IG',
                   'lime': 'Lime',
                   'mp': 'MP',
                   'occlusion': 'Occlusion',
                   'sg': 'SmoothGrad',
                   }
    method_name = method_dict[args.method_name]

    insertion_path = '/home/naman/CS231n/heatmap_tests/Madri/Madri_New/' \
                     'robustness_applications/metric_V3/Occlusion/' \
                     'Occlusion_Big_Patches/Method_Occlusion_Exp_Num_a02_Metric_insertion'
    deletion_path = '/home/naman/CS231n/heatmap_tests/Madri/Madri_New/' \
                    'robustness_applications/metric_V3/Occlusion/' \
                    'Occlusion_Big_Patches/Method_Occlusion_Exp_Num_a02_Metric_deletion'


    # input_dir_name = args.input_dir_path.split('/')[-1]
    #
    # metric_file_paths = [f'Method_{method_name}_Metric_{i}'
    #                      for i in ['ssim',]]
    #
    # assert metric_file_paths[0] == input_dir_name, \
    #     'Something is wrong with the input path'
    for path, metric_name, data_dicts in zip([insertion_path, deletion_path],
                                             ['insertion', 'deletion'],
                                             [insertion_dicts, deletion_dicts]):
        for settingIdx in range(num_settings):
            input_path = os.path.join(path, f'Setting_Idx_{settingIdx}')

            ## List for asserting the order of read
            order_list = ['googlenet', 'madry', 'madry_googlenet', 'pytorch']

            txt_data_files = glob.glob(os.path.join(input_path,
                                                    f'time*_Model_*_{method_name}_{metric_name}*.txt'))
            txt_data_files.sort()

            ## Data read
            print(f'Reading Data ... ', end='')
            for modelIdx, model_name in enumerate(order_list):
                txt_file = txt_data_files[modelIdx]

                assert model_name in txt_file.split('/')[-1].split('_Model_')[-1].split(f'_{method_name}_'), \
                    'Something wrong with the reading of data. Check'


                with open(txt_file, 'r') as f:
                    data_list = f.read().splitlines()
                    data_list = data_list[1:1736]
                    aa = [i.split(',') for i in data_list]
                    [(data_dicts[f'Setting_Idx_{settingIdx}']['img_dict'][model_name].append(int(i[0].strip())),
                      data_dicts[f'Setting_Idx_{settingIdx}']['mean_dict'][model_name].append(float(i[1].strip())))
                     for i in aa]

                data_dicts[f'Setting_Idx_{settingIdx}']['mean_dict'][model_name] = np.asarray(data_dicts[f'Setting_Idx_{settingIdx}']['mean_dict'][model_name])

            assert data_dicts[f'Setting_Idx_{settingIdx}']['img_dict']['googlenet'] == data_dicts[f'Setting_Idx_{settingIdx}']['img_dict']['pytorch'], \
                'Something is wrong'

            assert data_dicts[f'Setting_Idx_{settingIdx}']['img_dict']['googlenet'] == data_dicts[f'Setting_Idx_{settingIdx}']['img_dict']['madry'], \
                'Something is wrong'

            assert data_dicts[f'Setting_Idx_{settingIdx}']['img_dict']['googlenet'] == data_dicts[f'Setting_Idx_{settingIdx}']['img_dict']['madry_googlenet'], \
                'Something is wrong'



    img_idx_array = np.asarray(data_dicts[f'Setting_Idx_0']['img_dict']['googlenet']).astype(int)

    ## SettionIdx in Occlusion correspondance
    ## {0: Patch 5, 1: Patch 17, 2: Patch 29, 3: Patch 41, 4: Patch 53}
    for settingIdx in [2, ]:

        insertion_gNet = insertion_dicts[f'Setting_Idx_{settingIdx}']['mean_dict']['googlenet']
        sorted_insertion_gNet_idx = np.flip(np.argsort(insertion_gNet))  ## High to low
        sorted_insertion_gNet_img_name = img_idx_array[sorted_insertion_gNet_idx]
        sorted_insertion_gNet_vals = insertion_gNet[sorted_insertion_gNet_idx]

        deletion_gNet = deletion_dicts[f'Setting_Idx_{settingIdx}']['mean_dict']['googlenet']
        sorted_deletion_gNet_idx = np.flip(np.argsort(deletion_gNet)) ## High to Low
        sorted_deletion_gNet_img_name = img_idx_array[sorted_deletion_gNet_idx]
        sorted_deletion_gNet_vals = insertion_gNet[sorted_deletion_gNet_idx]

        ## Top 10
        top_imgs, gNet_idx, rNet_idx = np.intersect1d(sorted_insertion_gNet_img_name[:50],  ## Images with best (high) insertion value
                                                      sorted_deletion_gNet_img_name[:50], ## Images with worst (high) deletion values
                                                      assume_unique=True, return_indices=True)
        ipdb.set_trace()

        ### Code not completed afterwards

        if args.save:
            path = os.path.join(args.input_dir_path,
                                f'TOP_Results')
            eutils.mkdir_p(path)


        ## Top 100
        top_imgs, gNet_idx, rNet_idx = np.intersect1d(sorted_rNet_img_name[:100],
                                                      sorted_rNet_img_name[:100],
                                                      assume_unique=True, return_indices=True)

        sorted_ssim_score_gNet = sorted_ssim_diff_gNet[:100][gNet_idx]
        sorted_ssim_score_rNet = sorted_ssim_diff_rNet[:100][rNet_idx]

        # temp_arr = np.concatenate((top_imgs.reshape(-1, 1),
        #                            sorted_ssim_score_gNet.reshape(-1, 1),
        #                            sorted_ssim_score_rNet.reshape(-1, 1),
        #                            ), axis=-1)
        ipdb.set_trace()
        top_imgs = sorted_rNet_img_name[:50]
        temp_arr = np.array(top_imgs.reshape(-1, 1))

        fName = os.path.join(path, f'time_{f_time}_'
                                   f'top_100_images_with_mrNet_diff_of_ssim_scores.txt')
        file_handle = open(fName, 'ab')

        # np.savetxt(file_handle, temp_arr, fmt='%05d,  %.16f,  %.16f',
        #            header='ImCou,  GoogleNet         ,  ResNet ',
        #            comments='')
        np.savetxt(file_handle, temp_arr, fmt='%05d',
                   comments='')
        file_handle.close()




    print(f'\nTime taken is {time.time() - s_time}')
    print(f'Time stamp is {f_time}\n')

########################################################################################################################




