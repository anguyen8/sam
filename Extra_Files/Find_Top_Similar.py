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

    parser.add_argument('-idp', '--input_dir_path', help='Path of the image directory', metavar='DIR')

    parser.add_argument('-op', '--out_path',
                        help='Path of the output directory where you want to save the results (Default is ./)')

    parser.add_argument('-mn', '--method_name', choices=['occlusion', 'ig', 'sg', 'grad', 'lime', 'mp', 'inpgrad'],
                        help='Method you are analysing')

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

    # # args.model_name_dir_flag = False
    # args.img_name_dir_flag = True
    # args.if_noise = False

    if args.input_dir_path is None:
        print('Please provide image dir path. Exiting')
        sys.exit(1)
    args.input_dir_path = abs_path(args.input_dir_path)

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

    ssim_dicts = {'img_dict': deepcopy(empty_dict), 'mean_dict': deepcopy(empty_dict)}


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

    input_dir_name = args.input_dir_path.split('/')[-1]

    metric_file_paths = [f'Method_{method_name}_Metric_{i}'
                         for i in ['ssim',]]

    assert metric_file_paths[0] == input_dir_name, \
        'Something is wrong with the input path'

    ## List for asserting the order of read
    order_list = ['googlenet', 'madry', 'madry_googlenet', 'pytorch']
    input_path = args.input_dir_path
    metric_name = input_path.split('/')[-1].split('_')[-1]
    txt_data_files = glob.glob(os.path.join(input_path,
                                            f'*_Model_*_{method_name}_{metric_name}*.txt'))
    txt_data_files.sort()


    ## Data read
    print(f'Reading Data ... ', end='')
    for modelIdx, model_name in enumerate(order_list):
        txt_file = txt_data_files[modelIdx]
        # assert model_name in txt_file.split('/')[-1].split('_'), \
        #     'Something wrong with the reading of data. Check'

        assert model_name in txt_file.split('/')[-1].split('_Model_')[-1].split(f'_{method_name}_'), \
            'Something wrong with the reading of data. Check'


        with open(txt_file, 'r') as f:
            data_list = f.read().splitlines()
            data_list = data_list[1:1736]
            aa = [i.split(',') for i in data_list]
            [(ssim_dicts['img_dict'][model_name].append(int(i[0].strip())),
              ssim_dicts['mean_dict'][model_name].append(float(i[1].strip())))
             for i in aa]

    assert ssim_dicts['img_dict']['googlenet'] == ssim_dicts['img_dict']['pytorch'], \
        'Something is wrong'

    assert ssim_dicts['img_dict']['googlenet'] == ssim_dicts['img_dict']['madry'], \
        'Something is wrong'

    assert ssim_dicts['img_dict']['googlenet'] == ssim_dicts['img_dict']['madry_googlenet'], \
        'Something is wrong'

    img_idx_array = np.asarray(ssim_dicts['img_dict']['googlenet']).astype(int)
    for key in ssim_dicts['mean_dict']:
        ssim_dicts['mean_dict'][key] = np.asarray(ssim_dicts['mean_dict'][key])

    gNet_diff = ssim_dicts['mean_dict']['madry_googlenet'] - ssim_dicts['mean_dict']['googlenet']
    rNet_diff = ssim_dicts['mean_dict']['madry'] - ssim_dicts['mean_dict']['pytorch']
    ipdb.set_trace()
    if args.save:
        sorted_ssim_diff_gNet_idx = np.flip(np.argsort(gNet_diff))
        sorted_gNet_img_name = img_idx_array[sorted_ssim_diff_gNet_idx]
        sorted_ssim_diff_gNet = gNet_diff[sorted_ssim_diff_gNet_idx]

        sorted_ssim_diff_rNet_idx = np.flip(np.argsort(rNet_diff))
        sorted_rNet_img_name = img_idx_array[sorted_ssim_diff_rNet_idx]
        sorted_ssim_diff_rNet = rNet_diff[sorted_ssim_diff_rNet_idx]

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

        # ## Bottom 100
        # top_imgs, gNet_idx, rNet_idx = np.intersect1d(sorted_gNet_img_name[-100:],
        #                                               sorted_rNet_img_name[-100:],
        #                                               assume_unique=True, return_indices=True)


        # sorted_ssim_score_gNet = sorted_ssim_diff_gNet[-100:][gNet_idx]
        # sorted_ssim_score_rNet = sorted_ssim_diff_rNet[-100:][rNet_idx]

        # temp_arr = np.concatenate((top_imgs.reshape(-1, 1),
        #                            sorted_ssim_score_gNet.reshape(-1, 1),
        #                            sorted_ssim_score_rNet.reshape(-1, 1),
        #                            ), axis=-1)

        # fName = os.path.join(path, f'time_{f_time}_'
        #                            f'bottom_100_images_with_diff_of_ssim_scores.txt')
        # file_handle = open(fName, 'ab')

        # np.savetxt(file_handle, temp_arr, fmt='%05d,  %.16f,  %.16f',
        #            header='ImCou,  GoogleNet         ,  ResNet ',
        #            comments='')
        # file_handle.close()



    print(f'\nTime taken is {time.time() - s_time}')
    print(f'Time stamp is {f_time}\n')

########################################################################################################################




