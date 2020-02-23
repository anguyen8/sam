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

    parser.add_argument('-mn', '--method_name', choices=['grad', 'inpgrad'],
                        help='Method you are analysing')

    parser.add_argument('--save', action='store_true', default=False,
                        help=f'Flag to say that plot need to be saved. '
                             f'Default=False')

    parser.add_argument('--if_same_comp', action='store_true', default=False,
                        help=f'Flag to say if you want to compare with the same method for Madry'
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

    # args.model_name_dir_flag = False
    args.img_name_dir_flag = True
    args.if_noise = False

    if args.out_path is None:
        args.out_path = './'
    args.out_path = os.path.abspath(args.out_path)

    if args.method_name is None:
        print('Please provide the name of the method.\nExiting')
        sys.exit(0)

    return args


########################################################################################################################
def preprocess_img(img):
    size = 224
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])

    preprocessed_img_tensor = transform(img).numpy()
    preprocessed_img_tensor = np.rollaxis(preprocessed_img_tensor, 0, 3)

    if preprocessed_img_tensor.shape[-1] == 1: #Black and white image
        preprocessed_img_tensor = preprocessed_img_tensor[:, :, 0]
    return preprocessed_img_tensor


########################################################################################################################
def sort_fun_selection(method, exp_name):
    if method.lower() == 'grad':
        def sort_fun(elem):
            key_val = elem.split('/')[-1].split('_mT_')[-1].split('_')[0]
    elif method_name.lower() == 'inpgrad':
        pass
    else:
        print(f'Not implemented')
    return sort_fun


########################################################################################################################
def compute_score(h1, h2, method):
    if method.lower() == 'mp':
        data_range = 1
    else:
        data_range = 2
    out =  ssim(h1, h2, data_range=data_range, win_size=5) #to match the implementation of Been KIM

    return out


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

    incorrect_img_list = np.load('/home/naman/CS231n/heatmap_tests/Madri/Madri_New/'
                                 'robustness_applications/img_name_files/'
                                 'incorrect_img_names.npy').tolist()

    ###############################################
    model_names = []
    model_names.append('googlenet')
    model_names.append('pytorch')
    model_names.append('madry')

    method_dict = {'grad': 'Grad',
                   'inpgrad': 'InpGrad',
                   'ig': 'IG',
                   'lime': 'Lime',
                   'mp': 'MP',
                   'occlusion': 'Occlusion',
                   'sg': 'SmoothGrad',
                   }
    method_name = method_dict[args.method_name]

    ###################################################################################################################
    print('Getting the probabilities')
    clean_probs_path = '/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/results/' \
                           'Probs_Without_Noise'
    noisy_probs_path = '/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/results/' \
                       'Probs_With_Noise'

    probs_clean_img = {'pytorch': [],
                       'googlenet': [],
                       'madry': []}

    if args.if_noise:
        probs_noisy_img = deepcopy(probs_clean_img)

    for model_name in model_names:
        txt_file_names = glob.glob(os.path.join(clean_probs_path,
                                                       f'Grad_{model_name}/time_*{model_name}*.txt'))
        txt_file_names.sort()
        prob_files = [i for i in txt_file_names if '_Probs_' in i]
        img_name_files = [i for i in txt_file_names if '_Probs_' not in i]

        prob_list = []
        img_list = []
        for fIdx, prob_file in enumerate(prob_files):
            prob_file_batch_idx = int(prob_file.split('/')[-1].split('_batch_idx_')[-1].split('_')[0])
            img_file_batch_idx = int(img_name_files[fIdx].split('/')[-1].split('_batch_idx_')[-1].split('_')[0])
            assert prob_file_batch_idx == img_file_batch_idx, 'Something is wrong'

            with open(prob_file, 'r') as probFile:
                prob_list.extend(probFile.read().splitlines())
            with open(img_name_files[fIdx], 'r') as imgFile:
                img_list.extend(imgFile.read().splitlines())

        prob_list = np.array(prob_list).astype('float32').tolist()
        img_list = [i.split('/')[-1].split('.')[0] for i in img_list]
        probs_clean_img[model_name] = dict(zip(img_list, prob_list))

    if args.if_noise:
        for model_name in model_names:
            txt_file_names = glob.glob(os.path.join(noisy_probs_path,
                                                    f'Grad_{model_name}/time_*{model_name}*.txt'))
            txt_file_names.sort()
            prob_files = [i for i in txt_file_names if '_Probs_' in i]
            img_name_files = [i for i in txt_file_names if '_Probs_' not in i]

            prob_list = []
            img_list = []
            for fIdx, prob_file in enumerate(prob_files):
                prob_file_batch_idx = int(prob_file.split('/')[-1].split('_batch_idx_')[-1].split('_')[0])
                img_file_batch_idx = int(img_name_files[fIdx].split('/')[-1].split('_batch_idx_')[-1].split('_')[0])
                assert prob_file_batch_idx == img_file_batch_idx, 'Something is wrong'

                with open(prob_file, 'r') as probFile:
                    prob_list.extend(probFile.read().splitlines())
                with open(img_name_files[fIdx], 'r') as imgFile:
                    img_list.extend(imgFile.read().splitlines())

            prob_list = np.array(prob_list).astype('float32').tolist()
            img_list = [i.split('/')[-1].split('.')[0] for i in img_list]
            probs_noisy_img[model_name] = dict(zip(img_list, prob_list))

    ###################################################################################################################
    if method_name.lower() == 'grad':
        grad_dir = '/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/' \
                   'results/Gradient/Combined'
        smooth_dir = '/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/' \
                 'results/SmoothGrad/A31/Samples_50'
    else:
        if args.if_same_comp:
            grad_dir = '/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/' \
                       'results/IG/IG_low_setting'
        else:
            grad_dir = '/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/' \
                      'results_backup/InputxGradient/Img_Dir'
        smooth_dir = '/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/' \
                     'results/IG/IG_Best_Res/Combined'

    ssim_score_gNet = []
    ssim_score_rNet = []
    img_name_list = []
    for idx, img_name in enumerate(img_filenames):
        iter_time = time.time()
        img_name_list.append(img_name)
        img_path = os.path.join(base_img_dir, f'{img_name}.JPEG')
        orig_img = preprocess_img(Image.open(img_path))
        targ_label = eutils.get_image_class(os.path.join(base_img_dir,
                                                         f'{img_name}.JPEG'))

        if args.if_noise:
            noisy_img = skimage.util.random_noise(np.asarray(orig_img), mode='gaussian',
                                            mean=0, var=0.1, seed=0,
                                            )  # numpy, dtype=float64,range (0, 1)
            noisy_img = Image.fromarray(np.uint8(noisy_img * 255))
            noisy_img = preprocess_img(noisy_img)

        #############################################################################
        ## Reading the npy files
        if args.img_name_dir_flag:
            path = f'{img_name}/'
        else:
            path = f''

        gNet_path = glob.glob(os.path.join(smooth_dir, f'{path}*_googlenet*.npy'))
        mNet_path = glob.glob(os.path.join(grad_dir, f'{path}*_madry*.npy'))
        mNet_path.sort()
        rNet_path = glob.glob(os.path.join(smooth_dir, f'{path}*_pytorch*.npy'))

        if method_name.lower() == 'grad':
            gNet_path = [i for i in gNet_path if '_num_samples_50_stdev_spread_0.10_' in i]
            rNet_path = [i for i in rNet_path if '_num_samples_50_stdev_spread_0.10_' in i]
            mNet_path = [i for i in mNet_path if '_if_noise_0_' in i]
        else:
            mNet_path = [i for i in mNet_path if '_if_noise_0_' in i]

        assert len(gNet_path) == 1, 'Something is wrong in reading the .npy files\n'
        assert len(mNet_path) == 1, 'Something is wrong in reading the .npy files\n'
        assert len(rNet_path) == 1, 'Something is wrong in reading the .npy files\n'
        #############################################################################

        gNet_heatmaps = [np.load(i) for i in gNet_path]
        mNet_heatmaps = [np.load(i) for i in mNet_path]
        rNet_heatmaps = [np.load(i) for i in rNet_path]

        gNet_heatmaps = [i / max(abs(i.min()), abs(i.max())) for i in gNet_heatmaps]
        mNet_heatmaps = [i / max(abs(i.min()), abs(i.max())) for i in mNet_heatmaps]
        rNet_heatmaps = [i / max(abs(i.min()), abs(i.max())) for i in rNet_heatmaps]

        ssim_score_gNet.append(compute_score(gNet_heatmaps[0], mNet_heatmaps[0], method_name))
        ssim_score_rNet.append(compute_score(rNet_heatmaps[0], mNet_heatmaps[0], method_name))

        # print(f'Time taken is {time.time() - iter_time}')




    ssim_score_gNet = np.array(ssim_score_gNet)
    ssim_score_rNet = np.array(ssim_score_rNet)

    print(f'For method {method_name}: gNet_avg_ssim: {np.mean(ssim_score_gNet)}, '
          f'rNet_avg_ssim: {np.mean(ssim_score_rNet)}')
    print(f'For method {method_name}: gNet_dev_ssim: {np.std(ssim_score_gNet)}, '
          f'rNet_dev_ssim: {np.std(ssim_score_rNet)}')

    img_name_list = np.array(img_name_list)

    if args.save:
        sorted_ssim_scores_gNet_idx = np.flip(np.argsort(ssim_score_gNet))
        sorted_gNet_img_name = img_name_list[sorted_ssim_scores_gNet_idx]
        sorted_ssim_score_gNet = ssim_score_gNet[sorted_ssim_scores_gNet_idx]

        sorted_ssim_scores_rNet_idx = np.flip(np.argsort(ssim_score_rNet))
        sorted_rNet_img_name = img_name_list[sorted_ssim_scores_rNet_idx]
        sorted_ssim_score_rNet = ssim_score_rNet[sorted_ssim_scores_rNet_idx]

        top_imgs, gNet_idx, rNet_idx = np.intersect1d(sorted_gNet_img_name[:100], sorted_rNet_img_name[:100],
                                                      assume_unique=True, return_indices=True)

        sorted_ssim_score_gNet = sorted_ssim_score_gNet[gNet_idx]
        sorted_ssim_score_rNet = sorted_ssim_score_rNet[rNet_idx]

        path = os.path.join(args.out_path,
                            f'Method_{method_name}')

        temp_arr = np.concatenate((top_imgs.reshape(-1, 1),
                                   sorted_ssim_score_gNet.reshape(-1, 1),
                                   sorted_ssim_score_rNet.reshape(-1, 1),
                                   ), axis=-1)

        eutils.mkdir_p(path)
        fName = os.path.join(path, f'time_{f_time}_'
                                   f'top_100_images_with_ssim_scores.txt')

        file_handle = open(fName, 'ab')

        np.savetxt(file_handle, temp_arr,
                   fmt='%s,  %s,  %s',
                   header='Img Name               ,  GoogleNet         ,  ResNet            ', comments='', )
        file_handle.close()

    print(f'\nTime taken is {time.time() - s_time}')
    print(f'Time stamp is {f_time}\n')

########################################################################################################################




