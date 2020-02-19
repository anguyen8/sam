import ipdb, os, time, argparse, sys, glob, skimage
from srblib import abs_path

import sys; sys.path.append("..")
import utils as eutils

import numpy as np
from copy import deepcopy
from functools import reduce
from PIL import Image
from torchvision.transforms import transforms

import warnings
warnings.filterwarnings("ignore")


def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Paramters for sensitivity analysis of heatmaps')

    # parser.add_argument('-idp', '--input_dir_path', help='Path of the image directory', metavar='DIR')

    parser.add_argument('-op', '--out_path',
                        help='Path of the output directory where you want to save the results (Default is ./)')

    # parser.add_argument('-mn', '--method_name', choices=['grad', 'inpgrad'],
    #                     help='Method you are analysing')

    # parser.add_argument('--metric_name', choices=['ssim', 'hog', 'spearman'],
    #                     help='Metric to be computed')

    parser.add_argument('--save', action='store_true', default=False,
                        help=f'Flag to say that plot need to be saved. '
                             f'Default=False')

    parser.add_argument('-s_idx', '--start_idx', type=int,
                        help='Start index for selecting images. Default: 0', default=0,
                        )

    parser.add_argument('-e_idx', '--end_idx', type=int,
                        help='End index for selecting images. Default: 2K', default=2000,
                        )

    parser.add_argument('--if_same_comp', action='store_true', default=False,
                        help=f'Flag to say if you want to compare with the same method for Madry'
                             f'Default=False')

    # parser.add_argument('--if_noise', action='store_true',
    #                     help='Flag whether noise was present in the input while calculating the heatmaps. Default: False',
    #                     default=False,
    #                     )

    # Parse the arguments
    args = parser.parse_args()

    # args.model_name_dir_flag = False
    args.img_name_dir_flag = True
    args.if_noise = False

    # if args.input_dir_path is None:
    #     print('Please provide image dir path. Exiting')
    #     sys.exit(1)
    # args.input_dir_path = abs_path(args.input_dir_path)

    if args.out_path is None:
        args.out_path = './'
    args.out_path = os.path.abspath(args.out_path)

    # if args.method_name is None:
    #     print('Please provide the name of the method.\nExiting')
    #     sys.exit(0)

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
    # method_name = method_dict[args.method_name]

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
    grad_dir = '/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/' \
               'results/Gradient/Combined'
    if args.if_same_comp:
        inpgrad_dir = '/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/' \
                       'results/IG/IG_low_setting'
    else:
        inpgrad_dir = '/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/' \
                      'results_backup/InputxGradient/Img_Dir'
    smooth_dir = '/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/' \
                 'results/SmoothGrad/A31/Samples_50'
    ig_dir = '/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/' \
                 'results/IG/IG_Best_Res/Combined'

    for idx, img_name in enumerate(img_filenames):
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

        ## GoogleNet
        gNet_path = []
        # aa = glob.glob(os.path.join(grad_dir, f'{path}*_googlenet*.npy')) #Grad
        # aa.extend(glob.glob(os.path.join(inpgrad_dir, f'{path}*_googlenet*.npy'))) #InpxGrad
        # gNet_path.extend([i for i in aa if '_if_noise_0_' in i])
        aa = glob.glob(os.path.join(smooth_dir, f'{path}*_googlenet*.npy')) #SG
        gNet_path.extend([i for i in aa if '_num_samples_50_stdev_spread_0.10_' in i])
        gNet_path.extend(glob.glob(os.path.join(ig_dir, f'{path}*_googlenet*.npy'))) #IG

        ## MadryNet
        mNet_path = []
        aa = glob.glob(os.path.join(grad_dir, f'{path}*_madry*.npy'))  # Grad
        aa.extend(glob.glob(os.path.join(inpgrad_dir, f'{path}*_madry*.npy')))  # InpxGrad
        mNet_path.extend([i for i in aa if '_if_noise_0_' in i])
        # aa = glob.glob(os.path.join(smooth_dir, f'{path}*_madry*.npy'))  # SG
        # mNet_path.extend([i for i in aa if '_num_samples_50_stdev_spread_0.10_' in i])
        # mNet_path.extend(glob.glob(os.path.join(ig_dir, f'{path}*_madry*.npy')))  # IG

        ## ResNet
        ## MadryNet
        rNet_path = []
        # aa = glob.glob(os.path.join(grad_dir, f'{path}*_pytorch*.npy'))  # Grad
        # aa.extend(glob.glob(os.path.join(inpgrad_dir, f'{path}*_pytorch*.npy')))  # InpxGrad
        # rNet_path.extend([i for i in aa if '_if_noise_0_' in i])
        aa = glob.glob(os.path.join(smooth_dir, f'{path}*_pytorch*.npy'))  # SG
        rNet_path.extend([i for i in aa if '_num_samples_50_stdev_spread_0.10_' in i])
        rNet_path.extend(glob.glob(os.path.join(ig_dir, f'{path}*_pytorch*.npy')))  # IG


        assert len(gNet_path) == 2, 'Something is wrong in reading the .npy files\n'
        assert len(mNet_path) == 2, 'Something is wrong in reading the .npy files\n'
        assert len(rNet_path) == 2, 'Something is wrong in reading the .npy files\n'
        #############################################################################

        gNet_heatmaps = [np.load(i) for i in gNet_path]
        mNet_heatmaps = [np.load(i) for i in mNet_path]
        rNet_heatmaps = [np.load(i) for i in rNet_path]

        grid = []
        grid.append([orig_img] + gNet_heatmaps) # GoogleNet
        grid.append([np.ones_like(orig_img)] + rNet_heatmaps)  # ResNetNet
        grid.append([np.ones_like(orig_img)] + mNet_heatmaps)  # MadryNetNet


        gNet_prob_clear = probs_clean_img['googlenet'][img_name]
        mNet_prob_clear = probs_clean_img['madry'][img_name]
        rNet_prob_clear = probs_clean_img['pytorch'][img_name]

        row_labels_left = []
        row_labels_left.append((f'GoogleNet: Top-1:\n{im_label_map[int(targ_label)]}: {gNet_prob_clear:.03f}\n',))
        row_labels_left.append((f'ResNet50: Top-1:\n{im_label_map[int(targ_label)]}: {rNet_prob_clear:.03f}\n',))
        row_labels_left.append((f'MadryNet: Top-1:\n{im_label_map[int(targ_label)]}: {mNet_prob_clear:.03f}\n',))


        row_labels_right = []
        col_labels = []
        col_labels.append([f'Orig_Img', f'SmoothGrad', f'Integrated Gridents'])
        col_labels.append([f' ',        f'SmoothGrad', f'Integrated Gridents'])
        if args.if_same_comp:
            col_labels.append([f' ',        f'Gradient',   f'Integrated Gridents'])
        else:
            col_labels.append([f' ',        f'Gradient',   f'Input x Gradient'])
        out_dir = os.path.join(args.out_path, f'{img_name}')

        file_name = [f'time_{f_time}_Plot_Combined_{img_name}_gNet.png',
                     f'time_{f_time}_Plot_Combined_{img_name}_rNet.png',
                     f'time_{f_time}_Plot_Combined_{img_name}_mNet.png',
                     ]

        for i in range(len(grid)):
            eutils.zero_out_plot_multiple_patch([grid[i]],
                                                out_dir,
                                                [row_labels_left[i]],
                                                row_labels_right,
                                                col_labels[i],
                                                file_name=file_name[i],
                                                dpi=224,
                                                save=args.save
                                                )


    print(f'Time taken is {time.time() - s_time}')
    print(f'Time stamp is {f_time}\n')

########################################################################################################################




