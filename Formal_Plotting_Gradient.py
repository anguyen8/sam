import ipdb, os, time, argparse, sys, glob, skimage
from srblib import abs_path

# import sys; sys.path.append("..")
import utils as eutils

import numpy as np
from copy import deepcopy
from functools import reduce
from PIL import Image
from torchvision.transforms import transforms

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt


def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Paramters for sensitivity analysis of heatmaps')

    parser.add_argument('-idp', '--input_dir_path', help='Path of the image directory', metavar='DIR')

    parser.add_argument('-op', '--out_path',
                        help='Path of the output directory where you want to save the results (Default is ./)')

    parser.add_argument('-mn', '--method_name', choices=['grad', 'inpgrad'],
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

    args.model_name_dir_flag = False
    args.img_name_dir_flag = True
    args.if_noise = True

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
def sort_fun(val):
    try:
        return int(val.split('/')[-1].split('_if_noise_')[-1].split('_')[0])
    except:
        print(f'Something is wrong with the sorting function.Check')
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

    incorrect_img_list = np.load('/home/naman/CS231n/heatmap_tests/Madri/Madri_New/'
                                 'robustness_applications/img_name_files/incorrect_img_names.npy').tolist()

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

    # ###################################################################################################################
    # print('Getting the probabilities')
    # clean_probs_path = '/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/results/' \
    #                        'Probs_Without_Noise'
    # noisy_probs_path = '/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/results/' \
    #                    'Probs_With_Noise'
    #
    # probs_clean_img = {'pytorch': [],
    #                    'googlenet': [],
    #                    'madry': []}
    #
    # if args.if_noise:
    #     probs_noisy_img = deepcopy(probs_clean_img)
    #
    # for model_name in model_names:
    #     txt_file_names = glob.glob(os.path.join(clean_probs_path,
    #                                                    f'Grad_{model_name}/time_*{model_name}*.txt'))
    #     txt_file_names.sort()
    #     prob_files = [i for i in txt_file_names if '_Probs_' in i]
    #     img_name_files = [i for i in txt_file_names if '_Probs_' not in i]
    #
    #     prob_list = []
    #     img_list = []
    #     for fIdx, prob_file in enumerate(prob_files):
    #         prob_file_batch_idx = int(prob_file.split('/')[-1].split('_batch_idx_')[-1].split('_')[0])
    #         img_file_batch_idx = int(img_name_files[fIdx].split('/')[-1].split('_batch_idx_')[-1].split('_')[0])
    #         assert prob_file_batch_idx == img_file_batch_idx, 'Something is wrong'
    #
    #         with open(prob_file, 'r') as probFile:
    #             prob_list.extend(probFile.read().splitlines())
    #         with open(img_name_files[fIdx], 'r') as imgFile:
    #             img_list.extend(imgFile.read().splitlines())
    #
    #     prob_list = np.array(prob_list).astype('float32').tolist()
    #     img_list = [i.split('/')[-1].split('.')[0] for i in img_list]
    #     probs_clean_img[model_name] = dict(zip(img_list, prob_list))
    #
    # if args.if_noise:
    #     for model_name in model_names:
    #         txt_file_names = glob.glob(os.path.join(noisy_probs_path,
    #                                                 f'Grad_{model_name}/time_*{model_name}*.txt'))
    #         txt_file_names.sort()
    #         prob_files = [i for i in txt_file_names if '_Probs_' in i]
    #         img_name_files = [i for i in txt_file_names if '_Probs_' not in i]
    #
    #         prob_list = []
    #         img_list = []
    #         for fIdx, prob_file in enumerate(prob_files):
    #             prob_file_batch_idx = int(prob_file.split('/')[-1].split('_batch_idx_')[-1].split('_')[0])
    #             img_file_batch_idx = int(img_name_files[fIdx].split('/')[-1].split('_batch_idx_')[-1].split('_')[0])
    #             assert prob_file_batch_idx == img_file_batch_idx, 'Something is wrong'
    #
    #             with open(prob_file, 'r') as probFile:
    #                 prob_list.extend(probFile.read().splitlines())
    #             with open(img_name_files[fIdx], 'r') as imgFile:
    #                 img_list.extend(imgFile.read().splitlines())
    #
    #         prob_list = np.array(prob_list).astype('float32').tolist()
    #         img_list = [i.split('/')[-1].split('.')[0] for i in img_list]
    #         probs_noisy_img[model_name] = dict(zip(img_list, prob_list))

    ###################################################################################################################

    input_path = args.input_dir_path




    if args.model_name_dir_flag: ## This flag is False (BUT DON"T REMOVE IT)
        model_dir_names = [os.path.join(input_path, f'{method_name}_{i}/') for i in model_names]
        model_dir_names.sort()
        ##Order is googleNet, Madry, PyTorch
        file_names = glob.glob(os.path.join(input_path, '*/'))
        file_names.sort()

        check = [i == j for i, j in zip(file_names, model_dir_names)]
        assert all(check) == True, 'Something is wrong here'

    for _, img_name in enumerate(img_filenames):
        if img_name in incorrect_img_list:
            continue

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

        if args.model_name_dir_flag: ## This flag is False (BUT DON"T REMOVE IT)

            assert args.model_name_dir_flag == False, 'Something is wrong'
            gNet_path = glob.glob(os.path.join(model_dir_names[0], f'{path}*model_name_googlenet*.npy'))
            mNet_path = glob.glob(os.path.join(model_dir_names[1], f'{path}*model_name_madry_batch*.npy'))
            rNet_path = glob.glob(os.path.join(model_dir_names[2], f'{path}*model_name_pytorch*.npy'))

        else:
            gNet_path = glob.glob(os.path.join(input_path, f'{path}*model_name_googlenet*.npy'))
            mNet_path = glob.glob(os.path.join(input_path, f'{path}*model_name_madry_batch*.npy'))
            mGoogleNet_path = glob.glob(os.path.join(input_path, f'{path}*model_name_madry_googlenet*.npy'))
            rNet_path = glob.glob(os.path.join(input_path, f'{path}*model_name_pytorch*.npy'))


        assert len(gNet_path) == 2, 'Something is wrong in reading the .npy files\n'
        assert len(mNet_path) == 2, 'Something is wrong in reading the .npy files\n'
        assert len(mGoogleNet_path) == 2, 'Something is wrong in reading the .npy files\n'
        assert len(rNet_path) == 2, 'Something is wrong in reading the .npy files\n'

        gNet_path.sort(key=sort_fun)
        mNet_path.sort(key=sort_fun)
        mGoogleNet_path.sort(key=sort_fun)
        rNet_path.sort(key=sort_fun)

        assert 'if_noise_0' in gNet_path[0], 'Noise heatmap should be at zero index\n'
        assert 'if_noise_0' in rNet_path[0], 'Noise heatmap should be at zero index\n'
        assert 'if_noise_0' in mNet_path[0], 'Noise heatmap should be at zero index\n'
        assert 'if_noise_0' in mGoogleNet_path[0], 'Noise heatmap should be at zero index\n'
        #############################################################################

        gNet_heatmaps = [np.load(i) for i in gNet_path]
        mNet_heatmaps = [np.load(i) for i in mNet_path]
        mGoogleNet_heatmaps = [np.load(i) for i in mGoogleNet_path]
        rNet_heatmaps = [np.load(i) for i in rNet_path]

        grid = []
        grid.append([orig_img, gNet_heatmaps[0], mGoogleNet_heatmaps[0], rNet_heatmaps[0], mNet_heatmaps[0]])
        grid.append([noisy_img, gNet_heatmaps[1], mGoogleNet_heatmaps[1], rNet_heatmaps[1], mNet_heatmaps[1]])


        # gNet_prob_clear = probs_clean_img['googlenet'][img_name]
        # mNet_prob_clear = probs_clean_img['madry'][img_name]
        # rNet_prob_clear = probs_clean_img['pytorch'][img_name]
        #
        # gNet_prob_noisy = probs_noisy_img['googlenet'][img_name]
        # mNet_prob_noisy = probs_noisy_img['madry'][img_name]
        # rNet_prob_noisy = probs_noisy_img['pytorch'][img_name]

        row_labels_left = []
        # row_labels_left.append((f'Target:\n{im_label_map[int(targ_label)]}',
        #                         f'GoogleNet: Top-1:\n{im_label_map[int(targ_label)]}: {gNet_prob_clear:.03f}\n',
        #                         f'ResNet50: Top-1:\n{im_label_map[int(targ_label)]}: {rNet_prob_clear:.03f}\n',
        #                         f'MadryNet: Top-1:\n{im_label_map[int(targ_label)]}: {mNet_prob_clear:.03f}\n',
        #                         ))
        #
        # row_labels_left.append((f'Target:\n{im_label_map[int(targ_label)]}',
        #                         f'GoogleNet: Top-1:\n{im_label_map[int(targ_label)]}: {gNet_prob_noisy:.03f}\n',
        #                         f'ResNet50: Top-1:\n{im_label_map[int(targ_label)]}: {rNet_prob_noisy:.03f}\n',
        #                         f'MadryNet: Top-1:\n{im_label_map[int(targ_label)]}: {mNet_prob_noisy:.03f}\n',
        #                         ))

        row_labels_right = []
        col_labels = []
        col_labels.append(['Clean_Img', 'GoogleNet', 'GoogleNet-R', 'ResNet50', 'ResNet50-R'])
        col_labels.append(['Noisy_Img', 'GoogleNet', 'GoogleNet-R', 'ResNet50', 'ResNet50-R'])
        # out_dir = args.out_path
        out_dir = os.path.join(args.out_path, f'{img_name}')

        file_name = [f'time_{f_time}_Plot_{img_name}_Clean.png', f'time_{f_time}_Plot_{img_name}_Noisy.png']

        # ipdb.set_trace()
        for i in range(len(grid)):

            eutils.zero_out_plot_multiple_patch([grid[i]],
                                                out_dir,
                                                # [row_labels_left[i]],
                                                row_labels_left,
                                                row_labels_right,
                                                col_labels[i],
                                                file_name=file_name[i],
                                                dpi=224,
                                                save=args.save,
                                                )

        ipdb.set_trace()


    print(f'Time taken is {time.time() - s_time}')
    print(f'Time stamp is {f_time}\n')

########################################################################################################################




