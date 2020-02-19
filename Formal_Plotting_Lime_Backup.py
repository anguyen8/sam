import ipdb, os, time, argparse, sys, glob, skimage
from srblib import abs_path
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

    parser.add_argument('--image_idx', type=int, help='Index of the image')

    parser.add_argument('-op', '--out_path',
                        help='Path of the output directory where you want to save the results (Default is ./)')

    parser.add_argument('-mn', '--method_name', choices=['lime'],
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

    parser.add_argument('--exp_name', choices=['a01', 'a02', 'a03'],
                        help='Experiment number of LIME',
                        )

    parser.add_argument('--if_noise', action='store_true',
                        help='Flag whether noise was present in the input while calculating the heatmaps. Default: False',
                        default=False,
                        )

    # Parse the arguments
    args = parser.parse_args()

    args.model_name_dir_flag = False
    args.img_name_dir_flag = True

    if args.exp_name is None:
        print(f'Provide exp_num.\nExiting')
        sys.exit(1)

    if args.image_idx is None:
        print(f'Please provide this value\nExiting')
        sys.exit(1)

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
def sort_fun_selection(method, exp_name):
    if method.lower() == 'lime':
        if exp_name.lower() == 'a01':
            def sort_fun(elem):
                return int(elem.split('/')[-1].split('_sample_count_')[-1].split('_')[0])
        elif exp_name.lower() == 'a02':
            def sort_fun(elem):
                return int(elem.split('/')[-1].split('_explainer_seed_')[-1].split('_')[0])
        elif exp_name.lower() == 'a03':
            def sort_fun(elem):
                key_val = elem.split('/')[-1].split('_background_pixel_')[-1].split('_')[0]
                if key_val.isdigit():
                    if int(key_val) == 0:
                        out = 0
                elif key_val.lower() == 'grey':
                    out = 1
                elif key_val.lower() == 'none':
                    out = 2
                elif key_val.lower() == 'random':
                    out = 3
                else:
                    print(f'Key not implemented.\nExiting')
                    sys.exit(0)
                return out
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

    img_filenames = [f'ILSVRC2012_val_000{args.image_idx:05d}', ]

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

    input_path = args.input_dir_path


    if args.model_name_dir_flag:
        model_dir_names = [os.path.join(input_path, f'{method_name}_{i}/') for i in model_names]
        model_dir_names.sort()
        ##Order is googleNet, Madry, PyTorch
        file_names = glob.glob(os.path.join(input_path, '*/'))
        file_names.sort()



        check = [i == j for i, j in zip(file_names, model_dir_names)]
        assert all(check) == True, 'Something is wrong here'

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

        if args.model_name_dir_flag:
            gNet_path = glob.glob(os.path.join(model_dir_names[0], f'{path}*_googlenet*.npy'))
            gNet_path = [i for i in gNet_path if 'madry_googlenet' not in i]

            mNet_path = glob.glob(os.path.join(model_dir_names[1], f'{path}*_madry*.npy'))
            mNet_path = [i for i in mNet_path if 'madry_googlenet' not in i]

            rNet_path = glob.glob(os.path.join(model_dir_names[2], f'{path}*_pytorch*.npy'))

            mGoogleNet_path = glob.glob(os.path.join(model_dir_names[2], f'{path}*_madry_googlenet*.npy'))



        else:
            gNet_path = glob.glob(os.path.join(input_path, f'{path}*_googlenet*.npy'))
            gNet_path = [i for i in gNet_path if 'madry_googlenet' not in i]

            mNet_path = glob.glob(os.path.join(input_path, f'{path}*_madry*.npy'))
            mNet_path = [i for i in mNet_path if 'madry_googlenet' not in i]

            rNet_path = glob.glob(os.path.join(input_path, f'{path}*_pytorch*.npy'))
            mGoogleNet_path = glob.glob(os.path.join(input_path, f'{path}*_madry_googlenet*.npy'))

        npy_len_list = [len(gNet_path), len(mNet_path), len(rNet_path), len(mGoogleNet_path)]
        if args.exp_name.lower() == 'a01':
            ## Function would sort in increasing order of samples
            sort_fun = sort_fun_selection(method_name, args.exp_name.lower())
            assert all(i == 3 for i in npy_len_list), \
                'Something is wrong in reading the .npy files\n'
        elif args.exp_name.lower() == 'a02':
            sort_fun = sort_fun_selection(method_name, args.exp_name.lower())
            assert all(i == 5 for i in npy_len_list), \
                'Something is wrong in reading the .npy files\n'
        elif args.exp_name.lower() == 'a03':
            sort_fun = sort_fun_selection(method_name, args.exp_name.lower())
            assert all(i == 4 for i in npy_len_list), \
                'Something is wrong in reading the .npy files\n'
        else:
            print(f'Not implemented')
            sys.exit(0)


        gNet_path.sort(key=sort_fun)
        mNet_path.sort(key=sort_fun)
        rNet_path.sort(key=sort_fun)
        mGoogleNet_path.sort(key=sort_fun)

        gNet_heatmaps = [np.load(i) for i in gNet_path]
        mNet_heatmaps = [np.load(i) for i in mNet_path]
        rNet_heatmaps = [np.load(i) for i in rNet_path]
        mGoogleNet_heatmaps = [np.load(i) for i in mGoogleNet_path]

        #############################################################################
        ## Plotting
        grid = []
        col_labels = []
        # col_labels = [' ', ' ', ' ', ' ', ' ']
        row_labels_left = []
        row_labels_right = []
        # col_labels = ['Orig_Img', 'GoogleNet', 'GoogleNet-R', 'ResNet50', 'ResNet50-R']
        out_dir = args.out_path
        # out_dir = os.path.join(args.out_path, f'{img_name}')

        if args.if_noise:
            img = noisy_img
            gNet_prob = probs_noisy_img['googlenet'][img_name]
            mNet_prob = probs_noisy_img['madry'][img_name]
            rNet_prob = probs_noisy_img['pytorch'][img_name]
        else:
            img = orig_img
            gNet_prob = probs_clean_img['googlenet'][img_name]
            mNet_prob = probs_clean_img['madry'][img_name]
            rNet_prob = probs_clean_img['pytorch'][img_name]

        for i in range(npy_len_list[0]):
            grid.append([img, gNet_heatmaps[i], mGoogleNet_heatmaps[i], rNet_heatmaps[i], mNet_heatmaps[i]])
            # row_labels_left.append((f'Target:\n{im_label_map[int(targ_label)]}',
            #                         f'GoogleNet: Top-1:\n{im_label_map[int(targ_label)]}: {gNet_prob:.03f}\n',
            #                         f'ResNet50: Top-1:\n{im_label_map[int(targ_label)]}: {rNet_prob:.03f}\n',
            #                         f'MadryNet: Top-1:\n{im_label_map[int(targ_label)]}: {mNet_prob:.03f}\n',
            #                         ))

        eutils.zero_out_plot_multiple_patch(grid,
                                            out_dir,
                                            row_labels_left,
                                            row_labels_right,
                                            col_labels,
                                            file_name=f'time_{f_time}_Plot_IG_{img_name}.png',
                                            dpi=224,
                                            save=args.save,
                                            )


    print(f'Time taken is {time.time() - s_time}')
    print(f'Time stamp is {f_time}\n')

########################################################################################################################




