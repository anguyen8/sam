import argparse
import glob
import os
import sys
import time
import ipdb
from copy import deepcopy
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np

import utils as eutils

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 12, 'font.weight':'bold'})
plt.rc("font", family="sans-serif")


def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Paramters for sensitivity analysis of heatmaps')

    parser.add_argument('-idp', '--input_dir_path', help='Path of the input directory', metavar='DIR')

    parser.add_argument('-op', '--out_path',
                        help='Path of the output directory where you want to save the results (Default is ./)')

    parser.add_argument('-mn', '--method_name', choices=['occlusion', 'ig', 'sg', 'grad', 'lime', 'mp', 'inpgrad'],
                        help='Method you are analysing')

    # parser.add_argument('--exp_num', type=int,
    #                     help='Experiment index for a particular method.Default=0', default=0)

    # parser.add_argument('--metric_name', choices=['ssim', 'spearman', 'hog', 'insertion', 'deletion', 'iou'],
    #                     help='Metric to be computed')

    parser.add_argument('--save', action='store_true', default=False,
                        help=f'Flag to say that plot need to be saveed. '
                             f'Default=False')

    # Parse the arguments
    args = parser.parse_args()
    # args.start_idx = 0
    # args.end_idx = 2000

    # if args.num_variations is None:
    #     print('Please provide this number.\nExiting')
    #     sys.exit(0)
    # elif args.num_variations < 2:
    #     print('This number cant be less than 2.\nExiting')
    #     sys.exit(0)

    if args.input_dir_path is None:
        print('Please provide image dir path. Exiting')
        sys.exit(1)
    args.input_dir_path = os.path.abspath(args.input_dir_path)

    if args.method_name is None:
        print('Please provide the name of the method.\nExiting')
        sys.exit(0)

    # if args.metric_name is None:
    #     print('Please provide the name of the metric.\nExiting')
    #     sys.exit(0)

    if args.out_path is None:
        args.out_path = './'
    args.out_path = os.path.abspath(args.out_path)

    return args


########################################################################################################################
def combined_errorbar_plot(data_dicts, fName, out_dir, title=None, save=False):
    width = 0.5
    alpha = 1
    fig, ax = plt.subplots()
    eBar_color = 'dimgrey'
    err_kwargs = dict(elinewidth=1, capsize=0, markeredgewidth=0, ecolor=eBar_color)
    # bar_colors = ['tomato', 'cornflowerblue', 'mediumseagreen']
    bar_colors = ['crimson', 'lightcoral', 'green', 'springgreen', ]

    #################################################
    ## Here mean is not doing anything. Since there is just one element
    ## SSIM DATA
    ssim_mean_data = (np.mean(data_dicts['ssim']['mean_dict']['googlenet']),
                      np.mean(data_dicts['ssim']['mean_dict']['madry_googlenet']),
                      np.mean(data_dicts['ssim']['mean_dict']['pytorch']),
                      np.mean(data_dicts['ssim']['mean_dict']['madry'])
                      )

    ssim_std_data = (np.mean(data_dicts['ssim']['std_dict']['googlenet']),
                     np.mean(data_dicts['ssim']['std_dict']['madry_googlenet']),
                     np.mean(data_dicts['ssim']['std_dict']['pytorch']),
                     np.mean(data_dicts['ssim']['std_dict']['madry'])
                     )

    ## SPEARMAN DATA
    spearman_mean_data = (np.mean(data_dicts['spearman']['mean_dict']['googlenet']),
                          np.mean(data_dicts['spearman']['mean_dict']['madry_googlenet']),
                          np.mean(data_dicts['spearman']['mean_dict']['pytorch']),
                          np.mean(data_dicts['spearman']['mean_dict']['madry'])
                          )

    spearman_std_data = (np.mean(data_dicts['spearman']['std_dict']['googlenet']),
                         np.mean(data_dicts['spearman']['std_dict']['madry_googlenet']),
                         np.mean(data_dicts['spearman']['std_dict']['pytorch']),
                         np.mean(data_dicts['spearman']['std_dict']['madry']),
                         )

    ## HOG DATA
    hog_mean_data = (np.mean(data_dicts['hog']['mean_dict']['googlenet']),
                     np.mean(data_dicts['hog']['mean_dict']['madry_googlenet']),
                     np.mean(data_dicts['hog']['mean_dict']['pytorch']),
                     np.mean(data_dicts['hog']['mean_dict']['madry'])
                     )

    hog_std_data = (np.mean(data_dicts['hog']['std_dict']['googlenet']),
                    np.mean(data_dicts['hog']['std_dict']['madry_googlenet']),
                    np.mean(data_dicts['hog']['std_dict']['pytorch']),
                    np.mean(data_dicts['hog']['std_dict']['madry']),
                    )
    ################################################

    ##GOOGLENET (Position 0) ## Starting from Top (GoogleNet)
    rects0 = ax.bar(3 * np.arange(3) - 0.75, (spearman_mean_data[0], hog_mean_data[0], ssim_mean_data[0]),
                     width=width, 
                     alpha=alpha, color=3*[bar_colors[0]], align='center',
                     yerr=(spearman_std_data[0], hog_std_data[0], ssim_std_data[0]),
                     error_kw=err_kwargs, label='GoogleNet')

    ##MADRY_GOOGLENET (Position 1) ## Starting from Top (GoogleNet-R)
    rects1 = ax.bar(3 * np.arange(3) - 0.25, (spearman_mean_data[1], hog_mean_data[1], ssim_mean_data[1]),
                     width=width, 
                     alpha=alpha, color=3*[bar_colors[1]], align='center',
                     yerr=(spearman_std_data[1], hog_std_data[1], ssim_std_data[1]),
                     error_kw=err_kwargs, label='GoogleNet-R')

    ##PYTORCH (Position 2) ## Starting from Top (ResNet50)
    rects2 = ax.bar(3 * np.arange(3) + 0.25, (spearman_mean_data[2], hog_mean_data[2], ssim_mean_data[2]),
                     width=width, 
                     alpha=alpha, color=3*[bar_colors[2]], align='center',
                     yerr=(spearman_std_data[2], hog_std_data[2], ssim_std_data[2]),
                     error_kw=err_kwargs, label='ResNet50',)

    ##MADRY (Position 3) ## Starting from Top (ResNet50-R)
    rects3 = ax.bar(3 * np.arange(3) + 0.75, (spearman_mean_data[3], hog_mean_data[3], ssim_mean_data[3]),
                     width=width, 
                     alpha=alpha, color=3 * [bar_colors[3]], align='center',
                     yerr=(spearman_std_data[3], hog_std_data[3], ssim_std_data[3]),
                     error_kw=err_kwargs, label='ResNet50-R', )
    
    # Add bar patterns
    patterns = ('///', '///', '///')
    for bar, pattern in zip(rects3, patterns):
        bar.set_hatch(pattern)

    patterns = ('///', '///', '///')
    for bar, pattern in zip(rects2, patterns):
        bar.set_hatch(pattern)

    ax.set_xticks(3*np.arange(3), minor=False)
    # ax.set_yticks([])
    ax.set_xticklabels(('Spearman', 'Pearson of HOGs', 'SSIM'), minor=False, va='center')
    # ax.tick_params(axis='y', rotation=90,  ) #length=0)


    if title is not None:
        ax.set_title(title, fontweight='bold', fontsize=13)
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    ## (x, y, width, height)
    # ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, prop={'weight':'normal', 'size':15})
    ax.legend(bbox_to_anchor=(1, 1), ncol=2, prop={'weight':'normal', 'size':15})

    # plt.legend(rects0, ['GoogleNet', 'ResNet50', 'MadryNet'])

    if args.save:
        print(f'Saving file: {os.path.join(out_dir, fName)}')
        eutils.mkdir_p(out_dir)
        fig.savefig(os.path.join(out_dir, fName), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


########################################################################################################################
if __name__ == '__main__':

    s_time = time.time()
    f_time = ''.join(str(s_time).split('.'))
    args = get_arguments()
    im_label_map = eutils.imagenet_label_mappings()

    model_names = []
    model_names.append('googlenet')
    model_names.append('pytorch')
    model_names.append('madry') #Robust_ResNet
    model_names.append('madry_googlenet') #Robust Googlenet

    method_dict = {'grad': 'Grad',
                   'inpgrad': 'InpGrad',
                   'ig': 'IG',
                   'lime': 'Lime',
                   'mp': 'MP',
                   'occlusion': 'Occlusion',
                   'sg': 'SmoothGrad',
                   }
    method_name = method_dict[args.method_name]

    metric_file_paths = [os.path.join(args.input_dir_path,
                               f'Method_{method_name}_Metric_{i}') for i in ['ssim', 'hog', 'spearman']]

    assert all([os.path.isdir(i) for i in metric_file_paths]) == True, \
        'Something is wrong with the input path'

    model_name_dir_flag = False

    empty_dict = {'pytorch': [],
                  'googlenet': [],
                  'madry': [],
                  'madry_googlenet': [],
                  }

    data_dicts = {'ssim': {'mean_dict': deepcopy(empty_dict), 'std_dict': deepcopy(empty_dict)},
                  'hog': {'mean_dict': deepcopy(empty_dict), 'std_dict': deepcopy(empty_dict)},
                  'spearman': {'mean_dict': deepcopy(empty_dict), 'std_dict': deepcopy(empty_dict)},
                  }

    ## Data read
    print(f'Reading Data ... ', end='')
    ## List for asserting the order of read
    order_list = ['googlenet', 'madry', 'madry_googlenet', 'pytorch']
    for input_path in metric_file_paths:
        metric_name = input_path.split('/')[-1].split('_')[-1]
        txt_data_files = glob.glob(os.path.join(input_path,
                                                f'*_Model_*_{method_name}_{metric_name}*.txt'))
        txt_data_files.sort()

        mean_dict = deepcopy(empty_dict)
        std_dict = deepcopy(empty_dict)


        for modelIdx, model_name in enumerate(order_list):
            txt_file = txt_data_files[modelIdx]
            assert model_name in txt_file.split('/')[-1].split('_Model_')[-1].split(f'_{method_name}_'), \
                'Something wrong with the reading of data. Check'
            with open(txt_file, 'r') as f:
                data_list = f.read().splitlines()
                # data_list = data_list[1:2001]
                data_list = [data_list[i] for i in [-1]]

                [(mean_dict[model_name].append(float(i.split(',')[1])),
                  std_dict[model_name].append(float(i.split(',')[2])))
                 for i in data_list]
        print(f'Done')

        # ## Check for NAN
        # orig_len = len(mean_dict['googlenet'])
        # # ipdb.set_trace()
        # pNans = np.argwhere(np.isnan(mean_dict['pytorch']))
        # mNans = np.argwhere(np.isnan(mean_dict['madry']))
        # gNans = np.argwhere(np.isnan(mean_dict['googlenet']))
        #
        # nan_idxs = reduce(np.union1d, (pNans, mNans, gNans))
        #
        # for data in [mean_dict, std_dict]:
        #     for key in data.keys():
        #         data[key] = np.delete(data[key], nan_idxs).tolist()
        #
        # f_len = orig_len - len(nan_idxs)
        # for data in [mean_dict, std_dict]:
        #     for key in data.keys():
        #         assert np.isnan(data[key]).any() == False, 'Something is worng in checking for nans'
        #         assert len(data[key]) == f_len, 'Something is worng in checking for nans'
        #
        # print(f'Nans removed.\nFinal no of images are {f_len}/{orig_len}')

        data_dicts[metric_name]['mean_dict'] = mean_dict
        data_dicts[metric_name]['std_dict'] = std_dict

    ## Plotting

    ## Error Bar Plot
    print(f'Plotting ErrorBar Plot ... ')
    dName = args.out_path
    fName = f'time_{f_time}_Error_Plot_Combined_Method_{method_name}.pdf'
    combined_errorbar_plot(data_dicts, fName=fName, out_dir=dName, save=args.save)
    print(f'Done')
    #################################################################################

    print(f'Time taken is {time.time() - s_time}')
    print(f'Time stamp is {f_time}\n')

########################################################################################################################
