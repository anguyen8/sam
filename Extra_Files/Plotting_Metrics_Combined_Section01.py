import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import ipdb, os, time, argparse, sys, glob
from srblib import abs_path
import utils as eutils
import numpy as np
from copy import deepcopy
from functools import reduce

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 12, 'font.weight':'bold'})
plt.rc("font", family="sans-serif")


def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Paramters for sensitivity analysis of heatmaps')

    # parser.add_argument('-idp', '--input_dir_path', help='Path of the input directory', metavar='DIR')

    parser.add_argument('-op', '--out_path',
                        help='Path of the output directory where you want to save the results (Default is ./)')

    parser.add_argument('-mn', '--method_name', choices=['grad', 'inpgrad'],
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

    # if args.input_dir_path is None:
    #     print('Please provide image dir path. Exiting')
    #     sys.exit(1)
    # args.input_dir_path = os.path.abspath(args.input_dir_path)

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
def combined_errorbar_plot(ref_dicts, data_dicts, model, fName, out_dir, title=None, save=False):
    width = 0.6
    alpha = 1
    fig, ax = plt.subplots(figsize=(10, 6))
    eBar_color = 'dimgrey'
    err_kwargs = dict(elinewidth=1, capsize=0, markeredgewidth=0, ecolor=eBar_color)
    bar_colors = ['forestgreen', 'lime'] #,  'tomato', 'cornflowerblue', 'mediumseagreen']

    #################################################
    ## SSIM DATA
    ssim_mean_data = (np.mean(ref_dicts['ssim'][model]),
                      np.mean(data_dicts['ssim'][model]),
                      )

    ## SPEARMAN DATA
    spearman_mean_data = (np.mean(ref_dicts['spearman'][model]),
                          np.mean(data_dicts['spearman'][model]),
                          )

    ## HOG DATA
    hog_mean_data = (np.mean(ref_dicts['hog'][model]),
                      np.mean(data_dicts['hog'][model]),
                      )

    #################################################

    ##GOOGLENET (Above Centre)
    rects0 = ax.barh(2 * np.arange(3) + 0.7, (ssim_mean_data[0], spearman_mean_data[0], hog_mean_data[0]),
                     height=width, alpha=alpha, color=3*[bar_colors[0]], align='center',
                     label='Reference')

    ##PYTORCH (Centre)
    rects1 = ax.barh(2 * np.arange(3), (ssim_mean_data[1], spearman_mean_data[1], hog_mean_data[1]),
                     height=width, alpha=alpha, color=3*[bar_colors[1]], align='center',
                     label='Our Val')

    ax.set_yticks(2*np.arange(3), minor=False)
    ax.set_yticklabels(('SSIM', 'Spearman', 'HOG'), minor=False)

    if title is not None:
        ax.set_title(title, fontweight='bold', fontsize=13)
    ax.set_xticks(np.arange(0, 1.01, 0.1))

    ## (x, y, width, height)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, prop={'weight':'normal'})

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

    if method_name.lower() == 'grad':
        grad_grad_dir = '/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/' \
                        'Formal_Plot_Results/Comp_Results_All_Imgs/Grad_SG/Compare_Grad_with_Grad'
        grad_sg_dir = '/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/' \
                      'Formal_Plot_Results/Comp_Results_All_Imgs/Grad_SG/Compare_Grad_with_SG_500_Samples'
    else:

        grad_grad_dir = '/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/' \
                        'Formal_Plot_Results/Comp_Results_All_Imgs/InpGrad_IG/Compare_InpGrad_with_InpGrad'
        grad_sg_dir = '/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/' \
                      'Formal_Plot_Results/Comp_Results_All_Imgs/InpGrad_IG/Compare_InpGrad_with_IG_Samples_100_Trials_100'

    # metric_file_paths = [os.path.join(args.input_dir_path,
    #                            f'Method_{method_name}_Metric_{i}') for i in ['ssim', 'hog', 'spearman']]

    model_name_dir_flag = False

    empty_dict = {'pytorch': [],
                 'googlenet': [],
                 }

    grad_grad_data_dicts = {'ssim': {'mean_dict': deepcopy(empty_dict), 'std_dict': deepcopy(empty_dict)},
                            'hog': {'mean_dict': deepcopy(empty_dict), 'std_dict': deepcopy(empty_dict)},
                            'spearman': {'mean_dict': deepcopy(empty_dict), 'std_dict': deepcopy(empty_dict)},
                            }

    grad_sg_data_dicts = {'ssim': deepcopy(empty_dict),
                            'hog': deepcopy(empty_dict),
                            'spearman': deepcopy(empty_dict),
                            }

    grad_grad_file_paths = [os.path.join(grad_grad_dir,
                                         f'Method_{method_name}_Metric_{i}') for i in ['ssim', 'hog', 'spearman']]

    grad_sg_file_paths = [os.path.join(grad_sg_dir,
                                         f'Method_{method_name}_Metric_{i}') for i in ['ssim', 'hog', 'spearman']]


    metric_names = ['ssim', 'hog', 'spearman']


    for paths, data_dicts in zip([grad_grad_file_paths, grad_sg_file_paths],
                                 [grad_grad_data_dicts, grad_sg_data_dicts]):
        for metric_name in metric_names:
            mean_dict = deepcopy(empty_dict)
            path = [i for i in paths if metric_name in i]
            assert len(path) == 1, 'Something is wrong'
            path = path[0]
            txt_data_file = glob.glob(os.path.join(path,
                                                    f'*{method_name}*{metric_name}*.txt'))

            assert len(txt_data_file) == 1, 'Something is wrong'
            txt_data_file = txt_data_file[0]

            with open(txt_data_file, 'r') as f:
                data_list = f.read().splitlines()
                data_list = data_list[1:2001]

            [(mean_dict['googlenet'].append(float(i.split(',')[1])),
              mean_dict['pytorch'].append(float(i.split(',')[2])))
             for i in data_list]

            ## Check for NAN
            orig_len = len(mean_dict['googlenet'])
            # ipdb.set_trace()
            pNans = np.argwhere(np.isnan(mean_dict['pytorch']))
            gNans = np.argwhere(np.isnan(mean_dict['googlenet']))

            nan_idxs = reduce(np.union1d, (pNans, gNans))

            for data in [mean_dict, ]:
                for key in data.keys():
                    data[key] = np.delete(data[key], nan_idxs).tolist()

            f_len = orig_len - len(nan_idxs)
            for data in [mean_dict, ]:
                for key in data.keys():
                    assert np.isnan(data[key]).any() == False, 'Something is worng in checking for nans'
                    assert len(data[key]) == f_len, 'Something is worng in checking for nans'

            print(f'Nans removed.\nFinal no of images are {f_len}/{orig_len}')

            data_dicts[metric_name] = mean_dict


    ## Plotting

    ## Error Bar Plot
    print(f'Plotting ErrorBar Plot ... ')
    for model in ['pytorch', 'googlenet']:
        dName = os.path.join(args.out_path, f'Method_{method_name}')
        fName = f'time_{f_time}_Error_Plot_Combined_Method_{method_name}_Model_{model}.png'
        combined_errorbar_plot(grad_grad_data_dicts, grad_sg_data_dicts, model, fName=fName, out_dir=dName, save=args.save)
    print(f'Done')
    #################################################################################

    print(f'Time taken is {time.time() - s_time}')
    print(f'Time stamp is {f_time}\n')

########################################################################################################################




