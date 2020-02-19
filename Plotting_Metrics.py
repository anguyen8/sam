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
plt.rcParams.update({'font.size': 10, 'font.weight':'bold'})
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

    parser.add_argument('--metric_name', choices=['ssim', 'spearman', 'hog', 'insertion', 'deletion', 'iou'],
                        help='Metric to be computed')

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

    if args.metric_name is None:
        print('Please provide the name of the metric.\nExiting')
        sys.exit(0)

    if args.out_path is None:
        args.out_path = './'
    args.out_path = os.path.abspath(args.out_path)

    return args


########################################################################################################################
def errorbar_plot(mean_dict, var_dict, xLabel, metric_name, fName, out_dir, title=None, save=False):
    pad = 15
    width = 0.5
    fig, ax = plt.subplots(figsize=(10, 6))

    mean_data = (np.mean(mean_dict['googlenet']),
                 np.mean(mean_dict['pytorch']),
                 np.mean(mean_dict['madry'])
                 )

    var_data = (np.sqrt(np.mean(var_dict['googlenet'])),
                 np.sqrt(np.mean(var_dict['pytorch'])),
                 np.sqrt(np.mean(var_dict['madry']))
                 )

    rects1 = ax.barh(np.arange(len(mean_dict)), mean_data, height=width, alpha=1.0,
                     color=['tomato', 'cornflowerblue', 'mediumseagreen'],  # 'cyan', '#8c564b', '#e377c2', '#17becf'],
                     xerr=var_data, align='center', error_kw=dict(lw=1))

    ax.set_xlabel(xLabel, fontweight='bold', labelpad=pad)
    ax.set_yticks(np.arange(len(mean_data)), minor=False)
    ax.set_yticklabels(('GoogleNet', 'ResNet50', 'MadryNet'), minor=False)

    if title is not None:
        ax.set_title(title, fontweight='bold', fontsize=13)
    ax.set_xticks(np.arange(0, 1.01, 0.1))

    if args.save:
        eutils.mkdir_p(out_dir)
        fig.savefig(os.path.join(out_dir, fName), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)



########################################################################################################################
def box_plot(data, xLabel, plot_label,
             metric_name, fName, out_dir, title=None, save=False, perc=[0, 100], fliers=True):
    pad = 15
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    labels = ['GoogleNet', 'ResNet50', 'MadryNet']
    medianprops = dict(color='white')
    meanpointprops = dict(markerfacecolor='yellow')
    bp = ax.boxplot((data['googlenet'], data['pytorch'], data['madry']),
                    whis=perc, patch_artist=True, labels=labels,
                    showmeans=True, medianprops=medianprops,
                    meanprops=meanpointprops, showfliers=fliers)
    colors = ['tomato', 'cornflowerblue', 'mediumseagreen']
    for bplot in (bp,):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    ax.set_xlabel(xLabel, fontweight='bold', labelpad=pad)
    ax.set_ylabel(f'Observed values of {plot_label} across 2K images', fontweight='bold', labelpad=20)
    if plot_label.lower() == 'mean':
        ax.set_yticks(np.arange(0, 1.01, 0.1))

    if title is not None:
        ax.set_title(title, fontweight='bold', fontsize=13)

    if save:
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
    metric_name = args.metric_name

    if metric_name == 'ssim':
        xLabel = 'SSIM'
    elif metric_name == 'hog':
        xLabel = 'Pearson Correlation of Hog filters'
    elif metric_name == 'spearman':
        xLabel = 'Spearman Rank Correlation'
    elif metric_name == 'insertion':
        xLabel = 'Insertion'
    elif metric_name == 'deletion':
        xLabel = 'Deletion'
    elif metric_name == 'iou':
        ## TODO
        print('Needs to be implemented.\nExiting')
        sys.exit(1)
        xLabel = 'IOU Scores'
    else:
        print(f'Not implemented.\nExiting')
        sys.exit(1)


    input_path = args.input_dir_path
    input_dir_name = input_path.split('/')[-1]

    aa = input_dir_name.split('_')

    assert aa[aa.index('Method')+1] == method_name, 'Input path does not match with the given Method name'
    assert aa[aa.index('Metric') + 1] == metric_name, 'Input path does not match with the given Metric name'


    if metric_name == 'iou':
        model_name_dir_flag = True
        input_path = input_path ## TODO Need to fix this as per the model
    else:
        model_name_dir_flag = False



    mean_dict = {'pytorch': [],
                 'googlenet': [],
                 'madry': []}

    var_dict = deepcopy(mean_dict)


    txt_data_files = glob.glob(os.path.join(input_path,
                                           f'*_Model_*_{method_name}_{metric_name}*.txt'))
    txt_data_files.sort()

    ## List for asserting the order of read
    order_list = ['googlenet', 'madry', 'pytorch']
    if metric_name == 'iou':
        ## TODO
        print('Needs to be implemented.\nExiting')
        sys.exit(1)
    else:
        ## Data read
        print(f'Reading Data ... ', end='')
        for modelIdx, model_name in enumerate(order_list):
            txt_file = txt_data_files[modelIdx]
            assert model_name in txt_file.split('/')[-1].split('_'), 'Something wrong with the reading of data. Check'
            with open(txt_file, 'r') as f:
                data_list = f.read().splitlines()
                data_list = data_list[1:2001]

                [(mean_dict[model_name].append(float(i.split(',')[1])),
                  var_dict[model_name].append(float(i.split(',')[2])))
                 for i in data_list]
        print(f'Done')

        ## Check for NAN
        orig_len = len(mean_dict['googlenet'])
        # ipdb.set_trace()
        pNans = np.argwhere(np.isnan(mean_dict['pytorch']))
        mNans = np.argwhere(np.isnan(mean_dict['madry']))
        gNans = np.argwhere(np.isnan(mean_dict['googlenet']))

        nan_idxs = reduce(np.union1d, (pNans, mNans, gNans))

        for data in [mean_dict, var_dict]:
            for key in data.keys():
                data[key] = np.delete(data[key], nan_idxs).tolist()

        f_len = orig_len - len(nan_idxs)
        for data in [mean_dict, var_dict]:
            for key in data.keys():
                assert np.isnan(data[key]).any() == False, 'Something is worng in checking for nans'
                assert len(data[key]) == f_len, 'Something is worng in checking for nans'

        print(f'Nans removed.\nFinal no of images are {f_len}/{orig_len}')


        ##

        ## Plotting

        ## Error Bar Plot
        print(f'Plotting ErrorBar Plot ... ')
        dName = args.out_path
        fName = f'time_{f_time}_Error_Plot_Method_{method_name}_Metric_{metric_name}.png'
        errorbar_plot(mean_dict, var_dict, xLabel,
                      metric_name, fName=fName, out_dir=dName, save=args.save)
        print(f'Done')
        #################################################################################

        ## Box Plot
        data_dicts = {'Mean': mean_dict, 'Var': var_dict}
        perc_dict = {'Mean': [5, 95], 'Var': [5, 95]}
        fliers_dict = {'Mean': True, 'Var': False}
        for plot_label in data_dicts.keys():
            print(f'Plotting BoxPlot for {plot_label}')
            data = data_dicts[plot_label]
            perc = perc_dict[plot_label]
            fliers = fliers_dict[plot_label]

            aa = {}
            if plot_label.lower() == 'var':
                for key in data.keys():
                    aa[key] = np.sqrt(data[key])
                data = aa



            dName = args.out_path
            fName = f'time_{f_time}_{plot_label}_BoxPlot_Method_{method_name}_Metric_{metric_name}.png'
            box_plot(data, xLabel, plot_label,
                     metric_name, fName=fName, out_dir=dName, save=args.save,
                     perc=perc, fliers=fliers)
            print('Done')

    print(f'Time taken is {time.time() - s_time}')
    print(f'Time stamp is {f_time}\n')

########################################################################################################################




