################################################################################################################
## IG generates n number of samples along the straight line path between baseline image and the orig image.
## In our implementation, we put all these samples on the cuda. So assign this value as per the gpu memory.
## Default value is 50
################################################################################################################
from __future__ import print_function
import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import ipdb, skimage

import sys, glob, ipdb, time, os, cv2, argparse, warnings
import numpy as np
from PIL import Image
from tqdm import tqdm
from srblib import abs_path
import utils as eutils
import settings

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_file = abs_path(settings.paper_img_txt_file)
# text_file = f'/home/naman/CS231n/heatmap_tests/' \
#             f'Madri/Madri_New/robustness_applications/img_name_files/' \
#             f'time_15669152608009198_seed_0_' \
#             f'common_correct_imgs_model_names_madry_ressnet50_googlenet.txt'
img_name_list = []
with open(text_file, 'r') as f:
    for line in f:
        img_name_list.append(line.split('\n')[0])

## For reproducebility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Input paramters for IG')

    parser.add_argument('-idp', '--img_dir_path', help='Path to the input image dir', metavar='DIR')

    parser.add_argument('-op', '--out_path',
                        help=f'Path of the output directory (Default is ./img_name/)')

    # ## IG wants gradients w.r.t probability
    # parser.add_argument('-ifp', '--if_pre', type=int, choices=range(2),
    #                     help='It is clear from name. Default: Post (0)', default=0,
    #                     )

    parser.add_argument('-n_mean', '--noise_mean', type=float,
                        help='Mean of gaussian noise. Default: 0', default=0,
                        )

    parser.add_argument('-n_var', '--noise_var', type=float,
                        help='Variance of gaussian noise. Default: 0.1', default=0.1,
                        )

    parser.add_argument('-n_seed', '--noise_seed', type=int,
                        help='Seed for the Gaussian noise. Default: 0', default=0,
                        )

    parser.add_argument('-b_seed', '--baseline_seed', type=int,
                        help='Seed for the Gaussian noise. Default: 0', default=0,
                        )

    parser.add_argument('-if_n', '--if_noise', type=int, choices=range(2),
                        help='Whether to add noise to the image or not. Default: 0', default=0,
                        )

    parser.add_argument('-s_idx', '--start_idx', type=int,
                        help='Start index for selecting images. Default: 0', default=0,
                        )

    parser.add_argument('-e_idx', '--end_idx', type=int,
                        help='End index for selecting images. Default: 1735', default=1735,
                        )

    parser.add_argument('--idx_flag', type=int,
                        help=f'Flag whether to use some images in the folder (1) or all (0). '
                             f'This is just for testing purposes. '
                             f'Default=0', default=0,
                        )

    parser.add_argument('-n_steps', '--num_steps', type=int,
                        help=f'Number of steps for SmoothGrad.'
                             f'Given our GPU memory, max value can be 100.'
                             f'Has to be positive integer.'
                             f' Default: 50',
                        default=50,
                        )

    parser.add_argument('--num_rand_trials', type=int,
                        help=f'Number of random trials'
                             f' Default: 10',
                        default=10,
                        )

    parser.add_argument('--baseline', choices=['random', 'zero', 'grey'],
                        help=f'Baseline image.'
                             f'If baseline is zero, num_rand_trails will be ignored and will be set to 1'
                             f' Default: random',
                        default='random',
                        )

    parser.add_argument('-if_sp', '--if_save_plot', type=int, choices=range(2),
                        help='Whether save the plots. Default: No (0)', default=0,
                        )

    parser.add_argument('-if_sn', '--if_save_npy', type=int, choices=range(2),
                        help='Whether save the plots. Default: Yes (1)', default=1,
                        )

    # Parse the arguments
    args = parser.parse_args()

    args.if_pre = 0

    if args.baseline == 'zero':
        args.num_rand_trials = 1
        print(f'Num of trails for all zero baseline is just 1')

    print(f'Num of random trails - {args.num_rand_trials}')

    if not args.num_rand_trials > 0:
        print('\nnum_rand_trails has to be a positive integer.\nExiting')
        sys.exit(0)

    if not args.num_steps > 0:
        print('\nnum_steps has to be a positive integer.\nExiting')
        sys.exit(0)

    # if args.noise_seed is not None:
    #     print(f'Setting the numpy seed with value: {args.noise_seed}')
    #     np.random.seed(args.noise_seed)

    if args.baseline == 'random':
        print(f'For the random baseline, Setting the numpy seed with value: {args.baseline_seed}')
        np.random.seed(args.baseline_seed)

    if args.img_dir_path is None:
        print('Please provide path to image dir. Exiting')
        sys.exit(1)
    else:
        args.img_dir_path = os.path.abspath(args.img_dir_path)

    if args.out_path is None:
        args.out_path = './'
    args.out_path = os.path.abspath(args.out_path)

    return args


########################################################################################################################
def integrated_gradients(
        inp, # torch, (1, 3, 224, 224), [0-1]
        model,
        device,
        target_label_index,
        predictions_and_gradients,
        baseline,
        steps=50):

    assert (baseline.shape == inp.shape)

    # Scale input and compute gradients.
    scaled_inputs = tuple(baseline + (float(i) / steps) * (inp - baseline) for i in range(0, steps + 1))
    scaled_inputs = torch.cat(scaled_inputs, dim = 0)

    predictions, grads = predictions_and_gradients(scaled_inputs,
                                                   model,
                                                   target_label_index,
                                                   device)  # shapes: <steps+1>, <steps+1, inp.shape>

    # Use trapezoidal rule to approximate the integral.
    # See Section 4 of the following paper for an accuracy comparison between
    # left, right, and trapezoidal IG approximations:
    # "Computing Linear Restrictions of Neural Networks", Matthew Sotoudeh, Aditya V. Thakur
    # https://arxiv.org/abs/1908.06214
    ## grads - [51, 224, 224, 3]
    grads = (grads[:-1] + grads[1:]) / 2.0  ## grads - [50, 224, 224, 3]
    avg_grads = torch.mean(grads, dim=0)  # (3, 224, 224)
    integrated_gradients = ((inp[0] - baseline[0]) * avg_grads) * 255  # shape: (3, 224, 224)
    return integrated_gradients, predictions # predictions - (51, 1000)


########################################################################################################################
def random_baseline_integrated_gradients(
        inp,  # torch, (1, 3, 224, 224), [0-1]
        model,
        target_label_index,
        device,
        predictions_and_gradients,
        steps=50,
        num_random_trials=10,
        baseline_type='random'):
    all_intgrads = []
    for i in range(num_random_trials):
        if baseline_type == 'random':
            baseline=torch.rand_like(inp)
        elif baseline_type == 'zero':
            baseline = torch.zeros_like(inp)
        elif baseline_type == 'grey':
            baseline = 255*torch.ones_like(inp) ##(255 * 0.485, 255 * 0.456, 255 * 0.406)
            baseline[:, 0] = baseline[:, 0] * 0.485
            baseline[:, 1] = baseline[:, 1] * 0.456
            baseline[:, 2] = baseline[:, 2] * 0.406
        else:
            print(f'This baseline has not been implemented\n.Exiting')
            sys.exit(1)

        intgrads, _ = integrated_gradients(
            inp,
            model,
            device,
            target_label_index=target_label_index,
            predictions_and_gradients=predictions_and_gradients,
            # baseline=255.0 * np.random.random([224, 224, 3]),
            baseline=baseline,
            steps=steps)
        all_intgrads.append(intgrads)

    all_intgrads = torch.stack(all_intgrads, dim=0)  # [10, 3, 224, 224]
    avg_intgrads = torch.mean(all_intgrads, dim=0) # [3, 224, 224]
    return avg_intgrads


########################################################################################################################
def comp_probs_and_grad(pre_pro_flag, if_pre=0):
    def calculate_outputs_and_gradients(inputs, model, target_label_idx, device):
        #inputs = batch of torch array [n, 3, 224, 224], [0-1]

        if pre_pro_flag:
            #do the preprocessing
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            mean = torch.as_tensor(mean, dtype=torch.float32, device=inputs.device)
            std = torch.as_tensor(std, dtype=torch.float32, device=inputs.device)

            x_ch0 = (torch.unsqueeze(inputs[:, 0], 1) - mean[0]) / std[0]
            x_ch1 = (torch.unsqueeze(inputs[:, 1], 1) - mean[1]) / std[1]
            x_ch2 = (torch.unsqueeze(inputs[:, 2], 1) - mean[2]) / std[2]
            inputs = torch.cat((x_ch0, x_ch1, x_ch2), 1)
            ## This is correct. Checked

        inputs = inputs.to(device)
        inputs = Variable(inputs, requires_grad=True)

        probs = model(inputs)

        ## Gradients
        model.zero_grad()
        num_samples = inputs.shape[0]
        repeated_targ_class = torch.cat(num_samples * [torch.tensor([target_label_idx])])
        ones = torch.ones(repeated_targ_class.shape).to(device)
        sel_nodes = probs[torch.arange(num_samples), repeated_targ_class]
        sel_nodes.backward(ones)
        probs = probs.cpu()
        grads = inputs.grad.cpu()  # [50, 3, 224, 224]
        return probs, grads

    return calculate_outputs_and_gradients


########################################################################################################################
if __name__ == '__main__':
    s_time = time.time()
    f_time = ''.join(str(s_time).split('.'))
    args = get_arguments()
    args.noise_mean = 0  ##Explicity set to zero

    im_label_map = eutils.imagenet_label_mappings()

    ############################################
    ## #Data Loader
    preprocessFn = transforms.Compose([transforms.Resize((256, 256)),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       #                      std=[0.229, 0.224, 0.225]),
                                       ])

    data_loader, img_count = eutils.load_data(args.img_dir_path, preprocessFn, batch_size=1,
                                              img_idxs=[args.start_idx, args.end_idx],
                                              idx_flag=args.idx_flag, args=args)

    print(f'Total number of images to be analyzed are {img_count}')

    ############################
    model_names = []
    model_names.append('pytorch') #ResNet
    model_names.append('googlenet') #GoogleNet
    model_names.append('madry') #ResNet_R
    model_names.append('madry_googlenet')  #GoogleNet_R

    if not model_names:
        print('Please provide at least one model to analyze.\nExiting')
        sys.exit(1)
    print(model_names)


    parallel = True
    print(f'Flag whether running code in parallel: {parallel}')
    load_model_fns = {'pytorch': eval('eutils.load_orig_imagenet_model'),
                      'madry': eval('eutils.load_madry_model'),
                      'madry_googlenet': eval('eutils.load_madry_model'),
                      'googlenet': eval('eutils.load_orig_imagenet_model')}

    im_sz_dict = {'pytorch': 224,
                  'madry': 224,
                  'madry_googlenet': 224,
                  'googlenet': 224}
    load_model_args = {'pytorch': 'resnet50',
                       'madry': 'madry',
                       'madry_googlenet': 'madry_googlenet',
                       'googlenet': 'googlenet'}

    preprocessing_flag_dict  = {'pytorch': True,
                                'madry': False,
                                'madry_googlenet': False,
                                'googlenet': True}

    ############################
    for idx, model_name in enumerate(model_names):
        print(f'\nAnalyzing for model: {model_name}')
        load_model = load_model_fns[model_name]
        model_arg = load_model_args[model_name]
        im_sz = im_sz_dict[model_name]
        preprocessing_flag = preprocessing_flag_dict [model_name]
        probs_and_grad_fun = comp_probs_and_grad(preprocessing_flag, if_pre=args.if_pre)

        ## Load Model
        print(f'Loading model {model_arg}')
        model = load_model(arch=model_arg, if_pre=args.if_pre, parallel=parallel)  # Returns probs

        par_name = f'baseline_{args.baseline[:4]}_num_steps_{args.num_steps:03d}_' \
                   f'num_rand_trails_{args.num_rand_trials:02d}_' \
                   f'start_idx_{args.start_idx}_' \
                   f'end_idx_{args.end_idx}_baseline_seed_{args.baseline_seed}_' \
                   f'noise_seed_{args.noise_seed}_' \
                   f'if_noise_{args.if_noise}_noise_mean_{args.noise_mean}_' \
                   f'noise_var_{args.noise_var}_model_name_{model_name}'
        print(f'Par name is - {par_name}')

        for i, (img, targ_class, img_path) in enumerate(data_loader):
            # img - [1, 3, 224, 224], [0-1]
            batch_time = time.time()
            print(f'Analysing batch: {i} of size {len(targ_class)}')

            ## Creating the save path
            img_name = img_path[0].split('/')[-1].split('.')[0]
            print(f'Image Name is {img_name}')
            out_dir = os.path.join(args.out_path, f'IG_{model_name}/{img_name}')
            eutils.mkdir_p(out_dir)
            # print(f'Saving results in {out_dir}')

            targ_class = targ_class.cpu()

            # ## Get grad function and prob value
            # probs, gradients = probs_and_grad_fun(img, model, targ_class.item(), device)
            ## gradients - (1, 3, 224, 224)
            probs, grads = probs_and_grad_fun(img, model, targ_class.item(), device)
            grads = grads.cpu().data.numpy()[0]
            grads = np.rollaxis(grads, 0, 3)
            grads = np.mean(grads, axis=-1)
            pred_prob = probs[0, targ_class.item()]

            attributions = random_baseline_integrated_gradients(img, model, targ_class.item(),
                                                                device, probs_and_grad_fun,
                                                                steps=args.num_steps,
                                                                num_random_trials=args.num_rand_trials,
                                                                baseline_type=args.baseline
                                                                ) # (3, 224, 224)

            attributions = attributions.cpu().data.numpy() # (3, 224, 224)
            attributions = np.rollaxis(attributions, 0, 3) # (224, 224, 3)
            attributions = np.mean(attributions, axis=-1)


            if args.if_save_npy == 1:
                np.save(os.path.join(out_dir,
                                     f'time_{f_time}_{img_name}_prob_{pred_prob:.3f}_'
                                     f'heatmaps_num_steps_{args.num_steps}_{par_name}.npy'),
                        attributions)

            ## Only saving the Madry results
            if args.if_save_plot == 1:
                orig_img = img.cpu().data.numpy()[0]
                orig_img = np.rollaxis(orig_img, 0, 3)

                grid = []
                grid.append([orig_img, grads, attributions])
                col_labels = ['Orig Image', 'Grad', 'IG']
                row_labels_left = []
                row_labels_right = []

                eutils.zero_out_plot_multiple_patch(grid,
                                                    out_dir,
                                                    row_labels_left,
                                                    row_labels_right,
                                                    col_labels,
                                                    file_name=f'time_{f_time}_{img_name}_'
                                                              f'heatmaps_'
                                                              f'{par_name}.jpeg',
                                                    dpi=224,
                                                    )
            print(f'Time taken for batch is {time.time() - batch_time}')

    ##########################################
    print(f'Time stamp is {f_time}')
    print(f'Time taken is {time.time() - s_time}')
########################################################################################################################
