import time, os, ipdb, sys, argparse, glob
import numpy as np
from PIL import Image
from srblib import abs_path
import utils as eutils
from torchvision.transforms import transforms
import torch

############################################################################
def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Input paramters for meaningful perturbation explanation of the image')

    parser.add_argument('--results_dir', help='Dir where results are stored', metavar='DIR')

    parser.add_argument('--out_path',
                        help=f'Path of the output directory where you want '
                             f'to save the results (Default is - results_dir/)')

    parser.add_argument('--method_name', choices=['grad', 'inpgrad', 'sg',
                                                  'ig', 'lime', 'mp', 'occlusion'],
                        help=f'Method Name.', default=None)

    # Parse the arguments
    args = parser.parse_args()

    if args.method_name is None:
        print('Please provide the name of the method.\nExiting')
        sys.exit(0)

    if args.results_dir is None:
        print('Please provide path to results dir where .npy files are stored.\n Exiting')
        sys.exit(1)
    else:
        args.results_dir = abs_path(args.results_dir)


    if args.out_path is None:
        args.out_path = args.results_dir
    args.out_path = abs_path(args.out_path)

    return args
########################################################################################################################

if __name__ == '__main__':
    base_img_dir = '/home/naman/CS231n/heatmap_tests/images/ILSVRC2012_img_val'
    s_time = time.time()
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

    col_labels = ['Orig Image', 'GoogleNet', 'ResNet50', 'MadryNet']

    img_filenames = []
    for file in glob.glob(os.path.join(args.results_dir, f"{method_name}_madry", '*', '')):
        if file[-1] == '/':
            file = file[:-1]
        img_filenames.append(file.split('/')[-1])
    img_filenames.sort()

    for idx, img_name in enumerate(img_filenames):
        im = Image.open(os.path.join(base_img_dir, f'{img_name}.JPEG'))
        preprocessFn = transforms.Compose([transforms.Resize((256, 256)),
                                                 transforms.CenterCrop(224),
                                                 ])
        im = preprocessFn(im)
        im = np.asarray(im).astype(float)/255
        grid = [im]

        for model_name in model_names:
            npy_str_list = glob.glob(os.path.join(args.results_dir,
                                                  f'{method_name}_{model_name}/{img_name}/*_{img_name}_*.npy'))
            assert len(npy_str_list) == 1, f'There should be just one .npy file.\n ' \
                                                f'Something is wrong'
            heatmap = np.load(npy_str_list[0])
            grid.append(heatmap)
        grid = [grid]

        out_dir = args.results_dir
        out_file_name = npy_str_list[0].split('/')[-1].split('.npy')[0]
        eutils.zero_out_plot_multiple_patch(grid,
                                            out_dir,
                                            row_labels_left=[],
                                            row_labels_right=[],
                                            col_labels=col_labels,
                                            file_name=f'{out_file_name}.jpeg',
                                            dpi=224,
                                            )

print(f'Time taken is {time.time() - s_time}')




