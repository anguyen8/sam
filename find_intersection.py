import argparse, os, sys, ipdb, time
import numpy as np
from functools import reduce

def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Input paramters for meaningful perturbation explanation of the image')

    parser.add_argument('-fdp', '--file_dir_path', help='Path of the text files directory', metavar='DIR')

    parser.add_argument('-op', '--out_path',
                        help='Path of the output directory where you want to save the results (Default is ./)'
                        )

    parser.add_argument('-np_s', '--np_seed', type=int,
                        help='Seed for numpy. Default is: 0',
                        default=0,
                        )

    # Parse the arguments
    args = parser.parse_args()

    np.random.seed(args.np_seed)

    if args.file_dir_path is None:
        print('Please provide image dir path. Exiting')
        sys.exit(1)
    args.file_dir_path = os.path.abspath(args.file_dir_path)

    if args.out_path is None:
        args.out_path = './'
    args.out_path = os.path.abspath(args.out_path)

    return args

if __name__ == '__main__':
    s_time = time.time()
    f_time = ''.join(str(s_time).split('.'))
    args = get_arguments()

    out_dir = args.out_path

    txt_files = ['googlenet', 'pytorch', 'madry']
    name_lists = {'googlenet': [],
                  'inception': [],
                  'pytorch': [],
                  'madry': [],
                  }

    str_name = f'time_15669152608009198_correctly_pred_imgs_model_name'
    time_str = str_name.split('time')[-1].split('_')[1]

    for t_file in txt_files:
        with open(os.path.join(args.file_dir_path, f'{str_name}_{t_file}.txt'), 'r') as f:
            for line in f:
                name_lists[t_file].append(line.split('\n')[0])
            name_lists[t_file] = np.asarray(name_lists[t_file], dtype=str)

    # out_inception = reduce(np.intersect1d, (name_lists['madry'], name_lists['pytorch'], name_lists['inception']))
    # print(f'Number of common and correctly predicted images with inception are {len(out_inception)}')

    out_googlenet = reduce(np.intersect1d, (name_lists['madry'], name_lists['pytorch'], name_lists['googlenet']))
    print(f'Number of common and correctly predicted images with googlenet are {len(out_googlenet)}')

    ## Randomly permute the images
    out_googlenet = np.random.permutation(out_googlenet)

    # path = os.path.join(out_dir, f'time_{time_str}_common_correct_imgs_model_names_madry_ressnet50_inception.txt')
    # print(f'Saving the text file here: {path}')
    # np.savetxt(path, out_inception, fmt='%s')

    path = os.path.join(out_dir, f'time_{time_str}_seed_{args.np_seed}_common_correct_imgs_model_names_madry_ressnet50_googlenet.txt')
    print(f'Saving the text file here: {path}')
    np.savetxt(path, out_googlenet, fmt='%s')

    print('Done')