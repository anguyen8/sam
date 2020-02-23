import numpy as np
import argparse, sys, os, ipdb, glob, time
from srblib import abs_path
import utils as eutils

def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Input paramters for IG')

    parser.add_argument('-idp', '--input_dir_path', help='Path to the input dir', metavar='DIR')

    parser.add_argument('-op', '--out_path',
                        help=f'Path of the output directory (Default is ./)')

    parser.add_argument('-mn', '--method_name', choices=['grad', 'inpgrad'],
                        help='Method you are analysing')

    parser.add_argument('-s_idx', '--start_idx', type=int,
                        help='Start index for selecting images. Default: 0', default=0,
                        )

    parser.add_argument('-e_idx', '--end_idx', type=int,
                        help='End index for selecting images. Default: 2K', default=2000,
                        )

    parser.add_argument('--no_model_name_dir_flag', action='store_false', default=True,
                        help=f'Flag to say that model name is stored as seperate directory in the input path. '
                             f'Default=True')

    # Parse the arguments
    args = parser.parse_args()

    if args.method_name is None:
        print('Please provide the name of the method.\nExiting')
        sys.exit(0)

    if args.input_dir_path is None:
        print('Please provide path to image dir. Exiting')
        sys.exit(1)
    else:
        args.input_dir_path = os.path.abspath(args.input_dir_path)

    if args.out_path is None:
        args.out_path = './'
    args.out_path = os.path.abspath(args.out_path)

    return args


########################################################################################################################
def get_batch_idx(elem):
    return int(elem.split('batch_idx_')[-1].split('_')[0])


########################################################################################################################
if __name__ == '__main__':
    s_time = time.time()
    f_time = ''.join(str(s_time).split('.'))
    args = get_arguments()

    ############################################
    model_names = []
    # model_names.append('pytorch')
    # model_names.append('googlenet')
    # model_names.append('madry')
    model_names.append('madry_googlenet')

    method_dict = {'grad': 'Grad',
                   'inpgrad': 'InpxGrad',
                   'ig': 'IG',
                   'lime': 'Lime',
                   'mp': 'MP',
                   'occlusion': 'Occlusion',
                   'sg': 'SmoothGrad',
                   }
    method_name = method_dict[args.method_name]

    for model_name in model_names:
        modelTime = time.time()
        if args.no_model_name_dir_flag:
            dir_name = os.path.join(args.input_dir_path,
                                    f"{method_name}_{model_name}")
        else:
            dir_name = args.input_dir_path


        txt_filenames = []
        for file in glob.glob(os.path.join(dir_name, "*.txt")):
            txt_filenames.append(file)

        txt_filenames.sort(key=get_batch_idx)


        for txtIdx, txt_file in enumerate(txt_filenames):
            with open(txt_file, 'r') as f:
                img_paths = f.read().splitlines()

            txt_file_name = txt_file.split('/')[-1]
            temp_name = txt_file_name.split('img_paths_')[-1].split('.txt')[0] + '.npy'
            npy_file_name = glob.glob(os.path.join(dir_name, f'*{temp_name}'))
            assert len(npy_file_name) == 1, 'Something is wrong'


            npy_file = np.load(npy_file_name[0])

            for imIdx, img_path in enumerate(img_paths):
                img_name = img_path.split('/')[-1].split('.')[0]

                heat_dir = os.path.join(args.out_path, img_name)
                eutils.mkdir_p(heat_dir)

                temp_list = npy_file_name[0].split('/')[-1].split('_')
                out_npy_name = '_'.join(temp_list[:3] + \
                                        ['Img_Name', img_name, 'Idx', f'{imIdx:04d}'] + \
                                        temp_list[3:])
                np.save(os.path.join(heat_dir, out_npy_name), npy_file[imIdx])


    ##########################################
    print(f'Time stamp is {f_time}')
    print(f'Time taken is {time.time() - s_time}')
########################################################################################################################