import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from PIL import Image

import sys, glob, cv2, argparse, time, ipdb
import numpy as np

from utils import *

use_cuda = torch.cuda.is_available()

## For reproducebility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Input paramters for meaningful perturbation explanation of the image')

    parser.add_argument('-idp', '--img_dir_path', help='Path of the image directory', metavar='DIR')

    parser.add_argument('-op', '--out_path',
                        help='Path of the output directory where you want to save the results (Default is ./img_name/)')

    parser.add_argument('-gpu', '--gpu', type=int, choices=range(8),
                        help='GPU index', default=0,
                        )

    parser.add_argument('--idx_flag', type=int,
                        help=f'Flag whether to use some images in the folder (1) or all (0). '
                             f'This is just for testing purposes. '
                             f'Default=1', default=0,
                        )

    parser.add_argument('-ifp', '--if_pre', type=int, choices=range(2),
                        help='It is clear from name. Default: Post (0)', default=0,
                        )

    parser.add_argument('-bs', '--batch_size', type=int,
                        help='Size for batch of images. Default: 800', default=800,
                        )

    # Parse the arguments
    args = parser.parse_args()

    if args.img_dir_path is None:
        print('Please provide image dir path. Exiting')
        sys.exit(1)
    args.img_dir_path = os.path.abspath(args.img_dir_path)

    if args.out_path is None:
        args.out_path = './'
    args.out_path = os.path.abspath(args.out_path)

    return args

class DataProcessing:
    def __init__(self, data_path, transform, idx_flag=1):
        self.data_path = data_path
        self.transform = transform

        self.img_filenames = []
        for file in glob.glob(os.path.join(data_path, "*.JPEG")):
            self.img_filenames.append(file)
        self.img_filenames.sort()

        if idx_flag == 1:
            print('Using the provided img_idxs')
            img_idxs=[0]
            self.img_filenames = [self.img_filenames[i] for i in img_idxs]

    def __getitem__(self, index):
        # ipdb.set_trace()
        img = Image.open(os.path.join(self.data_path, self.img_filenames[index])).convert('RGB')
        y = self.get_image_class(os.path.join(self.data_path, self.img_filenames[index]))
        img = img.convert('RGB')
        img = self.transform(img)
        return img, y, self.img_filenames[index], self.img_filenames[index].split('/')[-1].split('.')[0]

    def __len__(self):
        return len(self.img_filenames)

    def get_image_class(self, filepath):
        # base_dir = '/home/naman/CS231n/heatmap_tests/'
        base_dir = '/home/naman/heatmap_tests/'
        # ImageNet 2012 validation set images?
        with open(os.path.join(base_dir, "imagenet_class_mappings", "ground_truth_val2012")) as f:
            ground_truth_val2012 = {x.split()[0]: int(x.split()[1])
                                    for x in f.readlines() if len(x.strip()) > 0}
        with open(os.path.join(base_dir, "imagenet_class_mappings", "synset_id_to_class")) as f:
            synset_to_class = {x.split()[1]: int(x.split()[0])
                               for x in f.readlines() if len(x.strip()) > 0}

        def get_class(f):
            # ipdb.set_trace()
            # File from ImageNet 2012 validation set
            ret = ground_truth_val2012.get(f, None)
            if ret is None:
                # File from ImageNet training sets
                ret = synset_to_class.get(f.split("_")[0], None)
            if ret is None:
                # Random JPEG file
                ret = 1000
            return ret

        image_class = get_class(filepath.split('/')[-1])
        return image_class

def load_data(img_dir, preprocessFn, batch_size=800, idx_flag=1):
    data = DataProcessing(img_dir, preprocessFn, idx_flag=idx_flag)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers=4)
    return test_loader, len(data)

if __name__ == '__main__':
    s_time = time.time()
    f_time = ''.join(str(s_time).split('.'))
    args = get_arguments()

    ############################################
    ## #Indices for images
    pytorch_preprocessFn = transforms.Compose([transforms.Resize((256, 256)),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

    inception_preprocessFn = transforms.Compose([transforms.Resize((299, 299)),
                                               transforms.CenterCrop(299),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

    madry_preprocessFn = transforms.Compose([transforms.Resize((256, 256)),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             ])

    pytorch_data_loader, img_count = load_data(args.img_dir_path, pytorch_preprocessFn,
                                               batch_size=args.batch_size ,idx_flag=args.idx_flag)
    inception_data_loader, img_count = load_data(args.img_dir_path, inception_preprocessFn,
                                               batch_size=args.batch_size, idx_flag=args.idx_flag)
    madry_data_loader, img_count = load_data(args.img_dir_path, madry_preprocessFn,
                                             batch_size=args.batch_size, idx_flag=args.idx_flag)

    ############################
    model_names = []
    model_names.append('pytorch')
    model_names.append('googlenet')
    model_names.append('madry')
    model_names.append('inception')

    data_loader_dict = {'pytorch': pytorch_data_loader,
                        'madry': madry_data_loader,
                        'googlenet': pytorch_data_loader,
                        'inception': inception_data_loader,
                        }

    load_model_fns = {'pytorch': eval('load_orig_imagenet_model'),
                      'madry': eval('load_madry_model'),
                      'googlenet': eval('load_orig_imagenet_model'),
                      'inception': eval('load_orig_imagenet_model'),
                      }
    im_sz_dict = {'pytorch': 224, 'madry': 224, 'googlenet': 224, 'inception':299}
    load_model_args = {'pytorch': 'resnet50', 'madry': 'madry', 'googlenet': 'googlenet', 'inception': 'inception'}

    ############################

    out_dir = args.out_path
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for idx, model_name in enumerate(model_names):
        print(f'\nAnalyzing for model: {model_name}')
        load_model = load_model_fns[model_name]
        model_arg = load_model_args[model_name]
        data_loader = data_loader_dict[model_name]
        im_sz = im_sz_dict[model_name]

        ## Load Model
        print(f'Loading model {model_arg}')
        model = load_model(arch=model_arg, if_pre=0)  # Returns probs

        file_names_list = []

        for i, (img, targ_class, img_path, img_name) in enumerate(data_loader):

            img_name = np.asarray(list(img_name), dtype=str)

            print(f'Analysing batch: {i}')
            targ_class = targ_class.cpu().numpy()
            if use_cuda:
                img = img.cuda(args.gpu)

            # Prob
            print('Post softmax analysis')
            probs = model(img)
            preds = torch.argmax(probs, dim=-1).cpu().numpy()
            img_name = img_name[np.where(preds == targ_class)]
            file_names_list.extend(list(img_name))

        path = os.path.join(out_dir, f'time_{f_time}_correctly_pred_imgs_model_name_{model_name}.txt')
        print(f'Saving the text file here: {path}')
        np.savetxt(path, np.asarray(file_names_list, dtype=str), fmt='%s')

    #######################
    print(f'Time string is {f_time}')
    print(f'Total time taken is {time.time() - s_time}')