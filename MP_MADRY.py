import os, sys, time, ipdb, argparse, cv2, scipy, skimage, glob

import torch
import torch.optim
from torch.autograd import Variable
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torchvision.transforms as transforms
# from torch.utils.data import Dataset, TensorDataset

from srblib import abs_path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

from PIL import ImageFilter, Image

# from robustness import model_utils, datasets
from user_constants import DATA_PATH_DICT
import settings

import warnings
warnings.filterwarnings("ignore")

import utils as eutils

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

# os.environ.set("MAX_LEN_IDENTIFIER", 300)
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


########################################################################################################################
def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Input paramters for meaningful perturbation explanation of the image')

    # Add the paramters positional/optional (here only optional)
    parser.add_argument('--mask_init', default='circular', type=str, choices=['circular', 'ones', 'random'],
                        help='random|circular|ones. Default - circular')

    parser.add_argument('--mask_init_size', type=int,
                        help='Size of mask to be initilaized. Default=224', default=224,
                        )

    parser.add_argument('--img_dir_path', help='Path to the image directory')

    parser.add_argument('--out_path',
                        help='Path of the output directory where you want to save the results (Default is ./)')

    parser.add_argument('--tv_beta', type=float,
                        help='TV_Beta value', default=3.0,
                        )

    parser.add_argument('--tv_coeff', type=float,
                        help='TV Coefficient value', default=1e-2,
                        )

    parser.add_argument('--l1_coeff', type=float,
                        help='L1 coefficient value', default=1e-4,
                        )

    parser.add_argument('--category_coeff', type=float,
                        help='Category coefficient value', default=1,
                        )

    parser.add_argument('--learning_rate', type=float,
                        help='Learning rate', default=0.1,
                        )

    parser.add_argument('--num_iter', type=int,
                        help='Maximum Iterations', default=300,
                        )

    parser.add_argument('--seed', type=int,
                        help='Seed for reproducability.', default=None,
                        )

    parser.add_argument('--jitter', type=int,
                        help='Jitter. Default=4', default=4,
                        )

    parser.add_argument('--blur_radius', type=int,
                        help='Blur Radius. Default=10', default=10,
                        )

    parser.add_argument('--start_idx', type=int,
                        help='Start index for selecting images. Default: 0', default=0,
                        )

    parser.add_argument('--end_idx', type=int,
                        help='End index for selecting images. Default: 1735', default=1735,
                        )

    parser.add_argument('--idx_flag', type=int,
                        help=f'Flag whether to use some images in the folder (1) or all (0). '
                             f'This is just for testing purposes. '
                             f'Default=0', default=0,
                        )

    parser.add_argument('--if_save_npy', type=int, choices=range(2),
                        help='Flag whether to save npy version of masks or not. Default=Yes (1)', default=1,
                        )

    parser.add_argument('--if_save_plot', type=int, choices=range(2),
                        help='Flag whether to save plot or not. Default=No (0)', default=0,
                        )

    parser.add_argument('--if_save_mask_evolution', type=int, choices=range(2),
                        help='Flag whether to save evolution of mask or not. Default=No (0)', default=0,
                        )

    parser.add_argument('--if_noise', type=int, choices=range(2),
                        help='Flag whether to add Gaussian noise to the image or not before processing. Default=No (0)',
                        default=0,
                        )

    parser.add_argument('--noise_seed', type=int,
                        help='Seed for Gaussian noise. Default=0',
                        default=0,
                        )

    parser.add_argument('--noise_mean', type=float,
                        help='Mean of gaussian noise. Default: 0', default=0,
                        )

    parser.add_argument('--noise_var', type=float,
                        help='Variance of gaussian noise. Default: 0.1', default=0.1,
                        )

    # Parse the arguments
    args = parser.parse_args()

    if args.seed is not None:
        print(f'Using the numpy seed: {args.seed}')
        np.random.seed(seed=args.seed)

    if args.out_path is None:
        args.out_path = './'
    args.out_path = os.path.abspath(args.out_path) + '/'

    if args.img_dir_path is None:
        print("\nImage Dir Path not given.\nExiting")
        sys.exit(0)
    elif os.path.isdir(args.img_dir_path):
        args.img_dir_path = os.path.abspath(args.img_dir_path)
    else:
        print('\nIncorrect dir path.\nExiting\n')
        sys.exit(1)

    if args.num_iter < 0:
        parser.error("-mi/--num_iter: must be a positive integer")

    return args


########################################################################################################################
class DataProcessing:
    def __init__(self, data_path, img_idxs=[0, 1], idx_flag=1):
        self.data_path = data_path
        if data_path == abs_path(settings.imagenet_val_path):
            aa = img_name_list[img_idxs[0]:img_idxs[1]]
            self.img_filenames = [os.path.join(data_path, f'{ii}.JPEG') for ii in aa]
        else:
            self.img_filenames = []
            for file in glob.glob(os.path.join(data_path, "*.JPEG")):
                self.img_filenames.append(file)
            self.img_filenames.sort()
            self.img_filenames = self.img_filenames[:50]

        print(f'\nNo. of images to be analyzed are {len(self.img_filenames)}\n')
        if idx_flag == 1:
            print('Only prodicing results for 1 image')
            img_idxs = [0]
            self.img_filenames = [self.img_filenames[i] for i in img_idxs]

    def __getitem__(self, index):
        y = self.get_image_class(os.path.join(self.data_path, self.img_filenames[index]))
        return y, os.path.join(self.data_path, self.img_filenames[index])

    def __len__(self):
        return len(self.img_filenames)

    def get_image_class(self, filepath):
        base_dir = '/home/naman/CS231n/heatmap_tests/'
        # ImageNet 2012 validation set images?
        with open(os.path.join(settings.imagenet_class_mappings, "ground_truth_val2012")) as f:
        # with open(os.path.join(base_dir, "imagenet_class_mappings", "ground_truth_val2012")) as f:
            ground_truth_val2012 = {x.split()[0]: int(x.split()[1])
                                    for x in f.readlines() if len(x.strip()) > 0}

        with open(os.path.join(settings.imagenet_class_mappings, "synset_id_to_class")) as f:
        # with open(os.path.join(base_dir, "imagenet_class_mappings", "synset_id_to_class")) as f:
            synset_to_class = {x.split()[1]: int(x.split()[0])
                               for x in f.readlines() if len(x.strip()) > 0}

        def get_class(f):
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


########################################################################################################################
def load_data(img_dir, batch_size=1, img_idxs=[0, 1], idx_flag=1):
    data = DataProcessing(img_dir, img_idxs=img_idxs, idx_flag=idx_flag)
    test_loader = torch.utils.data.DataLoader(data, batch_size=1)
    return test_loader, len(data)


########################################################################################################################
def numpy_to_torch(img, requires_grad=True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    if use_cuda:
        output = output.to('cuda')  # cuda()

    output.unsqueeze_(0)
    v = Variable(output, requires_grad=requires_grad)
    return v


########################################################################################################################
def unnormalize(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    preprocessed_img = img.copy()
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] * stds[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] + means[i]
    return preprocessed_img


########################################################################################################################
def unnormalize_madry(img):
    means = [0, 0, 0]
    stds = [1, 1, 1]
    preprocessed_img = img.copy()
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] * stds[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] + means[i]
    return preprocessed_img


########################################################################################################################
def normalize(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    preprocessed_img = img.copy()
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] * stds[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] + means[i]
    preprocessed_img = np.expand_dims(preprocessed_img, 0)
    return preprocessed_img


########################################################################################################################
def create_blurred_circular_mask(mask_shape, radius, center=None, sigma=10):
    assert (len(mask_shape) == 2)
    if center is None:
        x_center = int(mask_shape[1] / float(2))
        y_center = int(mask_shape[0] / float(2))
        center = (x_center, y_center)
    y, x = np.ogrid[-y_center:mask_shape[0] - y_center, -x_center:mask_shape[1] - x_center]
    mask = x * x + y * y <= radius * radius
    grid = np.zeros(mask_shape)
    grid[mask] = 1

    if sigma is not None:
        grid = scipy.ndimage.filters.gaussian_filter(grid, sigma)
    return grid


########################################################################################################################
def create_blurred_circular_mask_pyramid(mask_shape, radii, sigma=10):
    assert (len(mask_shape) == 2)
    num_masks = len(radii)
    masks = np.zeros((num_masks, 3, mask_shape[0], mask_shape[1]))
    for i in range(num_masks):
        masks[i, :, :, :] = create_blurred_circular_mask(mask_shape, radii[i], sigma=sigma)
    return masks


########################################################################################################################
def test_circular_masks(model, o_img, m_size,
                        upsample, gt_category, preprocess_image,
                        radii=np.arange(0, 175, 5), thres=1e-2,
                        ):
    # net_transformer = get_ILSVRC_net_transformer(net)
    size = 224
    masks = create_blurred_circular_mask_pyramid((m_size, m_size), radii)
    masks = 1 - masks
    u_mask = upsample(torch.from_numpy(masks)).float().to('cuda')
    num_masks = len(radii)
    img = preprocess_image(np.float32(o_img) / 255, size)

    gradient = np.zeros((1, 1000))
    gradient[0][gt_category] = 1
    # ipdb.set_trace()
    scores = np.zeros(num_masks)
    batch_masked_img = []
    for i in range(num_masks):
        null_img = preprocess_image(get_blurred_img(np.float32(o_img)), size)  ##TODO: blurred image operating on BRG
        masked_img = img.mul(u_mask[i]) + null_img.mul(1 - u_mask[i])

        outputs = F.softmax(model(masked_img), dim=1)

        scores[i] = outputs[0, gt_category].cpu().detach()
        batch_masked_img.append(masked_img)
    img_output = torch.nn.Softmax(dim=1)(model(img)).cpu().detach()
    orig_score = img_output[0, gt_category]

    percs = (scores - scores[-1]) / float(orig_score - scores[-1])

    try:
        first_i = np.where(percs < thres)[0][0]
    except:
        first_i = -1
    return radii[first_i]


########################################################################################################################
def get_blurred_img(img, radius=10):
    img = Image.fromarray(np.uint8(img))
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius))
    return np.array(blurred_img) / float(255)


########################################################################################################################
def pytorch_preprocess_image(img, size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size + 32, size + 32)),  # 224+32 =256
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    preprocessed_img_tensor = transform(np.uint8(255 * img))

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    preprocessed_img = preprocessed_img_tensor.permute(1, 2, 0).numpy()[:, :, ::-1]
    preprocessed_img = (preprocessed_img - means) / stds

    if use_cuda:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img).to('cuda')
    else:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

    # preprocessed_img_tensor = torch.from_numpy(preprocessed_img_tensor)
    preprocessed_img_tensor.requires_grad = False
    preprocessed_img_tensor = preprocessed_img_tensor.permute(2, 0, 1)
    preprocessed_img_tensor.unsqueeze_(0)
    preprocessed_img_tensor = preprocessed_img_tensor.float()
    preprocessed_img_tensor.requires_grad = False
    return preprocessed_img_tensor


########################################################################################################################
def madry_preprocess_image(img, size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size + 32, size + 32)),  # 224+32 =256
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    preprocessed_img_tensor = transform(np.uint8(255 * img))

    means = [0, 0, 0]
    stds = [1, 1, 1]
    preprocessed_img = preprocessed_img_tensor.permute(1, 2, 0).numpy()[:, :, ::-1]
    preprocessed_img = (preprocessed_img - means) / stds

    if use_cuda:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img).to('cuda')
    else:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

    # preprocessed_img_tensor = torch.from_numpy(preprocessed_img_tensor)
    preprocessed_img_tensor.requires_grad = False
    preprocessed_img_tensor = preprocessed_img_tensor.permute(2, 0, 1)
    preprocessed_img_tensor.unsqueeze_(0)
    preprocessed_img_tensor = preprocessed_img_tensor.float()
    preprocessed_img_tensor.requires_grad = False
    return preprocessed_img_tensor


########################################################################################################################
def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta).sum()
    col_grad = torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta).sum()
    return row_grad + col_grad


########################################################################################################################
def create_random_maks(size, init):
    if init == 'random':
        mask = np.random.rand(size, size)
    elif init == 'ones':
        mask = np.ones((size, size))
    else:
        print('Incorrect Init!\nExiting')
        sys.exit(0)
    return mask


########################################################################################################################
def add_text(x, text, x_pt, size, scale):
    # --- Here I created a white background to include the text ---
    text_patch = np.zeros((25, x.shape[1], 3), np.uint8)
    text_patch[:] = (255, 255, 255)
    # --- I then concatenated it vertically to the image with the border ---
    vcat = cv2.vconcat((text_patch, x))
    # --- Now I included some text ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vcat, text, (x_pt, 15), font, size, (0, 0, 0), scale, 0)
    return vcat


########################################################################################################################
def save_mask(mask, label, label_prob, max_prob, max_label, save_path, ind, tot_iters, im_sz, f_time, model_name,
              **kwargs):
    # label is gt_category
    category_map_dict = eutils.imagenet_label_mappings()
    mask = get_blurred_img(255 * mask, 1)
    mask = 1 - mask
    aa = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_VIRIDIS)
    aa = cv2.resize(aa, (im_sz, im_sz))

    aa = add_text(aa,
                  'Target: {} {:.3f}'.format(category_map_dict[label].split(',')[0],
                                             label_prob),
                  **kwargs)
    # x_pt=50, scale=1, size=0.35)
    aa = add_text(aa,
                  'Top-1: {} {:.3f}'.format(category_map_dict[max_label].split(',')[0],
                                            max_prob),
                  **kwargs)

    aa = add_text(aa,
                  'Index is: {:3d}/{}'.format(ind,
                                              tot_iters),
                  **kwargs)

    temp_path = os.path.join(save_path, f'evolution_mask_time_{f_time}/{model_name}')
    eutils.mkdir_p(temp_path)
    cv2.imwrite(os.path.join(temp_path,
                             "Model_{}_{:03d}_mask_{}.png".format(model_name, ind, label)),
                aa)


########################################################################################################################
def add_gaussian_noise(orig_img, mean=0, var=0.1, seed=0):
    ## orig_img is BGR format
    aa = orig_img.copy()
    aa = aa[:, :, ::-1]  # converting BGR to RGB
    aa = skimage.util.random_noise(aa,
                                   mode='gaussian',
                                   mean=mean,
                                   var=var,
                                   seed=seed)  # numpy, dtype=float64,range (0, 1)
    aa = Image.fromarray(np.uint8(aa * 255))  # convert noisy Image to PIL format
    aa = np.asarray(aa)  # numpy image, dtype=uint8, range (0-255) (RGB format)
    aa = aa[:, :, ::-1]  # converting RGB to BGR
    return aa


########################################################################################################################
def save_init_mask(numpy_mask, save_path, img_name, f_time, model_name, save_npy=0, post_pro=0):
    if save_npy == 1:
        temp_path = os.path.join(save_path, f'time_{f_time}_'
                                            f'evolution_mask_'
                                            f'imN_{img_name}/Model_{model_name}')
        eutils.mkdir_p(temp_path)
        temp_npy_path = os.path.join(temp_path,
                                     f"imN_{int(img_name.split('_')[-1]):05d}_"
                                     f"postPro_{post_pro}_"
                                     f"init_mask_"
                                     f"{model_name}.npy")
        np.save(temp_npy_path, numpy_mask)


########################################################################################################################
if __name__ == '__main__':

    s_time = time.time()
    f_time = ''.join(str(s_time).split('.'))
    args = get_arguments()
    #######################

    ## #Hyperparameters
    img_shape = 224
    args.save_path = args.out_path
    tv_beta = args.tv_beta
    learning_rate = args.learning_rate
    max_iterations = args.num_iter
    l1_coeff = args.l1_coeff
    tv_coeff = args.tv_coeff
    size = 224
    jitter = args.jitter
    category_coeff = args.category_coeff
    blur_radius = args.blur_radius

    im_label_map = eutils.imagenet_label_mappings()

    ###################################
    data_loader, img_count = load_data(args.img_dir_path, batch_size=1,
                                       img_idxs=[args.start_idx, args.end_idx],
                                       idx_flag=args.idx_flag)

    ## DO NOT JUST MAKE ATTACKER MODEL AS FALSE IN THIS CODE

    model_names = []
    model_names.append('pytorch')
    model_names.append('googlenet')
    model_names.append('madry') #Robust_ResNet
    model_names.append('madry_googlenet') #Robust GoogleNet

    print(model_names)

    preprocessing_fns = {'pytorch': eval('pytorch_preprocess_image'),
                         'madry': eval('madry_preprocess_image'),
                         'madry_googlenet': eval('madry_preprocess_image'),
                         'googlenet': eval('pytorch_preprocess_image')}
    load_model_fns = {'pytorch': eval('eutils.load_orig_imagenet_model'),
                      'madry': eval('eutils.load_madry_model'),
                      'madry_googlenet': eval('eutils.load_madry_model'),
                      'googlenet': eval('eutils.load_orig_imagenet_model')}
    load_model_args = {'pytorch': 'resnet50',
                       'madry': 'madry',
                       'madry_googlenet': 'madry_googlenet',
                       'googlenet': 'googlenet'}
    unnormalize_fn_dict = {'pytorch': eval('unnormalize'),
                           'madry': eval('unnormalize_madry'),
                           'madry_googlenet': eval('unnormalize_madry'),
                           'googlenet': eval('unnormalize')}

    heatmaps = {'pytorch': 0, 'madry': 0, 'madry_googlenet': 0, 'googlenet': 0}
    probs_dict = {'pytorch': 0, 'madry': 0, 'madry_googlenet': 0, 'googlenet': 0}
    final_probs_dict = {'pytorch': 0, 'madry': 0, 'madry_googlenet': 0, 'googlenet': 0}
    prepro_images = {'pytorch': 0, 'madry': 0, 'madry_googlenet': 0, 'googlenet': 0}

    l1_loss_dict = {'pytorch': [], 'madry': [], 'madry_googlenet': [], 'googlenet': []}
    tv_loss_dict = {'pytorch': [], 'madry': [], 'madry_googlenet': [], 'googlenet': []}
    cat_loss_dict = {'pytorch': [], 'madry': [], 'madry_googlenet': [], 'googlenet': []}

    res_mask_npy = np.zeros((len(model_names), img_shape, img_shape))

    #########################################################
    ## #Upsampling fn
    if use_cuda:
        upsample = torch.nn.UpsamplingNearest2d(size=(size, size)).cuda()
    else:
        upsample = torch.nn.UpsamplingNearest2d(size=(size, size))

    ################################################################
    ## Out_path
    # mask_size = 224
    mask_size = args.mask_init_size

    par_name = f'sI_{args.start_idx:04d}_eI_{args.end_idx:04d}_' \
               f'iter_{max_iterations:03d}_' \
               f'blurR_{blur_radius:02d}_seed_{args.seed}_' \
               f'mT_{args.mask_init[:4]}_mS_{mask_size:03d}_' \
               f'ifN_{args.if_noise}_nS_{args.noise_seed}'
    print(f'Par_Name is {par_name}')

    if args.mask_init in ['random', 'ones']:
        orig_mask = create_random_maks(mask_size, args.mask_init)
    # elif args.mask_init == 'ones':
    #     orig_mask = create_random_maks(mask_size, args.mask_init)

    for idx, model_name in enumerate(model_names):
        print(f'\n\nAnalyzing for model: {model_name}')
        load_model = load_model_fns[model_name]
        model_arg = load_model_args[model_name]
        preprocess_image = preprocessing_fns[model_name]
        unnormalize_fn = unnormalize_fn_dict[model_name]

        ## Load Model
        print(f'Loading model: {model_name}')
        model = load_model(arch=model_arg, if_pre=1)  # Returns logits

        for ii, (targ_class, img_path) in enumerate(data_loader):
            batch_time = time.time()
            print(f'Analysing batch: {ii} of size {len(targ_class)}')

            targ_class = targ_class.cpu().item()
            gt_category = targ_class

            # print(f'Orig class label is {targ_class}')
            # print(f'Orig class name is {im_label_map[targ_class]}')
            img_name = img_path[0].split('/')[-1].split('.')[0]
            print(f'Image Name is {img_name}')
            out_dir = os.path.join(args.out_path, f'{img_name}')
            save_path = out_dir
            eutils.mkdir_p(out_dir)

            #####################################

            original_img = cv2.imread(img_path[0], 1)  # BGR Format
            if args.if_noise == 1:
                print('Adding gaussian noise to the image')
                original_img = add_gaussian_noise(original_img,
                                                  mean=args.noise_mean,
                                                  var=args.noise_var,
                                                  seed=args.noise_seed)  # BGR format

            shape = original_img.shape

            ## Preprocess Image
            print(f'Preprocessing image')
            img = np.float32(original_img) / 255
            img = preprocess_image(img, size + jitter)  # img
            prepro_images[model_name] = img

            ## Checking whether prediction matches the orig label
            # Will break if prediction does not match for any of the models
            outputs = F.softmax(model(img[:, :, :size, :size]), dim=1)
            pred_prob, pred_label = torch.max(outputs, dim=-1)
            pred_prob = pred_prob.cpu().item()
            pred_label = pred_label.cpu().item()
            print(f'Pred class is {pred_label} and prob is {pred_prob}')

            probs_dict[model_name] = pred_prob

            ##################################################
            print(f'Initializing with {args.mask_init} mask')
            if args.mask_init in ['random', 'ones']:
                mask = orig_mask.copy()
            else:
                # CAFFE mask_init
                mask_radius = test_circular_masks(model, original_img, mask_size,
                                                  upsample, gt_category, preprocess_image,
                                                  )
                print(f'Mask Radius is {mask_radius}')
                mask = 1 - create_blurred_circular_mask((mask_size, mask_size), mask_radius, center=None, sigma=10)

            # ipdb.set_trace()
            mask = numpy_to_torch(mask)

            #############################
            ## #Save initial mask
            ## if args.if_save_mask_evolution == 1:
            ##     aa = 1 - get_blurred_img(upsample(mask).data.cpu().numpy()[0, 0, :, :]*255,
            ##                                              radius=1)
                ## save_init_mask(aa,
                ##                save_path, img_name, f_time, model_name,
                ##                save_npy=args.if_save_npy, post_pro=1)

                # save_init_mask(upsample(mask).data.cpu().numpy()[0, 0],
                #                save_path, img_name, f_time, model_name,
                #                save_npy=args.if_save_npy, post_pro=0)
            ################################

            ## Blurred Image
            null_img = preprocess_image(get_blurred_img(np.float32(original_img), radius=blur_radius), size + jitter)

            ## Optimizer
            optimizer = torch.optim.Adam([mask], lr=learning_rate)

            ####################################################
            print("Optimizing.. ", end='')

            l1 = []  # l1 loss
            l2 = []  # tv_loss
            l3 = []  # category_loss
            cuml = []
            # iter_true_prob = []
            # iter_max_prob = []
            # iter_max_idx = []
            for i in range(max_iterations):

                if jitter != 0:
                    j1 = np.random.randint(jitter)
                    j2 = np.random.randint(jitter)
                else:
                    j1 = 0
                    j2 = 0

                upsampled_mask = upsample(mask)

                # The single channel mask is used with an RGB image,
                # so the mask is duplicated to have 3 channel,
                upsampled_mask = upsampled_mask.expand(1, 3, upsampled_mask.size(2), upsampled_mask.size(3))

                perturbated_input = img[:, :, j1:(size + j1), j2:(size + j2)].mul(upsampled_mask) + \
                                    null_img[:, :, j1:(size + j1), j2:(size + j2)].mul(1 - upsampled_mask)

                outputs = F.softmax(model(perturbated_input), dim=1)

                #######################
                ## Loss
                l1_loss = l1_coeff * torch.sum(torch.abs(1 - mask))
                tv_loss = tv_coeff * tv_norm(mask, tv_beta)
                cat_loss = category_coeff * outputs[0, gt_category]
                loss = l1_loss + tv_loss + cat_loss  ## total loss

                # For plotting the loss function
                l1.append(l1_loss.item())
                l2.append(tv_loss.item())
                l3.append(cat_loss.item())
                cuml.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                mask.data.clamp_(0, 1)

                #############################
                ## #Evolution plots

                if args.if_save_mask_evolution == 1:
                    max_prob, max_ind = outputs.max(dim=1)
                    kwargs = {'x_pt': 5, 'scale': 1, 'size': 0.35}
                    if args.if_save_plot == 1:
                        save_mask(upsample(mask).cpu().data.numpy()[0, 0, :],
                                  gt_category, outputs[0, gt_category].item(),
                                  max_prob.item(), max_ind.item(),
                                  save_path, i, max_iterations, img_shape, f_time, model_name, **kwargs)

                    if args.if_save_npy == 1:
                        # if (i+1)%10 == 0:
                        if i in [299, ]:
                            temp_path = os.path.join(save_path, f'time_{f_time}_'
                                                                f'evolution_mask_'
                                                                f'imN_{img_name}/Model_{model_name}')
                            eutils.mkdir_p(temp_path)
                            temp_npy_path = os.path.join(temp_path,
                                                    f"imN_{int(img_name.split('_')[-1]):05d}_"
                                                    f"postPro_1_"
                                                    f"iter_{i:03d}_"
                                                    f"iterProb_{outputs[0, gt_category].item():.3f}_"  
                                                    f"iterMaxProb_{max_prob.item():.3f}_"  # FMP - final_max_prob
                                                    f"iterMaxInd_{max_ind.item():3d}_"
                                                    f"{model_name}.npy")
                            t_mask = 1 - get_blurred_img(upsample(mask).data.cpu().numpy()[0, 0, :, :]*255,
                                                         radius=1)
                            np.save(temp_npy_path, t_mask)
                ################################
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                # mask.data.clamp_(0, 1)

            print('Done')
            ## End of Optimization
            ################################################
            if i == max_iterations - 1:
                final_max_prob, final_max_ind = outputs.max(dim=1)
                final_pred_prob = outputs[0, gt_category].cpu().detach().item()
                final_probs_dict[model_name] = final_pred_prob
                print(f'Prob after optimization is {outputs[0, gt_category]}')

            upsampled_mask = upsample(mask)
            mask = upsampled_mask
            mask = mask.cpu().detach().numpy()[0, 0, :]

            mask = get_blurred_img(255 * mask, radius=1)
            mask = 1 - mask

            if args.if_save_npy == 1:
                npy_path = os.path.join(save_path,
                                        f"mp_imN_{int(img_name.split('_')[-1]):05d}_"
                                        f"FTP_{final_pred_prob:.3f}_"  # FTP - final_true_prob
                                        f"FMP_{final_max_prob.item():.3f}_"  # FMP - final_max_prob
                                        f"FMInd_{final_max_ind.item():3d}_{par_name}_"  # FMInd - final_max_ind
                                        f"model_name_{model_name}.npy")
                # npy_path = os.path.join(save_path,
                #                         f"t_{f_time}_imN_{int(img_name.split('_')[-1]):05d}_"
                #                         f"FTP_{final_pred_prob:.3f}_" #FTP - final_true_prob
                #                         f"FMP_{final_max_prob.item():.3f}_" #FMP - final_max_prob
                #                         f"FMInd_{final_max_ind.item():3d}_{par_name}_" # FMInd - final_max_ind
                #                         f"{model_name}.npy")
                np.save(npy_path, mask)

            assert mask.shape[0] == img_shape
            # heatmaps[model_name] = mask
            # res_mask_npy[idx] = mask

            print(f'Batch time is {time.time() - batch_time}\n')
    #
    # #################################
    # if args.idx_flag == 1:
    #     if args.if_save_npy == 1:
    #         ## Saving npy files
    #         # TODO: ADD Orig Image and well as other details (orig_prob, pred_prob, img_path etc).
    #         # TODO: As well as label for each dimensions
    #         npy_path = os.path.join(save_path, f"NPY_{par_name}_time_{f_time}.npy")
    #         np.save(npy_path, res_mask_npy)
    #
    #     j1 = 0
    #     j2 = 0
    #     pytorch_img = prepro_images['pytorch']
    #     madry_img = prepro_images['madry']
    #
    #     pytorch_img = unnormalize(
    #         np.moveaxis(pytorch_img[:, :, j1:(size + j1), j2:(size + j2)][0, :].cpu().detach().numpy().transpose(), 0, 1))
    #     madry_img = unnormalize_madry(
    #         np.moveaxis(madry_img[:, :, j1:(size + j1), j2:(size + j2)][0, :].cpu().detach().numpy().transpose(), 0, 1))
    #
    #     assert np.amax(np.abs(pytorch_img - madry_img)) < 1e-7
    #
    #     ## Plotting
    #     grid = []
    #     grid.append([madry_img, heatmaps['googlenet'], heatmaps['pytorch'], heatmaps['madry']])
    #     # ipdb.set_trace()
    #     googlenet_prob = final_probs_dict['googlenet']
    #     resnet_prob = final_probs_dict['pytorch']
    #     madry_prob = final_probs_dict['madry']
    #     col_labels = ['Orig Image',
    #                   f'GoogleNet\nFinal_Prob:{googlenet_prob:.3f}',
    #                   f'ResNet_MP\nFinal_Prob:{resnet_prob:.3f}',
    #                   f'Madry_ResNet_MP\nFinal_Prob:{madry_prob:.3f}']
    #
    #     text = []
    #     text.append(("%.3f" % probs_dict['madry'],  # Madry prob (pL)
    #                  "%3d" % targ_class,  # Madry Label (pL)
    #                  "%.3f" % probs_dict['pytorch'],  # pytorch_prob (pL)
    #                  "%3d" % targ_class,  # Pytorch Label (pL)
    #                  "%.3f" % probs_dict['googlenet'],  # pytorch_prob (pL)
    #                  "%3d" % targ_class,  # Pytorch Label (pL)
    #                  "%3d" % targ_class,  # label for given neuron (cNL)
    #                  ))
    #
    #     madryProb, madryLabel, pytorchProb, pytorchLabel, googlenetProb, googlenetLabel, trueLabel = zip(*text)
    #     row_labels_left = [(f'Madry: Top-1:\n{im_label_map[int(madryLabel[i])]}: {madryProb[i]}\n',
    #                         f'ResNet: Top-1:\n{im_label_map[int(pytorchLabel[i])]}: {pytorchProb[i]}\n',
    #                         f'GoogleNet: Top-1:\n{im_label_map[int(googlenetLabel[i])]}: {googlenetProb[i]}\n',
    #                         f'Target Label: {int(trueLabel[i])}\n{im_label_map[int(trueLabel[i])]}')
    #                        for i in range(len(madryProb))]
    #
    #     row_labels_right = []
    #
    #     eutils.zero_out_plot_multiple_patch(grid,
    #                                         save_path,
    #                                         row_labels_left,
    #                                         row_labels_right,
    #                                         col_labels,
    #                                         file_name=f'MP_heatmap_{par_name}_time_{f_time}.jpeg',
    #                                         dpi=img_shape,
    #                                         )

    print(f'\nTime taken is {time.time() - s_time}')
    print(f'TIme stamp is {f_time}')
    aa = 1
########################################################################################################################
