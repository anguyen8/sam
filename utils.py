import torch
from torchvision import models
from torchvision.transforms import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import os, ipdb, glob, skimage, sys
from srblib import abs_path
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap
from PIL import Image
from naman_robustness import model_utils, datasets
from user_constants import DATA_PATH_DICT
import settings

# use_cuda = torch.cuda.is_available()
text_file = abs_path(settings.paper_img_txt_file)
# text_file = f'/home/naman/CS231n/heatmap_tests/' \
#             f'Madri/Madri_New/robustness_applications/img_name_files/' \
#             f'time_15669152608009198_seed_0_' \
#             f'common_correct_imgs_model_names_madry_ressnet50_googlenet.txt'
img_name_list = []
with open(text_file, 'r') as f:
    for line in f:
        img_name_list.append(line.split('\n')[0])


class DataProcessing:
    def __init__(self, data_path, transform, img_idxs=[0, 1], idx_flag=1, args=None):
        self.data_path = data_path
        self.transform = transform
        self.args = args

        if data_path == abs_path(settings.imagenet_val_path):
            aa = img_name_list[img_idxs[0]:img_idxs[1]]
            self.img_filenames = [os.path.join(data_path, f'{ii}.JPEG') for ii in aa]
            # self.img_filenames.sort()
        else:
            self.img_filenames = []
            for file in glob.glob(os.path.join(data_path, "*.JPEG")):
                self.img_filenames.append(file)
            self.img_filenames.sort()

        if idx_flag == 1:
            print('Only prodicing results for 1 image')
            img_idxs = [0]
            self.img_filenames = [self.img_filenames[i] for i in img_idxs]

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, self.img_filenames[index])).convert('RGB')
        y = self.get_image_class(os.path.join(self.data_path, self.img_filenames[index]))

        if self.args is not None:
            if self.args.if_noise == 1:
                img = skimage.util.random_noise(np.asarray(img), mode='gaussian',
                                                mean=self.args.noise_mean, var=self.args.noise_var,
                                                )  # numpy, dtype=float64,range (0, 1)
                img = Image.fromarray(np.uint8(img * 255))

        img = self.transform(img)
        return img, y, os.path.join(self.data_path, self.img_filenames[index])

    def __len__(self):
        return len(self.img_filenames)

    def get_image_class(self, filepath):
        base_dir = abs_path('~/CS231n/heatmap_tests/')

        # ipdb.set_trace()

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

###########################
def load_data(img_dir, preprocessFn, batch_size=1, img_idxs=[0, 1], idx_flag=1, args=None):
    data = DataProcessing(img_dir, preprocessFn,
                          img_idxs=img_idxs, idx_flag=idx_flag, args=args)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    return test_loader, len(data)

###########################
def imagenet_label_mappings():
    # fileName = '/home/naman/CS231n/heatmap_tests/Madri/Madri_New/' \
    #            'robustness_applications/imagenet_label_mapping'
    fileName = os.path.join(settings.imagenet_class_mappings, 'imagenet_label_mapping')
    with open(fileName, 'r') as f:
        image_label_mapping = {int(x.split(":")[0]): x.split(":")[1].strip()
                               for x in f.readlines() if len(x.strip()) > 0}
        return image_label_mapping

###########################
def load_orig_imagenet_model(arch='resnet50', if_pre=0, my_attacker=False, parallel=False):  # resnet50
    if arch == 'googlenet':
        print('Loading GoogleNet')
        model = models.googlenet(pretrained=True)
    elif arch == 'inception':
        print('Loading Inception')
        model = models.inception_v3(pretrained=True)
    else:
        print('Loading ResNet50')
        model = models.resnet50(pretrained=True)

    if if_pre == 1:
        pass
    else:
        model = nn.Sequential(model, nn.Softmax(dim=1))

    if parallel:
        model = torch.nn.DataParallel(model)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


    if torch.cuda.is_available():
        model.cuda()
    return model

###########################
## TODO: Rewrite every code using my_attacker = True
###########################
## 1. By default, Madry model return a tuple (logits, image). This has been chnaged to retun only logits

## 2. There is other paramterer called 'my_attacker' that has been introduced by us.
## This is done to so that we can use default PyTorch image pre-processing. Instead of using the ones provided by Madry guys
## Refer to `./robustness/attacker.py` for more details/implementation

## This was done at a later stage. So some codes use my_attacker = True and some set it to False
def load_madry_model(arch='madry', if_pre=0, my_attacker=False, parallel=False):
    DATA = 'ImageNet'  # Choices: ['CIFAR', 'ImageNet', 'RestrictedImageNet']

    dataset_function = getattr(datasets, DATA)
    dataset = dataset_function(DATA_PATH_DICT[DATA])

    print(f'My Attacker is {my_attacker}')
    # Load model

    if arch == 'madry':
        ## ResNet-50
        print(f'Loading the robust ResNet-50 architectre')
        model_kwargs = {
            'arch': 'resnet50',
            'dataset': dataset,
            'resume_path': f'./models/ResNet50_R.pt',
            'parallel': parallel,
            'my_attacker':my_attacker,
        }

    elif arch == 'madry_googlenet':
        print(f'Loading the robust GoogleNet architectre')
        ## GoogleNet
        model_kwargs = {
            'arch': 'googlenet',
            'dataset': dataset,
            'resume_path': f'./models/GoogLeNet_R.pt',
            'parallel': parallel,
            'my_attacker': my_attacker,
        }
    else:
        print(f'Architecture {arch} not implemented.\nExiting')
        sys.exit(1)


    model_kwargs['state_dict_path'] = 'model'
    model, _ = model_utils.make_and_restore_model(**model_kwargs)

    if if_pre == 1:
        pass
    else:
        model = nn.Sequential(model, nn.Softmax(dim=1))

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if torch.cuda.is_available():
        model.cuda()
    return model

###########################
## Plotting for zero_out (InpxGrad)
def zero_out_plot_multiple_patch(grid,
                                 folderName,
                                 row_labels_left,
                                 row_labels_right,
                                 col_labels,
                                 file_name=None,
                                 dpi=224,
                                 save=True,
                                 rescale=True
                                 ):
    plt.rcParams.update({'font.size': 5})
    plt.rc("font", family="sans-serif")

    plt.rc("axes.spines", top=True, right=True, left=True, bottom=True)
    image_size = (grid[0][0]).shape[0]

    nRows = len(grid)
    nCols = len(grid[0])

    tRows = nRows + 2  # total rows
    tCols = nCols + 1  # total cols

    wFig = tCols
    hFig = tRows  # Figure height (one more than nRows becasue I want to add xlabels to the top of figure)

    fig, axes = plt.subplots(nrows=tRows, ncols=tCols, figsize=(wFig, hFig))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    axes = np.reshape(axes, (tRows, tCols))

    #########
    ## Creating colormap
    uP = cm.get_cmap('Reds', 129)
    dowN = cm.get_cmap('Blues_r', 128)
    newcolors = np.vstack((
        dowN(np.linspace(0, 1, 128)),
        uP(np.linspace(0, 1, 129))
    ))
    cMap = ListedColormap(newcolors, name='RedBlues')
    cMap.colors[257 // 2, :] = [1, 1, 1, 1]
    #######

    scale = 0.80
    fontsize = 5

    for r in range(tRows):
        # if r <= 1:
        for c in range(tCols):
            ax = axes[r][c]
            l, b, w, h = ax.get_position().bounds
            ax.set_position([l, b, w * scale, h * scale])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

            if r > 0 and c > 0 and r < tRows - 1:
                img_data = grid[r - 1][c - 1]
                abs_min = np.amin(img_data)
                abs_max = np.amax(img_data)
                abs_mx = max(np.abs(abs_min), np.abs(abs_max))
                r_abs_min = round(np.amin(img_data), 2)
                r_abs_max = round(np.amax(img_data), 2)
                r_abs_mx = round(max(np.abs(abs_min), np.abs(abs_max)), 2)

                # Orig Image
                if c == 1:
                    im = ax.imshow(img_data, interpolation='none')

                else:
                    if rescale:
                        im = ax.imshow(img_data, interpolation='none', cmap=cMap, vmin=-abs_mx, vmax=abs_mx)
                    else:
                        im = ax.imshow(img_data, interpolation='none', cmap=cMap, vmin=-1, vmax=1)

                zero = 0
                if not r - 1:
                    if col_labels != []:
                        ax.set_title(col_labels[c - 1] + '\n' + f'max: {str(r_abs_max)}, min: {str(r_abs_min)}',
                                     horizontalalignment='center',
                                     verticalalignment='bottom',
                                     fontsize=fontsize, pad=5)

                if c == tCols - 2:

                    if row_labels_right != []:
                        txt_right = [l + '\n' for l in row_labels_right[r - 1]]
                        ax2 = ax.twinx()
                        ax2.set_xticks([])
                        ax2.set_yticks([])
                        ax2.spines['top'].set_visible(False)
                        ax2.spines['right'].set_visible(False)
                        ax2.spines['bottom'].set_visible(False)
                        ax2.spines['left'].set_visible(False)
                        ax2.set_ylabel(''.join(txt_right), rotation=0,
                                       verticalalignment='center',
                                       horizontalalignment='left',
                                       fontsize=fontsize)

                if not c - 1:

                    if row_labels_left != []:
                        txt_left = [l + '\n' for l in row_labels_left[r - 1]]
                        ax.set_ylabel(''.join(txt_left),
                                      rotation=0,
                                      verticalalignment='center',
                                      horizontalalignment='right',
                                      fontsize=fontsize)

                # else:
                if c > 1:  # != 1:
                    w_cbar = 0.005
                    h_cbar = h * scale
                    b_cbar = b
                    l_cbar = l + scale * w + 0.001
                    cbaxes = fig.add_axes([l_cbar, b_cbar, w_cbar, h_cbar])
                    cbar = fig.colorbar(im, cax=cbaxes)
                    cbar.outline.set_visible(False)
                    cbar.ax.tick_params(labelsize=4, width=0.2, length=1.2, direction='inout', pad=0.5)
                    tt = abs_mx
                    if rescale:
                        cbar.set_ticks([-tt, zero, tt])
                        cbar.set_ticklabels([-r_abs_mx, zero, r_abs_mx])
                    else:
                        cbar.set_ticks([-1, zero, 1])
                        cbar.set_ticklabels([-1, zero, 1])

        #####################################################################################

    if save:
        dir_path = folderName
        print(f'Saving figure to {os.path.join(dir_path, file_name)}')

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        plt.savefig(os.path.join(dir_path, file_name), orientation='landscape', dpi=dpi / scale, transparent=False,
                    frameon=False)
        plt.close(fig)
    else:
        plt.show()

    plt.close(fig)

##########################

def mkdir_p(mypath):

    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise
###########################

def return_transform(model_name):
    if model_name == 'madry':
        preprocessFn = transforms.Compose([transforms.Resize((256, 256)),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           ])

    elif model_name in ['pytorch', 'googlenet']:
        preprocessFn = transforms.Compose([transforms.Resize((256, 256)),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225]),
                                           ])
    return preprocessFn


###########################
def get_image_class(filepath):
    base_dir = abs_path('~/CS231n/heatmap_tests/')

    # ipdb.set_trace()

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


#################################
## Plotting for zero_out (InpxGrad)
def zero_out_plot_multiple_patch_chirag(grid,
                                        folderName,
                                        row_labels_left,
                                        row_labels_right,
                                        col_labels,
                                        file_name=None,
                                        dpi=224,
                                        save=True,
                                        rescale=True, flag=0
                                        ):

    plt.rcParams['axes.linewidth'] = 0.0  # set the value globally
    plt.rcParams.update({'font.size': 5})
    plt.rc("font", family="sans-serif")
    plt.rc("axes.spines", top=True, right=True, left=True, bottom=True)
    image_size = (grid[0][0]).shape[0]
    nRows = len(grid)
    nCols = len(grid[0])
    tRows = nRows + 2  # total rows
    tCols = nCols + 1  # total cols
    wFig = tCols
    hFig = tRows  # Figure height (one more than nRows becasue I want to add xlabels to the top of figure)
    fig, axes = plt.subplots(nrows=tRows, ncols=tCols, figsize=(wFig, hFig))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    axes = np.reshape(axes, (tRows, tCols))
    #########

    # Creating colormap
    uP = cm.get_cmap('Reds', 129)
    dowN = cm.get_cmap('Blues_r', 128)
    newcolors = np.vstack((
        dowN(np.linspace(0, 1, 128)),
        uP(np.linspace(0, 1, 129))
    ))
    cMap = ListedColormap(newcolors, name='RedsBlues')
    cMap.colors[257//2, :] = [1, 1, 1, 1]

    #######
    scale = 0.99
    fontsize = 15
    o_img = grid[0][0]
    for r in range(tRows):
        # if r <= 1:
        for c in range(tCols):
            ax = axes[r][c]
            l, b, w, h = ax.get_position().bounds
            ax.set_position([l, b, w * scale, h * scale])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

            if r > 0 and c > 0 and r < tRows - 1:
                img_data = grid[r - 1][c - 1]
                abs_min = np.amin(img_data)
                abs_max = np.amax(img_data)
                abs_mx = max(np.abs(abs_min), np.abs(abs_max))
                r_abs_min = round(np.amin(img_data), 2)
                r_abs_max = round(np.amax(img_data), 2)
                r_abs_mx = round(max(np.abs(abs_min), np.abs(abs_max)), 2)

                # Orig Image
                if r == 1 and c == 1:
                    im = ax.imshow(img_data, interpolation='none')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)

                else:
                    # im = ax.imshow(o_img, interpolation='none', cmap=cMap, vmin=-1, vmax=1)
                    im = ax.imshow(img_data, interpolation='none', cmap=cMap, vmin=-1, vmax=1)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    # save 1

                zero = 0
                if r < tRows:
                    if col_labels != []:
                        # import ipdb
                        # ipdb.set_trace()
                        if c == 1:
                            ax.set_xlabel(col_labels[c - 1],
                                          horizontalalignment='center',
                                          verticalalignment='bottom',
                                          fontsize=9, labelpad=17)
                        else:
                            temp_label = col_labels[c - 1].split(' ')
                            ax.set_xlabel(' '.join(temp_label[:2]) + '\n' + ' '.join(temp_label[-2:]),
                                          horizontalalignment='center',
                                          verticalalignment='bottom',
                                          fontsize=9, labelpad=21)
                if c == tCols - 2:
                    if row_labels_right != []:
                        txt_right = [l + '\n' for l in row_labels_right[r - 1]]
                        ax2 = ax.twinx()
                        ax2.set_xticks([])
                        ax2.set_yticks([])
                        ax2.spines['top'].set_visible(False)
                        ax2.spines['right'].set_visible(False)
                        ax2.spines['bottom'].set_visible(False)
                        ax2.spines['left'].set_visible(False)
                        ax2.set_ylabel(''.join(txt_right), rotation=0,
                                       verticalalignment='center',
                                       horizontalalignment='left',
                                       fontsize=fontsize)
                if c == 1:  # (not c - 1) or (not c - 2) or (not c - 4) or (not c - 6):
                    if row_labels_left != []:
                        txt_left = [l + '\n' for l in row_labels_left[r - 1]]
                        ax.set_ylabel(''.join(row_labels_left[0]),
                                      # rotation=0,
                                      # verticalalignment='center',
                                      # horizontalalignment='center',
                                      fontsize=fontsize)
                # else:
                if c == tCols-1 and flag==0:  # > 1 # != 1:
                    w_cbar = 0.009
                    h_cbar = h * 0.9  # scale
                    b_cbar = b
                    l_cbar = l + scale * w + 0.001
                    cbaxes = fig.add_axes([l_cbar + 0.015, b_cbar + 0.015, w_cbar, h_cbar])
                    cbar = fig.colorbar(im, cax=cbaxes)
                    cbar.outline.set_visible(False)
                    cbar.ax.tick_params(labelsize=5, width=0.2, length=1.2, direction='inout', pad=0.5)
                    tt = 1
                    cbar.set_ticks([])
                    cbar.set_ticks([-tt, zero, tt])
                    cbar.set_ticklabels([-1, zero, 1])
    if save:
        dir_path = folderName
        print(f'Saving figure to {os.path.join(dir_path, file_name)}')

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(os.path.join(dir_path, file_name), dpi=dpi / scale, transparent=True, bbox_inches='tight', pad_inches=0)
        # plt.savefig(os.path.join(dir_path, file_name), orientation='landscape', dpi=dpi / scale, transparent=False,
        #             frameon=False)
        plt.close(fig)
    else:
        plt.show()

    plt.close(fig)


#################################
## Plotting for zero_out (InpxGrad)
def zero_out_plot_multiple_patch_chirag_text(grid,
                                             folderName,
                                             row_labels_left,
                                             row_labels_right,
                                             col_labels,
                                             file_name=None,
                                             dpi=224,
                                             save=True,
                                             rescale=True, flag=0
                                             ):

    plt.rcParams['axes.linewidth'] = 0.0  # set the value globally
    plt.rcParams.update({'font.size': 5})
    plt.rc("font", family="sans-serif")
    plt.rc("axes.spines", top=True, right=True, left=True, bottom=True)
    image_size = (grid[0][0]).shape[0]
    nRows = len(grid)
    nCols = len(grid[0])
    tRows = nRows + 2  # total rows
    tCols = nCols + 1  # total cols
    wFig = tCols
    hFig = tRows  # Figure height (one more than nRows becasue I want to add xlabels to the top of figure)
    fig, axes = plt.subplots(nrows=tRows, ncols=tCols, figsize=(wFig, hFig))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    axes = np.reshape(axes, (tRows, tCols))
    #########

    # Creating colormap
    uP = cm.get_cmap('Reds', 129)
    dowN = cm.get_cmap('Blues_r', 128)
    newcolors = np.vstack((
        dowN(np.linspace(0, 1, 128)),
        uP(np.linspace(0, 1, 129))
    ))
    cMap = ListedColormap(newcolors, name='RedsBlues')
    cMap.colors[257//2, :] = [1, 1, 1, 1]

    #######
    scale = 0.99
    fontsize = 15
    o_img = grid[0][0]
    for r in range(tRows):
        # if r <= 1:
        for c in range(tCols):
            ax = axes[r][c]
            l, b, w, h = ax.get_position().bounds
            ax.set_position([l, b, w * scale, h * scale])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

            if r > 0 and c > 0 and r < tRows - 1:
                img_data = grid[r - 1][c - 1]
                abs_min = np.amin(img_data)
                abs_max = np.amax(img_data)
                abs_mx = max(np.abs(abs_min), np.abs(abs_max))
                r_abs_min = round(np.amin(img_data), 2)
                r_abs_max = round(np.amax(img_data), 2)
                r_abs_mx = round(max(np.abs(abs_min), np.abs(abs_max)), 2)

                # Orig Image
                if r == 1 and c == 1:
                    im = ax.imshow(img_data, interpolation='none')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)

                else:
                    # im = ax.imshow(o_img, interpolation='none', cmap=cMap, vmin=-1, vmax=1)
                    im = ax.imshow(img_data, interpolation='none', cmap=cMap)  # , vmin=-1, vmax=1)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    # save 1

                zero = 0
                if r < tRows:
                    if col_labels != []:
                        # import ipdb
                        # ipdb.set_trace()
                        if c == 1:
                            ax.set_xlabel(col_labels[c - 1],
                                          horizontalalignment='center',
                                          verticalalignment='bottom',
                                          fontsize=9, labelpad=17)
                        else:
                            temp_label = col_labels[c - 1].split(' ')
                            ax.set_xlabel(' '.join(temp_label[:2]) + '\n' + ' '.join(temp_label[-2:]),
                                          horizontalalignment='center',
                                          verticalalignment='bottom',
                                          fontsize=9, labelpad=21)
                if c == tCols - 2:
                    if row_labels_right != []:
                        txt_right = [l + '\n' for l in row_labels_right[r - 1]]
                        ax2 = ax.twinx()
                        ax2.set_xticks([])
                        ax2.set_yticks([])
                        ax2.spines['top'].set_visible(False)
                        ax2.spines['right'].set_visible(False)
                        ax2.spines['bottom'].set_visible(False)
                        ax2.spines['left'].set_visible(False)
                        ax2.set_ylabel(''.join(txt_right), rotation=0,
                                       verticalalignment='center',
                                       horizontalalignment='left',
                                       fontsize=fontsize)
                if c == 1:  # (not c - 1) or (not c - 2) or (not c - 4) or (not c - 6):
                    if row_labels_left != []:
                        txt_left = [l + '\n' for l in row_labels_left[r - 1]]
                        ax.set_ylabel(''.join(row_labels_left[0]),
                                      # rotation=0,
                                      # verticalalignment='center',
                                      # horizontalalignment='center',
                                      fontsize=fontsize)
                # else:
                if c == tCols-1 and flag==0:  # > 1 # != 1:
                    w_cbar = 0.009
                    h_cbar = h * 0.9  # scale
                    b_cbar = b
                    l_cbar = l + scale * w + 0.001
                    cbaxes = fig.add_axes([l_cbar + 0.015, b_cbar + 0.015, w_cbar, h_cbar])
                    cbar = fig.colorbar(im, cax=cbaxes)
                    cbar.outline.set_visible(False)
                    cbar.ax.tick_params(labelsize=5, width=0.2, length=1.2, direction='inout', pad=0.5)
                    tt = 1
                    # cbar.set_ticks([])
                    cbar.set_ticks([-tt, zero, tt])
                    # cbar.set_ticklabels([-1, zero, 1])
    if save:
        dir_path = folderName
        print(f'Saving figure to {os.path.join(dir_path, file_name)}')

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(os.path.join(dir_path, file_name), dpi=dpi / scale, transparent=True, bbox_inches='tight', pad_inches=0)
        # plt.savefig(os.path.join(dir_path, file_name), orientation='landscape', dpi=dpi / scale, transparent=False,
        #             frameon=False)
        plt.close(fig)
    else:
        plt.show()

    plt.close(fig)

#########################


