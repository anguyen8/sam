import os
import cv2
import ipdb
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
from skimage.transform import resize
from matplotlib.colors import ListedColormap

def read_txt_file(txt_file):
    with open(txt_file, 'r') as f:
        data_list = f.read().splitlines()
        data_list = data_list[1:2001]
    return data_list


def zero_out_plot_multiple_patch(grid,
                  folderName,
                  row_labels_left,
                  row_labels_right,
                  col_labels,
                  file_name=None,
                  dpi=224,
                  title_label=''
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
    fontsize = 11
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
                if r < tRows:  # not r - 1:
                    if col_labels != []:
                        # ipdb.set_trace()
                        ax.set_xlabel(col_labels[c - 1],
                                     # + '\n' + f'max: {str(r_abs_max)}, min: {str(r_abs_min)}'
                                     horizontalalignment='center',
                                     verticalalignment='bottom',
                                     fontsize=fontsize, labelpad=11)
                        if c==4:
                            ax.set_title(title_label,
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
                if (not c - 1) or (not c - 2) or (not c - 4) or (not c - 6):
                    if row_labels_left != []:
                        txt_left = [l + '\n' for l in row_labels_left[r - 1]]
                        ax.set_ylabel(''.join(txt_left),
                                      rotation=0,
                                      verticalalignment='center',
                                      horizontalalignment='center',
                                      fontsize=fontsize)
                # else:
                if c == tCols-1:  # > 1 # != 1:
                    w_cbar = 0.009
                    h_cbar = h * 0.9  # scale
                    b_cbar = b
                    l_cbar = l + scale * w + 0.001
                    cbaxes = fig.add_axes([l_cbar+0.015, b_cbar+0.015, w_cbar, h_cbar])
                    cbar = fig.colorbar(im, cax=cbaxes)
                    cbar.outline.set_visible(False)
                    cbar.ax.tick_params(labelsize=4, width=0.2, length=1.2, direction='inout', pad=0.5)
                    tt = 1
                    cbar.set_ticks([])
                    cbar.set_ticks([-tt, zero, tt])
                    cbar.set_ticklabels([-1, zero, 1])

        #####################################################################################
    dir_path = folderName
    # print(f'Saving figure to {os.path.join(dir_path, file_name)}')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(os.path.join(dir_path, file_name), dpi=dpi / scale, transparent=True,
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def getbb_from_heatmap_cam(heatmap, size=None, thresh_val=0.5, thres_first=True):

    heatmap[heatmap < thresh_val] = 0
    if thres_first and size is not None:
        heatmap = resize(heatmap, size)
    bb_from_heatmap = np.zeros(heatmap.shape)

    if (heatmap == 0).all():
        if size is not None:
            bb_from_heatmap[1:size[1], 1:size[0]] = 1
            return bb_from_heatmap, [1, size[1]], [1, size[0]]
        else:
            bb_from_heatmap[1:heatmap.shape[1], 1:heatmap.shape[0]] = 1
            return bb_from_heatmap, [1, heatmap.shape[1]], [1, heatmap.shape[0]]

    x = np.where(heatmap.sum(0) > 0)[0] + 1
    y = np.where(heatmap.sum(1) > 0)[0] + 1
    bb_from_heatmap[y[0]:y[-1], x[0]:x[-1]] = 1

    return bb_from_heatmap, (x[0], x[-1]), (y[0], y[-1])


def preprocess_gt_bb(img, size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])

    preprocessed_img_tensor = transform(np.uint8(255 * img)).numpy()

    return preprocessed_img_tensor[0, :, :]


def generate_best_worst_figures_using_metrics(args, top_img_index, top_ssim, top_iou, save_path='./'):

    # Pre-processing function
    pytorch_preprocessFn = transforms.Compose([transforms.Resize((256, 256)),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor()
                                               ])
    result_batch = []
    label_batch = []
    color = (0, 255, 255)

    for i, f in enumerate(top_img_index):
        print(f)
        result_batch = []
        img_index = f
        o_img = cv2.cvtColor(cv2.imread(os.path.join(args.imagenet_path, 'ILSVRC2012_val_000{}.JPEG'.format(img_index)), 1),
                             cv2.COLOR_BGR2RGB)
        o_img_processed = pytorch_preprocessFn(Image.fromarray(o_img))

            
        # Ground truth bounding box
        # parse the xml for bounding box coordinates
        tree = ET.parse(os.path.join(args.xml_path, f'ILSVRC2012_val_000{img_index}.xml'))
        root = tree.getroot()
        img_processed_bb = np.uint8(255 * np.moveaxis(o_img_processed.numpy().transpose(), 0, 1)).copy()
        # Get Ground Truth ImageNet masks
        for type_tag in root.findall('object/bndbox'):
            xmin = int(type_tag[0].text)
            ymin = int(type_tag[1].text)
            xmax = int(type_tag[2].text)
            ymax = int(type_tag[3].text)
            gt_mask = np.zeros(o_img.shape)
            gt_mask[ymin:ymax, xmin:xmax] = 1

        gt = preprocess_gt_bb(gt_mask, 224)
        gt = (gt >= 0.5).astype(float)
        _, x, y = getbb_from_heatmap_cam(gt)
        cv2.rectangle(img_processed_bb, (x[0], y[0]), (x[1], y[1]), color, 2)

        data_1_path = os.path.join(args.heatmap_path, f'ILSVRC2012_val_000{img_index}')
        heatmap_names = sorted([j for j in os.listdir(data_1_path) if j.endswith('googlenet.npy')])
        heatmap_result = [np.load(os.path.join(data_1_path, j)) for j in heatmap_names]
        result_batch.append(img_processed_bb)
        for m in range(len(heatmap_names)):
            t_1 = np.arange(0.1, 0.51, 0.05)
            # print(int(args.path_2.split('.')[0][-1]))
            _, x, y = getbb_from_heatmap_cam(heatmap_result[m], size=gt.shape, thresh_val=t_1[int(args.path_2.split('.')[0][-1])] * heatmap_result[m].max())
            heatmap_disc = np.uint8(255 * np.moveaxis(o_img_processed.numpy().transpose(), 0, 1)).copy()
            cv2.rectangle(heatmap_disc, (x[0], y[0]), (x[1], y[1]), color, 2)

            heatmap_disc = resize(heatmap_disc, (224, 224))
            result_batch.append(resize(heatmap_result[m], (224, 224)))
            result_batch.append(heatmap_disc)
        
        label = ['Real', 'Circular', 'Circular-BB', 'Random', 'Random-BB', 'Ones', 'Ones-BB'] 
        zero_out_plot_multiple_patch([result_batch],
                                     folderName=save_path,
                                     row_labels_left=[],
                                     row_labels_right=[],
                                     col_labels=(label),
                                     title_label='SSIM: {:.3f} | IOU: {:.3f}'.format(top_ssim[i], top_iou[i]), 
                                     file_name=f'test_teaser_img_idx_{img_index}_{i:02d}.jpg',
                                     )


if __name__ == '__main__':

    # Hyper parameters.
    parser = argparse.ArgumentParser(description='Processing Meaningful Perturbation data')
    parser.add_argument('--imagenet_path', default='/home/naman/CS231n/heatmap_tests/images/ILSVRC2012_img_val/',
                        type=str, help='filepath for the example image')
    parser.add_argument('--xml_path', default='/home/naman/CS231n/heatmap_tests/images/ILSVRC2012_img_val_bb_xml/',
                        type=str, help='location of the imagenet bb xml files')
    parser.add_argument('--heatmap_path', default=f'/home/naman/CS231n/heatmap_tests/Madri/Madri_New/'
                                                  f'robustness_applications/results/MP/A20/Mask_Size_028/',
                        type=str, help='location of the heatmap results')
    parser.add_argument('--path_1',
                        default=f'/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/'
                                f'metric_result_text_files/MP/A20/Mask_Size_028/Method_MP_Metric_ssim/'
                                f'time_15686635513716524_Model_googlenet_MP_ssim.txt',
                        type=str, help='filepath')
    parser.add_argument('--path_2',
                        default=f'/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications'
                                f'/metric_result_text_files/MP/A20/Mask_Size_028/Method_MP_Metric_iou/'
                                f'Model_googlenet/time_15686794247601607_IOU_Model_googlenet_MP_iou_thresholdIdx_3.txt',
                        type=str, help='filepath')
    args = parser.parse_args()

# path_1 = '/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/metric_result_text_files/LIME/A02/Without_Noise/Method_Lime_Metric_ssim/time_15694285478539588_Model_googlenet_Lime_ssim.txt'
# path_2 = '/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/metric_result_text_files/LIME/A02/Without_Noise/Method_Lime_Metric_iou/Model_googlenet/time_15694287029344773_IOU_Model_googlenet_Lime_iou_thresholdIdx_3.txt'

img_index = np.array([f.split(',')[0].strip() for f in read_txt_file(args.path_1)])
ssim =  np.array([float(f.split(',')[1].strip()) for f in read_txt_file(args.path_1)])
IOU = np.array([float(f.split(',')[1].strip()) for f in read_txt_file(args.path_2)])

top_10_index = img_index[np.argsort(ssim)[-20:][::-1]]
top_10_ssim = ssim[np.argsort(ssim)[-20:][::-1]]
top_10_iou = IOU[np.argsort(ssim)[-20:][::-1]]

generate_best_worst_figures_using_metrics(args, top_10_index, top_10_ssim, top_10_iou, save_path='./temp_results/chirag_results/')
