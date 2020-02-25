import os
import cv2
import ipdb
import random
import argparse
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from skimage.transform import resize
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 10, 'font.weight':'bold'})
plt.rc("font", family="sans-serif")


if __name__ == '__main__':

    # Hyper parameters.
    parser = argparse.ArgumentParser(description='Processing Meaningful Perturbation data')
    parser.add_argument('--result_path',
                        default='./results/gradient/ILSVRC2012_val_00020735/',
                        type=str, help='filepath for the results')
    parser.add_argument('--input_img',
                        default='./Images/grad/ILSVRC2012_val_00020735.JPEG',
                        type=str, help='input image filepath')
    parser.add_argument('--add_noise',
                        default=0,
                        type=int, help='add noise to image')       
    parser.add_argument('--dataset',
                        default='imagenet',
                        type=str, help='dataset to run on imagenet | places365')
    parser.add_argument('--save_path',
                        default='',
                        type=str, help='path for saving images')

    args = parser.parse_args()

    # Read real image
    o_img = cv2.resize(cv2.cvtColor(cv2.imread(args.input_img, 1), cv2.COLOR_BGR2RGB), (224, 224))
    # print(o_img.shape)

    # Read generated heatmap
    if args.add_noise:
        heatmap_path = sorted([f for f in os.listdir(os.path.join(args.result_path)) if 'if_noise_1' in f])
        heatmap = [np.load(os.path.join(args.result_path, heatmap_path[ll])) for ll in [0, 2, 3, 1]]
        row_label=['Noisy']
    else:
        heatmap_path = sorted([f for f in os.listdir(os.path.join(args.result_path)) if 'if_noise_0' in f])
        heatmap = [np.load(os.path.join(args.result_path, heatmap_path[ll])) for ll in [0, 2, 3, 1]]
        row_label=['Clean']

    # Normalizing heatmaps
    heatmap = [i / np.max(np.abs(i)) for i in heatmap]
           
    # Make a list of all images to be plotted
    image_batch = [o_img]
    image_batch.extend(heatmap)
    col_label = []  # ['', 'GoogLeNet', 'GoogLeNet-R', 'ResNet', 'ResNet-R']
    zero_out_plot_multiple_patch_chirag([image_batch],
                                 folderName='./',
                                 row_labels_left=row_label,
                                 row_labels_right=[],
                                 col_labels=col_label,
                                 file_name=os.path.join(args.save_path, 'figure_noise_{}.jpg'.format(args.add_noise)))
