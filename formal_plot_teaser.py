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
                        default='./results/ILSVRC2012_val_00002056/',
                        type=str, help='filepath for the results')
    parser.add_argument('--input_img',
                        default='./Images/ILSVRC2012_val_00002056.JPEG',
                        type=str, help='input image filepath')
    parser.add_argument('--algo',
                        default='sg',
                        type=str, help='sg|occlusion|lime|mp')
    parser.add_argument('--dataset',
                        default='imagenet',
                        type=str, help='dataset to run on imagenet | places365')
    parser.add_argument('--save_path',
                        default='',
                        type=str, help='path for saving images')

    args = parser.parse_args()

    if args.algo == 'lime':
        row_label = ['LIME']
    elif args.algo == 'occlusion':
        row_label = ['SP']
    elif args.algo == 'sg':
        row_label = ['SG']
    elif args.algo == 'mp':
        row_label = ['MP']
    else:
        print('Incorrect choice!!')
        exit(0)

    # Read real image
    o_img = cv2.resize(cv2.cvtColor(cv2.imread(args.input_img, 1), cv2.COLOR_BGR2RGB), (224, 224))
    # print(o_img.shape)

    # Read generated heatmap
    heatmap_path = sorted([f for f in os.listdir(os.path.join(args.result_path)) if f.endswith('pytorch.npy') and f.startswith(args.algo)])

    heatmap = [resize(np.load(os.path.join(args.result_path, ll)), (224, 224)) for ll in heatmap_path]

    # Normalizing heatmaps
    if args.algo == 'sg' or args.algo == 'lime':
        heatmap = [i / np.max(np.abs(i)) for i in heatmap]
        
    # Make a list of all images to be plotted
    image_batch = [o_img]
    image_batch.extend(heatmap)
    
    zero_out_plot_multiple_patch_chirag([image_batch],
                                 folderName='./',
                                 row_labels_left=row_label,
                                 row_labels_right=[],
                                 col_labels=[],
                                 file_name=os.path.join(args.save_path, 'figure_{}.jpg'.format(args.algo)))
