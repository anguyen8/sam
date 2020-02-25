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


    row_label = [args.algo]

    # Read real image
    o_img = cv2.resize(cv2.cvtColor(cv2.imread(args.input_img, 1), cv2.COLOR_BGR2RGB), (224, 224))
    # print(o_img.shape)

    # Read generated heatmap
    heatmap_path = [f for f in os.listdir(os.path.join(args.result_path)) if f.endswith('pytorch.npy') and f.startswith(args.algo)]

    heatmap = [resize(np.load(os.path.join(args.result_path, ll)), (224, 224)) for ll in heatmap_path]

    # Make a list of all images to be plotted
    image_batch = [o_img]
    image_batch.extend(heatmap)
    
    zero_out_plot_multiple_patch_chirag_text([image_batch],
                                 folderName='./',
                                 row_labels_left=row_label,
                                 row_labels_right=[],
                                 col_labels=[],
                                 file_name=os.path.join(args.save_path, 'figure_{}.jpg'.format(args.algo)))
