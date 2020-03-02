import argparse, time, os, ipdb
from torchvision.transforms import transforms
import xml.etree.ElementTree as ET
from srblib import abs_path
from RISE_utils import *
import utils as eutils
from PIL import Image
import numpy as np

def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Paramters for sensitivity analysis of heatmaps')

    parser.add_argument('-op', '--out_path',
                        help='Path of the output directory where you want to save the results (Default is ./img_name/)')

    parser.add_argument('-s_idx', '--start_idx', type=int,
                        help='Start index for selecting images. Default: 0', default=0,
                        )

    parser.add_argument('-e_idx', '--end_idx', type=int,
                        help='End index for selecting images. Default: 2K', default=2000,
                        )

    # Parse the arguments
    args = parser.parse_args()

    if args.out_path is None:
        args.out_path = './'
    args.out_path = os.path.abspath(args.out_path)

    return args


########################################################################################################################
def preprocess_gt_bb(img, size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])

    preprocessed_img_tensor = transform(np.uint8(255 * img)).numpy()
    return preprocessed_img_tensor[0, :, :]


########################################################################################################################
def get_true_bbox(img_path, base_xml_dir='~/CS231n/heatmap_tests/images/ILSVRC2012_img_val_bb_xml/'):
    # parse the xml for bounding box coordinates
    temp_img = Image.open(img_path)
    sz = temp_img.size # width x height
    im_name = img_path.split('/')[-1].split('.')[0]
    tree = ET.parse(os.path.join(abs_path(base_xml_dir), f'{im_name}.xml'))

    root = tree.getroot()
    # Get Ground Truth ImageNet masks
    for iIdx, type_tag in enumerate(root.findall('object/bndbox')):
        xmin = int(type_tag[0].text)
        ymin = int(type_tag[1].text)
        xmax = int(type_tag[2].text)
        ymax = int(type_tag[3].text)
        gt_mask = np.zeros((sz[1], sz[0])) #because we want rox x col
        gt_mask[ymin:ymax, xmin:xmax] = 1

    gt = preprocess_gt_bb(gt_mask, 224)
    gt = (gt >= 0.5).astype(float) #binarize after resize
    return gt


########################################################################################################################
if __name__ == '__main__':
    base_img_dir = '/home/naman/CS231n/heatmap_tests/images/ILSVRC2012_img_val'
    text_file = f'/home/naman/CS231n/heatmap_tests/' \
                f'Madri/Madri_New/robustness_applications/img_name_files/' \
                f'time_15669152608009198_seed_0_' \
                f'common_correct_imgs_model_names_madry_ressnet50_googlenet.txt'

    s_time = time.time()
    f_time = ''.join(str(s_time).split('.'))
    args = get_arguments()
    im_label_map = eutils.imagenet_label_mappings()
    my_attacker = True
    eutils.mkdir_p(args.out_path)

    #############################################
    ## #Inits
    bbox_ratio = []


    img_filenames = []
    with open(text_file, 'r') as f:
        img_filenames = f.read().splitlines()
        # img_filenames = img_filenames[args.start_idx:args.end_idx]

    print(f'No. of images are {len(img_filenames)}')

    for idx, img_name in enumerate(img_filenames):
        img_path = os.path.join(base_img_dir, f'{img_name}.JPEG')
        gt_box = get_true_bbox(img_path)
        bbox_ratio.append(np.sum(gt_box)/(gt_box.shape[0]*gt_box.shape[1]))

    print(f'Time taken is {time.time() - s_time}')
    print(f'Time stamp is {f_time}')
    ipdb.set_trace()
    aa = 1



