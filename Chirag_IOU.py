import os
import cv2
import sys
import tqdm
import ipdb
import torch
import argparse
import numpy as np
from utils import *
import torch.nn as nn
from RISE_utils import *
from RISE_evaluation import CausalMetric, auc, gkern
from numpy import genfromtxt
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import torch.backends.cudnn as cudnn
from skimage.transform import resize
from torch.nn.functional import conv2d
# from pytorch_msssim import ssim, ms_ssim
from torchvision.transforms import transforms


def calculate_deletion_insertion_metric(b_img, b_attr_map, model, substrate, num_classes):

    cudnn.benchmark = True

    # Load black box model for explanations
    model = nn.Sequential(model, nn.Softmax(dim=1))
    model = model.eval()
    model = model.to('cuda')

    for p in model.parameters():
        p.requires_grad = False

    # To use multiple GPUs
    model = nn.DataParallel(model)

    # b_img = torch.stack(b_img, dim=0).reshape(-1, 3, 224, 224)
    # b_attr_map = np.stack(b_attr_map, axis=0)

    insertion = CausalMetric(model, 'ins', 224*8, substrate_fn=substrate, n_classes=num_classes)
    deletion = CausalMetric(model, 'del', 224*8, substrate_fn=torch.zeros_like, n_classes=num_classes)

    # Evaluate deletion
    # del_ins_metric = []
    # ipdb.set_trace()
    # for img, attr in tqdm.tqdm(zip(b_img, b_attr_map), total=len(b_img)):
    del_ins_metric = [deletion.single_run(b_img, b_attr_map, verbose=0),
                           insertion.single_run(b_img, b_attr_map, verbose=0)]
    # hh_del = deletion.evaluate(b_img.float(), b_attr_map, 2)
    # hh_ins = insertion.evaluate(b_img.cpu().float(), b_attr_map, 2)
    # ipdb.set_trace()
    # auc(hh_del.mean(1)), auc(hh_ins.mean(1)), hh_del.mean(1).mean(), hh_del.mean(1).std(),
    # hh_ins.mean(1).mean(), hh_ins.mean(1).std()
    return del_ins_metric


def calculate_MS_SSIM(hm_batch):
    ssim_score = []
    hm_batch = np.vstack(hm_batch)
    for _ in range(100):
        random_comb = np.random.permutation(hm_batch.shape[0])[:2]
        X = torch.from_numpy(hm_batch[random_comb[0]]).unsqueeze_(0).unsqueeze_(0)
        Y = torch.from_numpy(hm_batch[random_comb[1]]).unsqueeze_(0).unsqueeze_(0)
        if ms_ssim(X, Y, data_range=2) != ms_ssim(X, Y, data_range=2):
            ssim_score.append(0)
        else:
            ssim_score.append(ms_ssim(X, Y, data_range=2))
    return np.mean(ssim_score)


def getbb_from_heatmap_cam(heatmap, size=None, thresh_val=0.5, thres_first=True):

    heatmap[heatmap < thresh_val] = 0
    if thres_first and size is not None:
        heatmap = resize(heatmap, size)
    bb_from_heatmap = np.zeros(heatmap.shape)

    if (heatmap == 0).all():
        if size is not None:
            bb_from_heatmap[1:size[1], 1:size[0]] = 1
            return bb_from_heatmap
        else:
            bb_from_heatmap[1:heatmap.shape[1], 1:heatmap.shape[0]] = 1
            return bb_from_heatmap

    x = np.where(heatmap.sum(0) > 0)[0] + 1
    y = np.where(heatmap.sum(1) > 0)[0] + 1
    bb_from_heatmap[y[0]:y[-1], x[0]:x[-1]] = 1
    return bb_from_heatmap


def getbb_from_heatmap(heatmap, size=None, thresh_val=0.5, thres_first=True):

    percentile = np.percentile(heatmap, thresh_val*100)
    heatmap_f = np.zeros(heatmap.shape)
    heatmap_f[np.where(heatmap < percentile)] = 0
    heatmap_f[np.where(heatmap >= percentile)] = 1

    if thres_first and size is not None:
        heatmap_f = resize(heatmap_f, size)

    bb_from_heatmap = np.zeros(heatmap.shape)
    if (heatmap_f == 0).all():
        if size is not None:
            bb_from_heatmap[1:size[1], 1:size[0]] = 1
            return bb_from_heatmap
        else:
            bb_from_heatmap[1:heatmap.shape[1], 1:heatmap.shape[0]] = 1
            return bb_from_heatmap

    return heatmap_f


def calculate_WSL(gt, attr_map, max_heatmap_val=1.0):

    IOU = []
    ipdb.set_trace()
    for t in np.arange(0.05, 0.51, 0.05):
        bb_pred_mask = getbb_from_heatmap_cam(attr_map, size=gt.shape, thresh_val=t*max_heatmap_val)
        mask_intersection = cv2.bitwise_and(gt, bb_pred_mask.astype(np.float64))
        mask_union = cv2.bitwise_or(gt, bb_pred_mask.astype(np.float64))
        IOU.append(np.sum(mask_intersection) / np.sum(mask_union))

    ipdb.set_trace()

    return IOU


if __name__ == '__main__':
    # Hyper parameters.
    parser = argparse.ArgumentParser(description='Processing Meaningful Perturbation data')

    parser.add_argument('--image_1',
                        default='/home/naman/CS231n/heatmap_tests/images/ILSVRC2012_img_val/',
                        type=str, help='filepath for the example image')
    parser.add_argument('--image_places365',
                        default='/home/chirag/gpu3_codes/pytorch-explain-black-box/places365/val_256/',
                        type=str, help='filepath for the example image')
    parser.add_argument('--data_path',
                        default='/home/chirag/gpu3_codes/pytorch-explain-black-box/heatmap_LIME',
                        type=str, help='filepath for the saved heatmaps')
    parser.add_argument('--dataset',
                        default='imagenet',
                        type=str, help='dataset to run on imagenet | places365')
    parser.add_argument('--xml_path',
                        default='/home/naman/CS231n/heatmap_tests/images/ILSVRC2012_img_val_bb_xml/',
                        type=str, help='location of the imagenet bb xml files')
    parser.add_argument('--img_index',
                        default='-1', type=int, help='filepath for the inpainted image')
    parser.add_argument('--st_idx',
                        default=0, type=int, help='start index for 500 images')
    parser.add_argument('--end_idx',
                        default=100, type=int, help='end index for 500 images')
    parser.add_argument('--img_list',
                        default='/home/chirag/gpu3_codes/pytorch-explain-black-box/image-inpainting/e3_data_image_list.txt',
                        type=str, help='random selection of images from E3')
    parser.add_argument('--bin_size_1',
                        default=0.0,
                        type=float, help='lower limit of the bb relative area')
    parser.add_argument('--bin_size_2',
                        default=0.25,
                        type=float, help='upper limit of the bb relative area')
    parser.add_argument('--samp_size',
                        default=500,
                        type=int, help='number of images to sample from the given bin size')
    parser.add_argument('--metric',
                        type=str, default='IOU',
                        help='which metrics')
    parser.add_argument('--pkl_file',
                        default='/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/temp_results/Chirag_Img_Files/imagenet_val_bb_stats_imagenet_yolov3_reso_416_conf_0.15.pkl',
                        type=str, help='file containing all bb information')
    args = parser.parse_args()

    if args.dataset == 'imagenet':
        # model = load_model(arch_name='resnet50')

        # Fix the targeted class
        # label_map = load_imagenet_label_map()
        # id_map = load_imagenet_id_map()

        # Filtering images within the required bin sizes
        bb_stats = np.array(np.load(args.pkl_file, allow_pickle=True))
        bb_stats_id = bb_stats[:, :2]
        bb_stats_area = bb_stats[:, 2].astype(float)

        # Weight file for CA-inpainters
        weight_file = '/home/chirag/gpu3_codes/generative_inpainting_FIDO/model_logs/release_imagenet_256/'

    elif args.dataset == 'places365':
        model = load_model_places365(arch_name='resnet50')

        # load the class label
        label_map = load_class_label()

        # load validation data label
        gt_info = load_validation_label()

        # Filtering images within the required bin sizes
        bb_stats = np.array(np.load(args.pkl_file, allow_pickle=True))
        bb_stats_id = bb_stats[:, 0]
        bb_stats_area = bb_stats[:, 1].astype(float)

        # Weight file for CA-inpainter
        weight_file = '/home/chirag/gpu3_codes/generative_inpainting_FIDO/model_logs/release_places2_256/'
    else:
        print('Invalid datasest!!')
        exit(0)

    # model = torch.nn.DataParallel(model).to('cuda')
    # model.eval()

    # for p in model.parameters():
    #     p.requires_grad = False
    #
    if args.img_index == 1:
        random_sampl = [i for i in range(bb_stats.shape[0])]
        args.st_idx = 0
    else:

        # For new randomly sampled images
        # np.random.seed(seed=0)
        # random_sampl = list(set(np.random.choice(np.argwhere((args.bin_size_1 < bb_stats_area) &
        #                                                      (bb_stats_area < args.bin_size_2))[:, 0], args.samp_size)))
        random_sampl = [i for i in range(bb_stats.shape[0])]
        random_sampl = random_sampl[args.st_idx:args.end_idx]

    img_count = args.st_idx

    IOU_metric = []
    DEL_metric = []
    INS_metric = []
    WSL_err = 0
    imagenet_val_xml = sorted(os.listdir(args.xml_path))
    batch_imgs = []
    batch_exp = []
    pytorch_preprocessFn = transforms.Compose([transforms.Resize((256, 256)),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

    # Preparing substrate functions
    klen = 11
    ksig = 5
    kern = gkern(klen, ksig)

    # Function that blurs input image
    blur = lambda x: nn.functional.conv2d(x, kern, padding=klen // 2)
    del_ins_metric = []
    for ind, f in enumerate(tqdm.tqdm(random_sampl[:1])):
        # FOR VALIDATION DATA
        if args.dataset == 'imagenet':
            img_index = bb_stats_id[f, 1].split('.')[0]
            img_path = os.path.join(args.image_1, '{}.JPEG'.format(bb_stats_id[f, 1].split('.')[0]))
            original_img = cv2.imread(img_path, 1)
            pytorch_img = pytorch_preprocessFn(Image.open(img_path).convert('RGB')).cuda().unsqueeze(0)

        elif args.dataset == 'places365':
            img_index = bb_stats_id[f].split('.')[0]
            img_path = os.path.join(args.image_places365, bb_stats_id[f])
            original_img = cv2.imread(img_path, 1)
            pytorch_img = pytorch_preprocessFn(Image.open(img_path).convert('RGB')).cuda().unsqueeze(0)

        if args.dataset == 'imagenet':
            # category = id_map[bb_stats_id[f, 0]]
            # gt_category = list(label_map.keys())[list(label_map.values()).index(category)]
            num_classes = 1000
        elif args.dataset == 'places365':
            category = label_map[int(gt_info[bb_stats_id[f]])]
            gt_category = int(gt_info[bb_stats_id[f]])
            num_classes = 365

        data_path = os.path.join(args.data_path, args.dataset)

        # if 'MP' in args.data_path:
        #     heatmap_folder = [f for f in os.listdir(os.path.join(args.data_path, args.dataset))
        #                       if f.startswith('{:05d}'.format(ind + 1))][0]
        #     data_path = os.path.join(data_path, heatmap_folder)
        #     heatmap = 1 - np.load(os.path.join(data_path, [f for f in os.listdir(data_path)
        #                                                    if 'mask_00_299' in f][0]))
        #     thresh = 1
        # elif 'occlusion' in args.data_path:
        #     heatmap_folder = [f for f in os.listdir(os.path.join(args.data_path, args.dataset))
        #                       if f.startswith('{:03d}'.format(ind + 1))][0]
        #     data_path = os.path.join(data_path, heatmap_folder)
        #     heatmap = np.load(os.path.join(data_path, [f for f in os.listdir(data_path)
        #                                                if f.startswith('heatmap')][0]))
        # elif 'LIME' in args.data_path:
        #     heatmap_folder = [f for f in os.listdir(os.path.join(args.data_path, args.dataset))
        #                       if f.startswith('{:05d}'.format(ind + 1))][0]
        #     data_path = os.path.join(data_path, heatmap_folder)
        #     heatmap = np.load(os.path.join(data_path, [f for f in os.listdir(data_path)
        #                                                if f.startswith('mask_')][0]))
        # else:
        #     print('Invalid algorithm')
        #     exit(0)

        # outputs = torch.nn.Softmax(dim=1)(model(pytorch_img))
        # _, aind = outputs.max(dim=1)
        # if aind == gt_category:
        #     img_count += 1

        if args.metric == 'IOU':

            if args.dataset == 'imagenet':
                # parse the xml for bounding box coordinates
                tree = ET.parse(os.path.join(args.xml_path, f'{img_index}.xml'))

                root = tree.getroot()
                # Get Ground Truth ImageNet masks
                for type_tag in root.findall('object/bndbox'):
                    xmin = int(type_tag[0].text)
                    ymin = int(type_tag[1].text)
                    xmax = int(type_tag[2].text)
                    ymax = int(type_tag[3].text)
                    gt_mask = np.zeros(original_img.shape)
                    gt_mask[ymin:ymax, xmin:xmax] = 1
                # ipdb.set_trace()
                gt = (gt_mask >= 0.5).astype(float)
                ipdb.set_trace()

                ## #Random
                np.random.seed(seed=0)
                h = np.random.random((224, 224))

                # Ones
                # h = np.ones(heatmap.shape)

                IOU_metric.append(calculate_WSL(gt, h, h.max()))

                #IOU_metric.append(calculate_WSL(gt, heatmap, heatmap.max()))

        else:
            # creating batch for Deletion and Insertion metric calculation
            if heatmap.shape[0] != pytorch_img.shape[-1]:
                heatmap = resize(heatmap, (pytorch_img.shape[2], pytorch_img.shape[3]))
            # batch_imgs.append(pytorch_img.cpu())
            del_ins_metric.append(calculate_deletion_insertion_metric(pytorch_img.cpu(), heatmap, model, blur, num_classes))
            # For random heatmap baseline
            # batch_exp.append(np.random.random(heatmap.shape))
            # Real heatmaps
            # batch_exp.append(heatmap)

        img_count += 1

    print(f'# Images: {img_count}')
    if args.metric != 'IOU':
        # del_ins_metric = calculate_deletion_insertion_metric(batch_imgs, batch_exp, model, blur, num_classes)
        del_ins_metric = np.array(del_ins_metric)
        np.savetxt('DELINS_step_224_{}_{}.txt'.format(args.data_path.split('/')[-2], args.dataset), del_ins_metric,
                   fmt="%s")
        print("DEL-INS: {}+-{}".format(np.mean(del_ins_metric, axis=0), np.std(del_ins_metric, axis=0)))
    else:
        IOU_metric = np.array(IOU_metric)
        if args.metric == 'IOU':
            np.savetxt('IOU_CAM_ones_{}_{}.txt'.format(args.data_path.split('/')[-2], args.dataset), IOU_metric,
                       fmt="%s")
            print("WSL: {}+-{}".format(np.mean(IOU_metric < 0.5, axis=0) * 100, np.std(IOU_metric < 0.5, axis=0) * 100))
