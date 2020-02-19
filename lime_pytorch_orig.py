import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json, ipdb, sys

import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
from LIME_Madry import zero_out_plot_multiple_patch

print('Importing segmentation alg')
from lime.wrappers.scikit_image import SegmentationAlgorithm
print('Done')

def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


img = get_image('/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/images/orig_madri_test/ILSVRC2012_val_00034217.JPEG')
plt.imshow(img)
# plt.show()

# resize and take the center part of image to what our model expects
def get_input_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    return transf

def get_input_tensors(img):
    transf = get_input_transform()
    # unsqeeze converts single image to batch of 1
    return transf(img).unsqueeze(0)

# model = models.resnet50(pretrained=True)
# name = f'LIME_heatmap_resnet50.jpeg'
model = models.googlenet(pretrained=True)
name = f'LIME_heatmap_googlenet.jpeg'

idx2label, cls2label, cls2idx = [], {}, {}
with open(os.path.abspath('./imagenet_class_index.json'), 'r') as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
    cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}


img_t = get_input_tensors(img)
model.eval()
logits = model(img_t)

probs = F.softmax(logits, dim=1)
probs5 = probs.topk(5)
print(tuple((p,c, idx2label[c]) for p, c in zip(probs5[0][0].detach().numpy(), probs5[1][0].detach().numpy())))

def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])
    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    return transf

pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()


def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

test_pred = batch_predict([pill_transf(img)])
test_pred.squeeze().argmax()

from lime import lime_image
slic_parameters = {'n_segments': 150, 'compactness': 30, 'sigma': 3}

print('Creating segmenter')
segmenter = SegmentationAlgorithm('slic', **slic_parameters)
print('Done')
seg = segmenter(np.array(pill_transf(img)))

print('Creating explainer')
explainer = lime_image.LimeImageExplainer(random_state=0)
print('Explainer created')

# np.save('orig_image.npy', np.array(pill_transf(img)))
explanation = explainer.explain_instance(np.array(pill_transf(img)),
                                         batch_predict, # classification function
                                         segmentation_fn=segmenter,
                                         top_labels=1000,
                                         hide_color=0,
                                         num_samples=500,
                                         )

true_class = test_pred.squeeze().argmax()
print(f'True class is: {true_class}')
madry_segments = explanation.segments
# np.save('orig_segments.npy', madry_segments)
madry_heatmap = np.zeros(madry_segments.shape)
local_exp = explanation.local_exp
exp = local_exp[true_class]

for i, (seg_idx, seg_val) in enumerate(exp):
    madry_heatmap[madry_segments == seg_idx] = seg_val

orig_img = np.array(pill_transf(img)).astype('float32')/255

grid = [[orig_img, madry_heatmap]]

zero_out_plot_multiple_patch(grid,
                             os.path.abspath('./'),
                             row_labels_left=[],
                             row_labels_right=[],
                             col_labels=[],
                             file_name=name,
                             dpi=224,
                             )


