import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json, ipdb

import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F


def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


img = get_image('/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/images/orig_madri_test/ILSVRC2012_val_00034217.JPEG')

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


## Sup
sup = 50
sup = 150

## Model
# model = models.googlenet(pretrained=True)
# file_name = f'LIME_slic_sup_{sup}_googlenet.jpeg'

model = models.resnet50(pretrained=True)
file_name = f'LIME_slic__sup_{sup}_resnet50.jpeg'

print(file_name)

idx2label, cls2label, cls2idx = [], {}, {}
with open(os.path.abspath('./imagenet_class_index.json'), 'r') as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
    cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}

img_t = get_input_tensors(img)
model.eval()
logits = model(img_t)


####################
resNet = models.resnet50(pretrained=True)
resNet.eval()
gNet = models.googlenet(pretrained=True)
gNet.eval()
resNet_logits = resNet(img_t)
resNet_probs = F.softmax(resNet_logits, dim=1).cpu().detach().numpy()
gNet_logits = gNet(img_t)
gNet_probs = F.softmax(gNet_logits, dim=1).cpu().detach().numpy()

plt.subplot(2, 2, 1)
plt.plot(resNet_probs[0, :])
plt.yscale('log',basey=10)
plt.subplot(2, 2, 2)
plt.plot(gNet_probs[0, :])
plt.yscale('log',basey=10)
plt.subplot(2, 2, 3)
mask = resNet_probs == resNet_probs.max(1, keepdims = 1)
resNet_probs_masked = np.ma.masked_array(resNet_probs, mask = mask)
plt.hist(resNet_probs_masked[0, :])
plt.subplot(2, 2, 4)
mask = gNet_probs == gNet_probs.max(1, keepdims = 1)
gNet_probs_masked = np.ma.masked_array(gNet_probs, mask = mask)
plt.hist(gNet_probs_masked[0, :])

plt.savefig('distribution_diff.jpeg')
ipdb.set_trace()
aa = 1
#################
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
print(test_pred.squeeze().argmax())

from lime.wrappers.scikit_image import SegmentationAlgorithm
from lime import lime_image


slic_parameters = {'n_segments': sup, 'compactness': 30, 'sigma': 3}
segmenter = SegmentationAlgorithm('slic', **slic_parameters)
seg = segmenter(np.array(pill_transf(img)))

explainer = lime_image.LimeImageExplainer(random_state=0)
explanation = explainer.explain_instance(np.array(pill_transf(img)),
                                         batch_predict, # classification function
                                         segmentation_fn=segmenter,
                                         top_labels=5,
                                         hide_color=0,
                                         num_samples=500,
                                         random_seed=0)

from skimage.segmentation import mark_boundaries

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=100, hide_rest=False)
img_boundry2 = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry2)
plt.savefig(file_name, orientation='landscape', dpi=224, transparent=True, frameon=False)


