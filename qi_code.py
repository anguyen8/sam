import numpy as np
import torch
# import torchvision.models as models
import torchvision.transforms as transforms
from robustness import model_utils, datasets
from user_constants import DATA_PATH_DICT
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
import io
import glob
from PIL import Image, ImageDraw
import pdb
from torchvision.utils import save_image

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
        #transforms.Normalize([0.485, 0.456, 0.406],
        #                     [0.229, 0.224, 0.225])]
DATA = 'ImageNet'
arch = 'resnet50'
dataset_function = getattr(datasets, DATA)
dataset = dataset_function(DATA_PATH_DICT[DATA])

# Load model
model_kwargs = { 'arch': arch,
                 'dataset': dataset,
                 'resume_path': f'./models/{DATA}.pt',
                 'parallel':False}
model_kwargs['state_dict_path'] = 'model'
model, _ = model_utils.make_and_restore_model(**model_kwargs)
# net = models.resnet50(pretrained=True)
#net = models.alexnet(pretrained=True)

# net.cuda()
# net.eval()
model.cuda()
model.eval()
testdir='/home/naman/ImageNet_Val/test/013_n01534433_junco/'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
n=0
filenames = testdir + '*.JPEG'
for f in glob.glob(filenames):
    print(f)
    save_file = 'resize_' + f.split('/')[-1]
    image = Image.open(f).convert('RGB')
    image_tensor = preprocess(image)
    #image_tensor =  torch.clamp(image_tensor,0,1)
    #save_image(image_tensor,save_file)
    image_tensor = image_tensor.unsqueeze(0).cuda()
    out = model(image_tensor)
    #out = net(image_tensor)
    probs = nn.functional.softmax(out)
    pred_idx = torch.argmax(probs)
    #pdb.set_trace()

    print(f'prediction index: {pred_idx}, prob: {probs[0,pred_idx]}')
    if pred_idx == 13:
        n+=1
        print(f'prediction is correct. Total correction: {n}')