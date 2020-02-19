import numpy as np
import ipdb, glob, os
from shutil import copyfile

seed = 0
np.random.seed(seed)

input_dir = '/home/naman/CS231n/heatmap_tests/images/ILSVRC2012_img_val/'

count = 100
idxs = np.random.randint(low=0, high=50000, size=(count)).tolist()

img_filenames =[]
for file in glob.glob(os.path.join(input_dir, "*.JPEG")):
    img_filenames.append(file)
img_filenames.sort()

img_filenames = [img_filenames[i-1] for i in idxs]

dest = f'/home/naman/CS231n/heatmap_tests/images/random_count_{count}_seed_{seed}_ILSVRC2012_img_val/'

if not os.path.isdir(dest):
    os.mkdir(dest)

assert os.path.isdir(dest)

for file in img_filenames:
    img_name = file.split('/')[-1]
    copyfile(file, os.path.join(dest, img_name))

print(f'Files copied to destination: {dest}')
