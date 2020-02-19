import numpy as np
from PIL import Image
import skimage

print('Say your code requires noise generation')

print('Case 1: Noise first')
np.random.seed(0)
noise = np.random.normal(0, 1, (1, 5))
print(noise)
img = Image.open('/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/images/ILSVRC2012_val_00018430.JPEG').convert('RGB')


print('Case 2: Noise later')
np.random.seed(0)
img = Image.open('/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/images/ILSVRC2012_val_00018430.JPEG').convert('RGB')
noise = np.random.normal(0, 1, (1, 5))
print(noise)

print('Case 3: Noise twice')
np.random.seed(0)
img = Image.open('/home/naman/CS231n/heatmap_tests/Madri/Madri_New/robustness_applications/images/ILSVRC2012_val_00018430.JPEG').convert('RGB')
aa = skimage.util.random_noise(np.asarray(img), mode='gaussian',mean=0, var=0.1)
noise = np.random.normal(0, 1, (1, 5))
print(noise)