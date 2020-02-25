#/bin/bash
#
# Chirag Agarwal <chiragagarwall12.gmail.com>
# Naman Bansal <bnaman50@gmail.com>
# 2020

## Gradient for robust and non-robust models ##
# Arguments
input_path='./Images/grad/'
batch_size=1
add_noise=0
output_path='./results/gradient/'

# Code
CUDA_VISIBLE_DEVICES=0 python Gradient_Madry.py -idp ${input_path} -bs ${batch_size} -if_n ${add_noise} -op ${output_path}

python formal_plot_gradient.py --save_path ${output_path} --add_noise ${add_noise}

convert ${output_path}figure_noise_${add_noise}.jpg -trim ${output_path}figure_noise_${add_noise}.jpg

add_noise=1
CUDA_VISIBLE_DEVICES=0 python Gradient_Madry.py -idp ${input_path} -bs ${batch_size} -if_n ${add_noise} -op ${output_path}

python formal_plot_gradient.py --save_path ${output_path} --add_noise ${add_noise}

convert ${output_path}figure_noise_${add_noise}.jpg -trim ${output_path}figure_noise_${add_noise}.jpg

# DISPLAY IMAGE
montage ${output_path}figure_noise_0.jpg ${output_path}figure_noise_1.jpg -tile 1x -geometry +0+0 ./results/formal_gradient.jpg

echo 'Image path: ./results/formal_gradient.jpg'
imgcat ./results/formal_gradient.jpg
