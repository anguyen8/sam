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

add_noise=1
CUDA_VISIBLE_DEVICES=0 python Gradient_Madry.py -idp ${input_path} -bs ${batch_size} -if_n ${add_noise} -op ${output_path}


