#/bin/bash
#
# Chirag Agarwal <chiragagarwall12.gmail.com>
# Naman Bansal <bnaman50@gmail.com>
# 2020

## Gradient ##
# Arguments
img_path='./Images/'
batch_size=1
add_noise=0
output_path='./results'

# Code
CUDA_VISIBLE_DEVICES=0 python Gradient_Madry.py -idp  ${img_path} -bs ${batch_size} -if_n ${add_noise} -op ${output_path}

## LIME ##
# Arguments
l_bp=0
num_superpixel=50
l_es=0

# Code
CUDA_VISIBLE_DEVICES=0 python LIME_Madry.py -idp ${img_path} -op ${output_path} -l_bp ${l_bp} -l_sn ${num_superpixel} -l_es ${l_es}

## Meaningful Perturbation ##
# Arguments
num_seed=0
init_mask='circular'
mask_size=224
save_evolution_mask=0
save_plot=0
save_npy=1
num_iter=300

# Code
CUDA_VISIBLE_DEVICES=0 python MP_MADRY.py --seed ${num_seed} --img_dir_path ${img_path} --out_path ${output_path} --mask_init ${init_mask} --mask_init_size ${mask_size} --if_save_mask_evolution ${save_evolution_mask} --if_save_plot ${save_plot} --if_save_npy ${save_npy} --num_iter ${num_iter}


## Smooth Gradient ##
# Arguments
add_noise=0
index_flag=0
num_samples=50
std_dev=0.3

# Code
CUDA_VISIBLE_DEVICES=0 python SmoothGrad_Madry.py -idp ${img_path}  -if_n ${add_noise} --idx_flag ${index_flag} -op ${output_path} -n_sam ${num_samples} -std ${std_dev}

## Sliding-Patch ##
# Arguments
num_seed=0
patch_size=53

# Code
python Occlusion_Madry.py -np_s ${num_seed} -idp ${img_path}  -op ${output_path}  -ops ${patch_size}
