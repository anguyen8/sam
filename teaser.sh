#/bin/bash
#
# Chirag Agarwal <chiragagarwall12.gmail.com>
# Naman Bansal <bnaman50@gmail.com>
# 2020


## LIME ##
# Arguments
l_bp=0
num_superpixel=50
lime_explainer_seed=0
img_path='./Images/teaser/'
output_path='./results'

# Code
for lime_explainer_seed in 1 2 3 
do
    CUDA_VISIBLE_DEVICES=0 python LIME_Madry.py -idp ${img_path} -op ${output_path} -l_bp ${l_bp} -l_sn ${num_superpixel} -l_es ${lime_explainer_seed}
done


## Meaningful Perturbation ##
# Arguments
num_seed=0
init_mask='circular'
mask_size=28
save_evolution_mask=0
save_plot=0
save_npy=1
num_iter=300

# Code
for blur_radius in 5 10 30
do
    CUDA_VISIBLE_DEVICES=0 python MP_MADRY.py --seed ${num_seed} --img_dir_path ${img_path} --out_path ${output_path} --mask_init ${init_mask} --mask_init_size ${mask_size} --if_save_mask_evolution ${save_evolution_mask} --if_save_plot ${save_plot} --if_save_npy ${save_npy} --num_iter ${num_iter} --blur_radius ${blur_radius}
done


## Smooth Gradient ##
# Arguments
add_noise=0
index_flag=0
num_samples=50
std_dev=0.3

# Code
for num_samples in 25 50 75
do
    CUDA_VISIBLE_DEVICES=0 python SmoothGrad_Madry.py -idp ${img_path}  -if_n ${add_noise} --idx_flag ${index_flag} -op ${output_path} -n_sam ${num_samples} -std ${std_dev}
done


## Sliding-Patch ##
# Arguments
num_seed=0
patch_size=53

# Code
for patch_size in 5 29 53
do
    CUDA_VISIBLE_DEVICES=0 python Occlusion_Madry.py -np_s ${num_seed} -idp ${img_path}  -op ${output_path}  -ops ${patch_size}
done

## Plotting teaser image ## 
algo='sg'
python formal_plot_teaser.py --algo ${algo} --save_path ${output_path}
convert ${output_path}/figure_${algo}.jpg -trim ${output_path}/figure_${algo}.jpg

algo='occlusion'
python formal_plot_teaser.py --algo ${algo} --save_path ${output_path}
convert ${output_path}/figure_${algo}.jpg -trim ${output_path}/figure_${algo}.jpg

algo='lime'
python formal_plot_teaser.py --algo ${algo} --save_path ${output_path}
convert ${output_path}/figure_${algo}.jpg -trim ${output_path}/figure_${algo}.jpg

algo='mp'
python formal_plot_teaser.py --algo ${algo} --save_path ${output_path}
convert ${output_path}/figure_${algo}.jpg -trim ${output_path}/figure_${algo}.jpg

montage ${output_path}/figure_lime.jpg ${output_path}/figure_occlusion.jpg ${output_path}/figure_mp.jpg ${output_path}/figure_sg.jpg -tile 1x -geometry +0+0 ${output_path}/formal_teaser.jpg

# Display teaser image
echo 'Teaser path: '${output_path}'/formal_teaser.jpg'
imgcat ${output_path}/formal_teaser.jpg
