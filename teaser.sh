#/bin/bash
#
# Chirag Agarwal <chiragagarwall12.gmail.com>
# Naman Bansal <bnaman50@gmail.com>
# 2020

# Arguments
img_path='./Images/'
batch_size=1
add_noise=0
output_path='./results/Grad'

# Gradient
CUDA_VISIBLE_DEVICES=0 python Gradient_Madry.py -idp  ${img_path} -bs ${batch_size} -if_n ${add_noise} -op ${output_path}

# Save figure
# python formal_plot_figure.py --result_path ${save_path}/${algo_1} --dataset ${dataset} --save_path ${save_path}/${algo_1} --algo ${algo_1}

# convert ${save_path}/${algo_1}/figure_${algo_1}.jpg -trim ${save_path}/${algo_1}/figure_${algo_1}.jpg

# MP-G
# algo_2='MPG'
# perturb_binary_2=1

# CUDA_VISIBLE_DEVICES=0 python formal_MP_single_image.py --img_path ${img_path} --true_class ${true_class} --dataset ${dataset} --weight_file ${weight_file} --save_path ${save_path} --algo ${algo_2} --perturb_binary ${perturb_binary_2}

# Save figure
# python formal_plot_figure.py --result_path ${save_path}/${algo_2} --dataset ${dataset} --save_path ${save_path}/${algo_2} --algo ${algo_2}

# convert ${save_path}/${algo_2}/figure_${algo_2}.jpg -trim ${save_path}/${algo_2}/figure_${algo_2}.jpg

# Displaying figure
# montage -quiet ${save_path}/${algo_1}/figure_${algo_1}.jpg ${save_path}/${algo_2}/figure_${algo_2}.jpg -tile 1x -geometry +2+2 ${save_path}/test_MP.jpg
# imgcat ${save_path}/test_MP.jpg
