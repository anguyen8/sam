#/bin/bash
#
# Chirag Agarwal <chiragagarwall12.gmail.com>
# Naman Bansal <bnaman50@gmail.com>
# 2020

############################################################################################################################################################################################################################
## Gradient
# Heatmaps
for noise in {0,1}
do
  CUDA_VISIBLE_DEVICES=0 python Gradient_Madry.py -idp ./Images/images_sensitivity/ -bs 1 -if_n ${noise} -op ./results/Sensitivity/Gradient/
done

# Evaluations
python Sensitivity_Analysis_Basic.py -idp ./results/Sensitivity/Gradient/ -mn grad --metric_name spearman -op ./results/evaluation_result_text_files/Gradient/ #spearman
python Sensitivity_Analysis_Basic.py -idp ./results/Sensitivity/Gradient/ -mn grad --metric_name ssim -op ./results/evaluation_result_text_files/Gradient/ #ssim
python Sensitivity_Analysis_Basic.py -idp ./results/Sensitivity/Gradient/ -mn grad --metric_name hog -op ./results/evaluation_result_text_files/Gradient/ #hog
python Sensitivity_Analysis_IOU.py -idp ./results/Sensitivity/Gradient/ -mn grad -op ./results/evaluation_result_text_files/Gradient #iou

CUDA_VISIBLE_DEVICES=0 python Sensitivity_Analysis_Model_Dependent.py -idp ./results/Sensitivity/Gradient -mn grad --metric_name insertion -op ./results/evaluation_result_text_files/Gradient #insertion

CUDA_VISIBLE_DEVICES=0 python Sensitivity_Analysis_Model_Dependent.py -idp ./results/Sensitivity/Gradient -mn grad --metric_name deletion -op ./results/evaluation_result_text_files/Gradient #deletion
############################################################################################################################################################################################################################

############################################################################################################################################################################################################################
## Input x Gradient
# Heatmaps
for noise in {0,1}
do
  CUDA_VISIBLE_DEVICES=0 python Input_times_Gradient_Madry.py -idp ./Images/images_sensitivity/ -bs 1 -if_n ${noise} -op ./results/Sensitivity/InpGrad/
done

# Evaluations
python Sensitivity_Analysis_Basic.py -idp ./results/Sensitivity/InpGrad/ -mn inpgrad --metric_name spearman -op ./results/evaluation_result_text_files/InpGrad/ #spearman
python Sensitivity_Analysis_Basic.py -idp ./results/Sensitivity/Gradient/ -mn inpgrad --metric_name ssim -op ./results/evaluation_result_text_files/InpGrad/ #ssim
python Sensitivity_Analysis_Basic.py -idp ./results/Sensitivity/Gradient/ -mn inpgrad --metric_name hog -op ./results/evaluation_result_text_files/InpGrad/ #hog
python Sensitivity_Analysis_IOU.py -idp ./results/Sensitivity/Gradient/ -mn inpgrad -op ./results/evaluation_result_text_files/InpGrad #iou

CUDA_VISIBLE_DEVICES=0 python Sensitivity_Analysis_Model_Dependent.py -idp ./results/Sensitivity/Gradient -mn inpgrad --metric_name insertion -op ./results/evaluation_result_text_files/InpGrad #insertion

CUDA_VISIBLE_DEVICES=0 python Sensitivity_Analysis_Model_Dependent.py -idp ./results/Sensitivity/Gradient -mn inpgrad --metric_name deletion -op ./results/evaluation_result_text_files/InpGrad #deletion
############################################################################################################################################################################################################################

############################################################################################################################################################################################################################
## Occlusion
# Heatmaps
for patch in {52,53,54}
do
  CUDA_VISIBLE_DEVICES=0 python Occlusion_Madry.py -idp ./Images/images_sensitivity/ -ops ${patch} -op ./results/Sensitivity/Occlusion/
done

# Evaluations
python Sensitivity_Analysis_Basic_Occlusion_Comp_With_Default_Settings.py -idp ./results/Sensitivity/Occlusion/ -mn occlusion --metric_name hog -op ./results/evaluation_result_text_files/Occlusion --exp_num a03
############################################################################################################################################################################################################################

############################################################################################################################################################################################################################
## MP
# Heatmaps
for blur in {5,10,30}
do
  CUDA_VISIBLE_DEVICES=0 python MP_MADRY.py --seed 0 --img_dir_path ./Images/images_sensitivity/ --out_path ./results/Sensitivity/MP/ --mask_init circular --mask_init_size 28 --blur_radius ${blur}
done

# Evaluations
python Sensitivity_Analysis_Basic_MP_Comp_With_Default_Settings.py  -idp ./results/Sensitivity/MP/ -mn mp --metric_name spearman -op ./results/evaluation_result_text_files/MP --exp_num a22
############################################################################################################################################################################################################################

############################################################################################################################################################################################################################
## LIME
# Heatmaps
for seed in {0,1,2,3,4}
do
  CUDA_VISIBLE_DEVICES=0 python LIME_Madry.py -idp ./Images/images_sensitivity/  -op ./results/Sensitivity/LIME -l_bp 0 -l_sn 50 -l_es ${seed} -ifn 0
done

# Evaluations
python Sensitivity_Analysis_Basic_LIME_Comp_With_Default_Settings.py -idp ./results/Sensitivity/LIME/ -mn lime --metric_name hog -op ./results/evaluation_result_text_files/LIME/ --exp_num a02
############################################################################################################################################################################################################################

############################################################################################################################################################################################################################
## Smoothgrad
# Heatmaps
for std in {0.1,0.2,0.3}
do
  CUDA_VISIBLE_DEVICES=0 python SmoothGrad_Madry.py -idp ./Images/images_sensitivity/  -if_n 0 -op ./results/Sensitivity/SmoothGrad -n_sam 50 -std ${std}
done

# Evaluations
python Sensitivity_Analysis_Basic_SmoothGrad_Comp_With_Default_Settings.py -idp ./results/Sensitivity/SmoothGrad/ -mn sg --metric_name hog -op ./results/evaluation_result_text_files/SmoothGrad/ --exp_num a31
############################################################################################################################################################################################################################

