#/bin/bash
#
# Chirag Agarwal <chiragagarwall12.gmail.com>
# Naman Bansal <bnaman50@gmail.com>
# 2020

# Adversarial training script for GoogLeNet-R
source settings.py
data_path=$imagenet_train_path
output_dir='./imagenet_googlenet/'

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m robustness.main --dataset imagenet --data ${data_path} --adv-train 1 --arch googlenet --out-dir ${output_dir} --epochs 90 --lr 0.1 --batch-size 512 --step-lr 30 --eps 3 --attack-lr 0.5 --attack-steps 7 --constraint 2
