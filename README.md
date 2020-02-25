## SAM: The Sensitivity of Attribution Methods to Hyperparameters

This repository contains source code necessary to reproduce some of the main results in [the paper]():

**If you use this software, please consider citing:**
    
    @misc{bansal2020sam,
    title={SAM: The Sensitivity of Attribution Methods to Hyperparameters},
    author={Naman Bansal and Chirag Agarwal and Anh Nguyen},
    year={2020},
    eprint={},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
    
## 1. Setup

### Installing software
This repository is built using PyTorch. You can install the necessary libraries by pip installing the requirements text file `pip install -r ./requirements.txt`

## 2. Usage
The shell script for generating Figure 1 of our paper is in [teaser.sh](teaser.sh). Given an [image](./Images/ILSVRC2012_val_00002056.JPEG), the script runs SmoothGrad, Sliding-Patch, LIME, and Meaningful Perturbation algorithm for their different hyperparameters and produces a montage image of their respective [attribution maps](./results/formal_teaser.jpg)

### Examples
Generating the attribution map for the class "matchstick".
* Running `source teaser.sh` produces this result:

<p align="center">
    <img src="./results/formal_teaser.jpg" width=750px>
</p>
<p align="center"><i> The real image followed by the different attribution maps generated using (top-->bottom) LIME, Sliding-Patch, Meaningful Perturbation and SmoothGrad algorithms. We show the sensitivity of each explanation algorithm with respect to its respective hyperparameter.</i></p>

## 3. Licenses
Note that the code in this repository is licensed under MIT License, but, the pre-trained condition models used by the code have their own licenses. Please carefully check them before use. 

## 4. Questions?
If you have questions/suggestions, please feel free to [email](mailto:chiragagarwall12@gmail.com) or create github issues.
