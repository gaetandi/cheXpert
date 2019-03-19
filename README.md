# CheXpert : A Large Chest X-Ray Dataset and Competition
A repository created for the MAP583 Deep Learning project.

This competition launched by the Stanford ML group aims at finding a prediction model which could perform as well as radiologist to find different pathologies thanks to chest X-Ray. The Dataset available to train our model is composed of 223,414 chest radiographs of 65,240 patients.

The **dataset** (the smaller dataset size is 11 GB) can be requested on the website of the competition: https://stanfordmlgroup.github.io/competitions/chexpert/

Publication : Irvin, Jeremy, et al. "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison." arXiv preprint arXiv:1901.07031 (2019).

This GitHub repository is composed of:
1- All the code in a jupyter notebook
2- A few pretrained and saved models
3- Different plots showing main results

# 1. The code

We implemented this project using Python 3 in the notebook *cheXpert_final*.

To run this organized notebook, you need the following packages: pytorch, PIL, cv2.

# 2. Models

Running the code, you may ignore the training process if you use one of our pretrained models:
-  model_ones_3epoch.tar is a DenseNet121 trained for 3 epochs using the policy "ones" (uncertain labels are considered positive)
-  model_zeroes_1epoch.pth.tar is a DenseNet121 trainet for 1 epoch using the policy "zeroes" (uncertain labels are considered negative)

# 3. Results

![alt text](https://github.com/gaetandi/cheXpert/blob/master/view1_frontal.jpg)
