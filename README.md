# CheXpert : A Large Chest X-Ray Dataset and Competition
*A repository created for the MAP583 Deep Learning project*

Authors: Gaëtan Dissez & Guillaume Duboc

This repository uses different sources, listed in the references, and especially ref n°4 for the coding part.

This competition launched by the Stanford ML group aims at finding a prediction model which could perform as well as radiologist to find different pathologies thanks to chest X-Ray. The Dataset available to train our model is composed of 223,414 chest radiographs of 65,240 patients.

The **dataset** (the smaller dataset size is 11 GB) can be requested on the [website](https://stanfordmlgroup.github.io/competitions/chexpert/) of the competition.

This GitHub repository is composed of:
1- All the code in a jupyter notebook
2- A few pretrained and saved models
3- Different plots showing main results


# 1. Code

We implemented this project using Python 3 in the notebook *cheXpert_final.ipynb*.

To run this organized notebook, you need the following packages: pytorch, PIL, cv2.

# 2. Models

Running the code, you may ignore the training process if you use one of our pretrained models:
-  *model_ones_2epoch_densenet.tar* is a DenseNet121 trained for 2 epochs using the policy "ones" (uncertain labels are considered positive)
-  *model_ones_3epoch_densenet.tar* is a DenseNet121 trained for 3 epochs using the policy "ones" (uncertain labels are considered positive)
-  *model_zeroes_1epoch_densenet.pth.tar* is a DenseNet121 trainet for 1 epoch using the policy "zeroes" (uncertain labels are considered negative)

# 3. Results

We first trained a DenseNet121 model using the policy "ones" (uncertain labels are considered positive).
For this model, we had the following loss during the training of 3 epochs:

<p align="center">
  <img src="https://github.com/gaetandi/cheXpert/blob/master/results/loss_ones_densenet.png" width="480">
</p>

And the following ROC curves (after the first epoch, the second epoch and the third epoch):

<p align="center">
  <img src="https://github.com/gaetandi/cheXpert/blob/master/results/ROC_densenet.png" width="1800">
</p>

You may check our presentation to see further details about our results.
These results do not outperform the results given by the Stanford ML group or evn radiologist, but the are encouraging as you can see on the following plot:

<p align="center">
  <img src="https://github.com/gaetandi/cheXpert/blob/master/results/Edema_radiologistscompare.png" width="400">
</p>

These charts are available in the *results* directory.

# 4. References

Publications : 
1. CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison, *Irvin, Jeremy, et al.*, 2019  [[Arxiv:1901.07031]](https://arxiv.org/pdf/1901.07031.pdf)
2. CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning, *Rajpurkar, Irvin, et al.*, 2017 [[Arxiv:1711.05225]](https://arxiv.org/pdf/1711.05225.pdf)
3. Densely Connected Convolutional Networks, *Huang et al.*, 2018 [[Arxiv:1608.06993]](https://arxiv.org/pdf/1608.06993.pdf)
4. [GitHub Repository](https://github.com/zoogzog/chexnet?fbclid=IwAR11GtcTJDglJpNYbqNIZFPeE4Zk9Ac132-fIVwqIkMItk3GGKY8OvhvVQA)
