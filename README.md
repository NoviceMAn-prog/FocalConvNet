# Video Capsule Endoscopy Classification using Focal Modulation Guided Convolutional Neural Network
## 2.) Overview
### 2.1.)Introduction
In our paper, we have proposed FocalConvNet, a focal modulation network integrated with lightweight convolutional layers for the classification of small bowel anatomical landmarks and luminal findings. FocalConvNet leverages focal modulation to attain global context and allows global-local spatial interactions throughout the forward pass. Moreover, the convolutional block with its intrinsic inductive/learning bias and capacity to extract hierarchical features allows our FocalConvNet to achieve favourable results with high throughput.

## 2.2.) Model Architecture of Our FocalConvNet
![](FocalConvNet_v2.jpeg)

## 3.) Training and Testing
## 3.1)Data Preparation
Follow the data preparation procedure in the official dataset repository "[Kvasir-Capusle](https://github.com/simula/kvasir-capsule)"

## 3.2)Training and testing
1.) The architecture for the FocalConvNet is defined in focalconv.py 
2.) run the training script in the official dataset repo and replace the model definition in [Line 325](https://github.com/simula/kvasir-capsule/blob/master/experiments/resnet152_train.py) with the architecture of FocalConvNet.
## 4.) FAQ
Please feel free to contact me if you need any advice or guidance in using this work ([E-mail](abhisheksrivastava2397@gmail.com)) 

## Acknowlegment
For our codebase, we use the repo of [Focal Modulation networks](https://github.com/microsoft/FocalNet) and [Kvasir-Capusle](https://github.com/simula/kvasir-capsule). We thank the authors for the nicely organized code!
