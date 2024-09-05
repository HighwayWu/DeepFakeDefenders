# DeepFakeDefenders

<p align='center'>  
  <img src='https://github.com/HighwayWu/DeepFakeDefenders/blob/main/imgs/competition.png' width='850'/>
</p>

1st place solution for [The Global Multimedia Deepfake Detection (Image Track)](https://www.kaggle.com/competitions/multi-ffdi/overview) by "JTGroup" team.

## Table of Contents

- [Motivation](#motivation)
- [Framework](#framework)
- [Dependency](#dependency)
- [Inference](#inference)
- [Training](#training)
- [Acknowledgement](#acknowledgement)


## Motivation
The visualization of features in the official split of training and validation sets reveals two key insights: 1) the similarity in distribution between the training and validation sets makes it challenging to evaluate the model's performance on unseen test sets; 2) the feature types of fake data are richer compared to real data. This insight inspired us to initially use unsupervised clustering to partition the challenging validation set, followed by augmenting the types of real and fake data based on adversarial learning principles. These insights led to the development of our proposed generalizable deepfake detection method.

<p align='center'>  
  <img src='https://github.com/HighwayWu/DeepFakeDefenders/blob/main/imgs/motivation.jpg' width='850'/>
</p>


## Framework
The framework of our proposed method consists of two primary stages: Data Preparing and Training. In the Data Preparing stage, we focus on augmenting the existing dataset by generating new data through image editing and Stable Diffusion (SD) techniques. We then perform clustering to reassemble the dataset, aiming to enhance both the detection performance and the robustness of our method. In the Training stage, we introduce a series of expert models and optimize them using three types of losses: $L_{\mathsf{KL}}$, $L_{\mathsf{NCE}}$, and $L_{\mathsf{CE}}$. This multi-loss approach ensures that the model can effectively differentiate between authentic and manipulated images. Please refer to the tech report (will be on arXiv later) for more details.

<p align='center'>  
  <img src='https://github.com/HighwayWu/DeepFakeDefenders/blob/main/imgs/framework.jpg' width='850'/>
</p>

## Dependency
- torch 1.9.0
- timm 1.0.8
- scikit-learn 1.2.1
- torch-ema 0.3
- torch_kmeans 0.2.0
- albumentations 1.3.1

## Inference

```bash
>> python infer.py
```
Enter the image path and expect the following result:
```bash
>> demo/img_1_should_be_0.0016829653177410.jpg
>> Prediction of [demo/img_1_should_be_0.0016829653177410.jpg] being Deepfake: 0.001683078
```

**Note: The pretrained weights can be downloaded from [Baidu Pan](https://pan.baidu.com/s/1hh6Rub60T7UXok5rqACffQ?pwd=gxu5).**

## Training
The training code (e.g., unsupervised clustering and joint optimization loss) and detailed technical report will be available shortly.

## Acknowledgement
- THIS WORK WAS PERFORMED IN PART AT SICC WHICH IS SUPPORTED BY SKL-IOTSC, UNIVERSITY OF MACAU.

<p align='left'>  
  <img src='https://github.com/HighwayWu/DeepFakeDefenders/blob/main/imgs/organization.png' width='450'/>
</p>

- INCLUSION·CONFERENCE ON THE BUND, THE ORGANIZER OF THE COMPETITION.

## License
This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
