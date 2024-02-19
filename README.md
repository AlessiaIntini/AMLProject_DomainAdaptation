# Real-time Domain Adaptation in Semantic Segmentation
This repository contains the code of our project on real time domani adaptatio in semantic segmentation, for the course Advanced Machine Learning at Polytechnic Of Turin University  

**Authors**  
* Alessia Intini s309895  
* Antonio Iorio s317748  
* Giuseppe Scarso s308807  

## Project structure
* `dataset/`: classes to manage the two dataset
    * Cityscapes
    * GTA5
* `model/`: models used in the project:
     * discriminator: implementation of discriminator for Adversarial       Domain Adaptation
     * model_stages
     * stdcnet
* `options/`: contains all argument that it is possible pass to command line
    * option
* `training/`: contains different files of source code to perform train in different context
    * train: contains code to train Cityscapes, GTA5, Cross-Domain and Cross-Domain with Data Augmentation
    * traindADA: contains code to train Adversarial Domain Adaptation
    * trainFDA: contains code to train Fourier Domain Adaptation
* `utils/`: functions used in the various stages of the project 
    * augUtils: contains simple transformations for generic operations and specific transformations for Data Augmentation techniques
    * FDAUtils: contains operation to perform Fourier transformations
    * utils
* `validation/`: 
    * validate
* `Document_project3_s309895_s317748_s308807.pdf`: this document contains the main results of the experiments
* `main.py`: in this file there is code to execute all possible experiments that have been implemented
* `Paper_project3_s309895_s317748_s308807.pdf`: this document contain the paper on Real-time Domain Adaptation in Semantic Segmentation
* `SemanticSegmentation.ipynb`: it is Jupiter Notebook to execute the code on colab

## Command to execute Real-time Domain Adaptation in Semantic Segmentation
### 1. Training on Cityscapes
```bash
main.py --dataset Cityscapes --batch_size 4 --optimizer sgd --resume true --validation_step 10 --num_workers 6
```

### 2. Training on GTA5
Training with GTA5, where 75% of GTA5 dataset is used to train and 25% is used to validation
```bash
main.py --dataset GTA5 --batch_size 8 --optimizer sgd --resume true --validation_step 10 --num_workers 6
```

### 3. Training on Cross-Domain
* Training with GTA5 obtained from the previuos step, 75% of GTA5 dataset is used to train and 25% is used to validation
```bash
main.py --dataset Cityscapes --batch_size 6 --optimizer sgd --resume true --validation_step 10 --num_workers 6 --mode test 
```

* Training with all dataset of GTA5 
```bash
main.py --dataset CROSS_DOMAIN --batch_size 6 --optimizer sgd --resume true --validation_step 10 --num_workers 6 
```

### 4. Training on Cross-Domain with Data Augmentation
```bash
main.py --dataset GTA5 --batch_size 6 --optimizer sgd --resume true --validation_step 10 --num_workers 6 --augmentation true 
```
### 5. Training on Unsupervised Adversarial Domain Adaptation
```bash
main.py --dataset DA --batch_size 6 --optimizer sgd --resume true --validation_step 10 --num_workers 6 
```
During experiments the best lr_discr and lambda_d1 are setted as default value, but it is possibile to change these values through command line.

### 6. Training on Fourier Domain Adaptation
```bash
main.py --dataset FDA --batch_size 4 --optimizer sgd --resume true --validation_step 10 --num_workers 6 
```
Also in this case the best l, entW and ita are setted as default value, but it is possibile to change these values through command line.



