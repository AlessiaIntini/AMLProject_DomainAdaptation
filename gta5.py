#!/usr/bin/python
# -*- encoding: utf-8 -*-


from ctypes import util
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json

import utils

from transform import *



class GTA5(Dataset):
    def __init__(self, rootpth, cropsize=(512, 1024), mode='train', 
    randomscale=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5), *args, **kwargs):
        super(GTA5, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        print('self.mode', self.mode)
        
        self.label_info = utils.get_label_info("./label.csv")

        ## parse img directory
        self.imgs = {}
        imgnames = []
        impth = osp.join(rootpth, 'images')
        images = os.listdir(impth)
        for im_name in images:
            name = im_name.replace('.png', '')
            print(name)
            impths = osp.join(impth, im_name)
            print(impths)
            imgnames.append(name)
            dictioary = dict([(name, impths)])
            print(dictioary)
            self.imgs.update(dictioary)

        ## parse label directory
        self.labels = {}
        gtnames = []
        gtpth = osp.join(rootpth, 'labels')
        labels = os.listdir(gtpth)
        for lb_name in labels:
            name = lb_name.replace('.png', '')
            print(name)
            lbpths = osp.join(gtpth, lb_name)
            print(lbpths)
            gtnames.append(name)
            dictioary = dict([(name, lbpths)])
            print(dictioary)
            self.labels.update([(name, lbpths)])

        print(self.labels)
        self.imnames = imgnames
        self.len = len(self.imnames)
        print('self.len', self.mode, self.len)
        assert set(imgnames) == set(gtnames)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())

        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_train = Compose([
            ColorJitter(
                brightness = 0.5,
                contrast = 0.5,
                saturation = 0.5),
            HorizontalFlip(),
            # RandomScale((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomScale(randomscale),
            # RandomScale((0.125, 1)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)),
            RandomCrop(cropsize)
            ])


    def __getitem__(self, idx):
        fn  = self.imnames[idx]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        
        img = Image.open(impth).convert('RGB')
        label = Image.open(lbpth).convert('RGB')
        
        if self.mode == 'train' or self.mode == 'trainval':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)

        label = np.array(label).astype(np.int64)[np.newaxis, :]
        label = utils.one_hot_it_v11(label,self.label_info)
        

        
    
        #print("label size: ")
        #print(len(label))
        #label = self.convert_labels(label)
        #print(label)
        return img, label


    def __len__(self):
        return self.len


    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label



