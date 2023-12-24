#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json

from transform import *



class CityScapes(Dataset):
    def __init__(self, mode, rootpath):
        super(CityScapes, self).__init__()
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        print('self.mode', self.mode)
        #self.ignore_lb = 255

        self.data = []
        self.label = []

        image_path = osp.join(rootpath,"images",mode)
        label_path = osp.join(rootpath,"gtFine",mode)

        for folder in os.listdir(image_path):
            tmp_path = osp.join(image_path,folder)
            for image in os.listdir(tmp_path):
                self.data.add(Image.open(osp.join(tmp_path,image)).convert('RGB'))

        for folder in os.listdir(label_path):
            tmp_path = osp.join(label_path,folder)
            for label in os.listdir(tmp_path):
                self.label.add(Image.open(osp.join(tmp_path,label)))

        if len(self.data) != len(self.label):
            print("Error collecting data")
            exit()


    def __getitem__(self, idx):
        return  self.data[idx],self.label[idx]

    def __len__(self):
        return len(self.data)