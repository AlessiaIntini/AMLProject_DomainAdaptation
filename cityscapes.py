"""#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import v2

import os.path as osp
import os
from PIL import Image
import numpy as np
import json

from transform import *



class CityScapes(Dataset):
    def __init__(self, mode, rootpath, transforms):
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
            tmp_label_path = osp.join(label_path,folder)
            for image in os.listdir(tmp_path):
                self.data.append(osp.join(tmp_path,image))
                self.label.append(osp.join(tmp_label_path,image.replace('_leftImg8bit','_gtFine_color')))

        #for folder in os.listdir(label_path):
        #    tmp_path = osp.join(label_path,folder)
        #    for label in os.listdir(tmp_path):
        #        self.label.append(osp.join(tmp_path,label))
        print("Collected data: " + str(len(self.data))+" "+str(len(self.label)))
        if len(self.data) != len(self.label):
            
            print("Error collecting data " + str(len(self.data))+" "+str(len(self.label)))
            raise SystemExit('-1')
        
        self.transform = transforms

    def __getitem__(self, idx):

        image = Image.open(self.data[idx]).convert('RGB')
        label = Image.open(self.label[idx]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)
        #print("\n"+str(len(label)))
        #label = np.array(label).astype(np.int64)[np.newaxis, :]
        
        return  image,label

    def __len__(self):
        return len(self.data)"""

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
    def __init__(self, rootpth, cropsize=(640, 480), mode='train', 
    randomscale=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5), *args, **kwargs):
        super(CityScapes, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        print('self.mode', self.mode)
        self.ignore_lb = 255

        with open('./cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}
        
        #print(self.lb_map)

        ## parse img directory
        self.imgs = {}
        imgnames = []
        impth = osp.join(rootpth, 'images', mode)
        folders = os.listdir(impth)
        for fd in folders:
            fdpth = osp.join(impth, fd)
            im_names = os.listdir(fdpth)
            names = [el.replace('_leftImg8bit.png', '') for el in im_names]
            impths = [osp.join(fdpth, el) for el in im_names]
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, impths)))

        ## parse gt directory
        self.labels = {}
        gtnames = []
        gtpth = osp.join(rootpth, 'gtFine', mode)
        folders = os.listdir(gtpth)
        for fd in folders:
            fdpth = osp.join(gtpth, fd)
            lbnames = os.listdir(fdpth)
            lbnames = [el for el in lbnames if 'labelTrainIds' in el]
            names = [el.replace('_gtFine_labelTrainIds.png', '') for el in lbnames]
            lbpths = [osp.join(fdpth, el) for el in lbnames]
            gtnames.extend(names)
            self.labels.update(dict(zip(names, lbpths)))

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
        if idx < 3:
            print(lbpth)
        img = Image.open(impth).convert('RGB')
        label = Image.open(lbpth)
        if self.mode == 'train' or self.mode == 'trainval':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]

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



