from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset as torchDataset
import os
from abc import ABCMeta
from dataclasses import dataclass
from typing import Tuple, Optional, Any
from utils import ExtResize, ExtToTensor, ExtTransforms , ExtCompose
from cityscapes import CityScapes
class BaseGTALabels(metaclass=ABCMeta):
    pass

@dataclass
class GTA5Label:
    TRAIN_ID: int
    color: Tuple[int, int, int]


class GTA5Labels_TaskCV2017(BaseGTALabels):
    road = GTA5Label(TRAIN_ID=0, color=(128, 64, 128))
    sidewalk = GTA5Label(TRAIN_ID=1, color=(244, 35, 232))
    building = GTA5Label(TRAIN_ID=2, color=(70, 70, 70))
    wall = GTA5Label(TRAIN_ID=3, color=(102, 102, 156))
    fence = GTA5Label(TRAIN_ID=4, color=(190, 153, 153))
    pole = GTA5Label(TRAIN_ID=5, color=(153, 153, 153))
    light = GTA5Label(TRAIN_ID=6, color=(250, 170, 30))
    sign = GTA5Label(TRAIN_ID=7, color=(220, 220, 0))
    vegetation = GTA5Label(TRAIN_ID=8, color=(107, 142, 35))
    terrain = GTA5Label(TRAIN_ID=9, color=(152, 251, 152))
    sky = GTA5Label(TRAIN_ID=10, color=(70, 130, 180))
    person = GTA5Label(TRAIN_ID=11, color=(220, 20, 60))
    rider = GTA5Label(TRAIN_ID=12, color=(255, 0, 0))
    car = GTA5Label(TRAIN_ID=13, color=(0, 0, 142))
    truck = GTA5Label(TRAIN_ID=14, color=(0, 0, 70))
    bus = GTA5Label(TRAIN_ID=15, color=(0, 60, 100))
    train = GTA5Label(TRAIN_ID=16, color=(0, 80, 100))
    motocycle = GTA5Label(TRAIN_ID=17, color=(0, 0, 230))
    bicycle = GTA5Label(TRAIN_ID=18, color=(119, 11, 32))

    list_ = [
        road,
        sidewalk,
        building,
        wall,
        fence,
        pole,
        light,
        sign,
        vegetation,
        terrain,
        sky,
        person,
        rider,
        car,
        truck,
        bus,
        train,
        motocycle,
        bicycle,
    ]

    @property
    def support_id_list(self):
        ret = [label.ID for label in self.list_]
        return ret



class GTA5(torchDataset):
    label_map = GTA5Labels_TaskCV2017()

    train_id = np.array([c.train_id for c in CityScapes.classes])
    #id_to_train_id = np.append(train_id, 255)
    #print(id_to_train_id)

    class PathPair_ImgAndLabel:
        IMG_DIR_NAME = "images"
        LBL_DIR_NAME = "labels"
        SUFFIX = ".png"

        def __init__(self, root, labels_source="train_ids"):
            self.root = root
            self.labels_source = labels_source
            self.img_paths = self.create_imgpath_list()
            self.lbl_paths = self.create_lblpath_list()

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx: int):
            img_path = self.img_paths[idx]
            lbl_path = self.lbl_paths[idx]
            return img_path, lbl_path

        def create_imgpath_list(self):
            img_dir = os.path.join(self.root, self.IMG_DIR_NAME)
            #img_dir = os.path.join(self.root , self.IMG_DIR_NAME)
            img_path = [os.path.join(img_dir, path) for path in os.listdir(img_dir) if path.endswith(self.SUFFIX)]
            return img_path

        def create_lblpath_list(self):
            lbl_dir = os.path.join(self.root,self.LBL_DIR_NAME)
            if self.labels_source == "cityscapes":
                lbl_path = [os.path.join(lbl_dir,path) for path in os.listdir(lbl_dir) if path.endswith(self.SUFFIX) and path.__contains__("_labelTrainIds")]
            elif self.labels_source == "GTA5":
                lbl_path = [os.path.join(lbl_dir,path) for path in os.listdir(lbl_dir) if (path.endswith(self.SUFFIX) and not path.__contains__("_labelTrainIds.png"))]
            return lbl_path

    def __init__(self, 
                 root: Path,
                 labels_source: str = "GTA5", # "cityscapes" or "GTA5"
                 transforms:Optional[ExtTransforms]=None,
                 split="train"):
        """

        :param root: (Path)
            this is the directory path for GTA5 data
        """
        self.root = os.path.join(root , 'GTA5')
        self.labels_source = labels_source
        self.transforms = transforms
        self.paths = self.PathPair_ImgAndLabel(root=self.root,labels_source=labels_source)
        

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx, isPath=False):
        img_path, lbl_path = self.paths[idx]
        if isPath:
            return img_path, lbl_path
        img = self.read_img(img_path)
        lbl = self.read_img(lbl_path)
        
        lbl = Image.fromarray(np.array(self.convert_from_id_to_trainId(lbl),dtype='uint8'))

        #if self.labels_source == "GTA5":
        #    lbl = Image.fromarray(np.array(self.convert_from_id_to_trainId(lbl),dtype='uint8')) 
        #    #if not os.path.exists(lbl_path.split('.png')[0] + "_labelTrainIds.png"):
        #    #    lbl.convert('L').save(lbl_path.split('.png')[0] + "_labelTrainIds.png")
        #
        if self.transforms is not None:
            img, lbl = self.transforms(img, lbl)
        #else:
        #    img = ExtToTensor()(img)
        #    lbl = ExtToTensor()(lbl)
        return img, lbl

    @staticmethod
    def read_img(path):
        img = Image.open(str(path))
        #img = np.array(img)
        return img

    @classmethod
    def decode(cls, lbl):
        return cls._decode(lbl, label_map=cls.label_map.list_)

    @staticmethod
    def _decode(lbl, label_map):
        # remap_lbl = lbl[np.where(np.isin(lbl, cls.label_map.support_id_list), lbl, 0)]
        color_lbl = np.zeros((*lbl.shape, 3))
        for label in label_map:
            color_lbl[lbl == label.TRAIN_ID] = label.color
        return color_lbl
    
    def convert_from_id_to_trainId(self, lbl):
        return self.train_id[np.array(lbl)]
    
    @classmethod 
    def visualize_prediction(cls,outputs,labels) -> Tuple[Any, Any]:
        preds = outputs.max(1)[1].detach().cpu().numpy()
        lab = labels.detach().cpu().numpy()
        colorized_preds = cls.decode(preds).astype('uint8') # To RGB images, (N, H, W, 3), ranged 0~255, numpy array
        colorized_labels = cls.decode(lab).astype('uint8')
        colorized_preds = Image.fromarray(colorized_preds[0]) # to PIL Image
        colorized_labels = Image.fromarray(colorized_labels[0])
        return colorized_preds , colorized_labels