from abc import ABCMeta
from dataclasses import dataclass
from typing import Tuple
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset as torchDataset
from utils import ExtTransforms
from typing import Optional
class BaseGTALabels(metaclass=ABCMeta):
    pass

@dataclass
class GTA5Label:
    ID: int
    color: Tuple[int, int, int]


class GTA5Labels_TaskCV2017(BaseGTALabels):
    road = GTA5Label(ID=0, color=(128, 64, 128))
    sidewalk = GTA5Label(ID=1, color=(244, 35, 232))
    building = GTA5Label(ID=2, color=(70, 70, 70))
    wall = GTA5Label(ID=3, color=(102, 102, 156))
    fence = GTA5Label(ID=4, color=(190, 153, 153))
    pole = GTA5Label(ID=5, color=(153, 153, 153))
    light = GTA5Label(ID=6, color=(250, 170, 30))
    sign = GTA5Label(ID=7, color=(220, 220, 0))
    vegetation = GTA5Label(ID=8, color=(107, 142, 35))
    terrain = GTA5Label(ID=9, color=(152, 251, 152))
    sky = GTA5Label(ID=10, color=(70, 130, 180))
    person = GTA5Label(ID=11, color=(220, 20, 60))
    rider = GTA5Label(ID=12, color=(255, 0, 0))
    car = GTA5Label(ID=13, color=(0, 0, 142))
    truck = GTA5Label(ID=14, color=(0, 0, 70))
    bus = GTA5Label(ID=15, color=(0, 60, 100))
    train = GTA5Label(ID=16, color=(0, 80, 100))
    motocycle = GTA5Label(ID=17, color=(0, 0, 230))
    bicycle = GTA5Label(ID=18, color=(119, 11, 32))

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

    class PathPair_ImgAndLabel:
        IMG_DIR_NAME = "images"
        LBL_DIR_NAME = "labels"
        SUFFIX = ".png"

        def __init__(self, root):
            self.root = root
            self.img_paths = self.create_imgpath_list()
            self.lbl_paths = self.create_lblpath_list()

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx: int):
            img_path = self.img_paths[idx]
            lbl_path = self.lbl_paths[idx]
            return img_path, lbl_path

        def create_imgpath_list(self):
            img_dir = self.root / self.IMG_DIR_NAME
            img_path = [path for path in img_dir.glob(f"*{self.SUFFIX}")]
            return img_path

        def create_lblpath_list(self):
            lbl_dir = self.root / self.LBL_DIR_NAME
            lbl_path = [path for path in lbl_dir.glob(f"*{self.SUFFIX}")]
            return lbl_path

    def __init__(self, root: Path, transforms: Optional[ExtTransforms] = None ):
        """

        :param root: (Path)
            this is the directory path for GTA5 data
            must be the following
            e.g.)
                ./data
                ├── images
                │   ├── 00001.png
                │   ├── ...
                │   └── 24966.png
                ├── images.txt
                ├── labels
                │   ├── 00001.png
                │   ├── ...
                │   └── 24966.png
                ├── test.txt
                └── train.txt
        """
        self.root = root
        self.transforms = transforms
        self.paths = self.PathPair_ImgAndLabel(root=self.root)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx, isPath=False):
        img_path, lbl_path = self.paths[idx]
        if isPath:
            return img_path, lbl_path

        img = self.read_img(img_path)
        lbl = self.read_img(lbl_path)
        img, lbl = self.transforms(img, lbl)
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
            color_lbl[lbl == label.ID] = label.color
        return color_lbl