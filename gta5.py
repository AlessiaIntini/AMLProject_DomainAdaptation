import json
import os
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image

from torchvision.datasets.utils import iterable_to_str, verify_str_arg
from torchvision.datasets.vision import VisionDataset
import numpy as np
import torchvision.transforms as transforms
from transform import *
from torchvision.transforms import v2

class GTA5(VisionDataset):

    GTA5Class = namedtuple(
        "GTA5Class",
        ["name", "id", "train_id", "category", "category_id", "has_instances", "ignore_in_eval", "color"],
    )

    classes = [
        GTA5Class("unlabeled",            0, 255, "void", 0, False, True, (0, 0, 0)),
        GTA5Class("ego vehicle",          1, 255, "void", 0, False, True, (0, 0, 0)),
        GTA5Class("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
        GTA5Class("out of roi",           3, 255, "void", 0, False, True, (0, 0, 0)),
        GTA5Class("static",               4, 255, "void", 0, False, True, (0, 0, 0)),
        GTA5Class("dynamic",              5, 255, "void", 0, False, True, (111, 74, 0)),
        GTA5Class("ground",               6, 255, "void", 0, False, True, (81, 0, 81)),
        GTA5Class("road",                 7, 0, "flat", 1, False, False, (128, 64, 128)),
        GTA5Class("sidewalk",             8, 1, "flat", 1, False, False, (244, 35, 232)),
        GTA5Class("parking",              9, 255, "flat", 1, False, True, (250, 170, 160)),
        GTA5Class("rail track",           10, 255, "flat", 1, False, True, (230, 150, 140)),
        GTA5Class("building",             11, 2, "construction", 2, False, False, (70, 70, 70)),
        GTA5Class("wall",                 12, 3, "construction", 2, False, False, (102, 102, 156)),
        GTA5Class("fence",                13, 4, "construction", 2, False, False, (190, 153, 153)),
        GTA5Class("guard rail",           14, 255, "construction", 2, False, True, (180, 165, 180)),
        GTA5Class("bridge",               15, 255, "construction", 2, False, True, (150, 100, 100)),
        GTA5Class("tunnel",               16, 255, "construction", 2, False, True, (150, 120, 90)),
        GTA5Class("pole",                 17, 5, "object", 3, False, False, (153, 153, 153)),
        GTA5Class("polegroup",            18, 255, "object", 3, False, True, (153, 153, 153)),
        GTA5Class("traffic light",        19, 6, "object", 3, False, False, (250, 170, 30)),
        GTA5Class("traffic sign",         20, 7, "object", 3, False, False, (220, 220, 0)),
        GTA5Class("vegetation",           21, 8, "nature", 4, False, False, (107, 142, 35)),
        GTA5Class("terrain",              22, 9, "nature", 4, False, False, (152, 251, 152)),
        GTA5Class("sky",                  23, 10, "sky", 5, False, False, (70, 130, 180)),
        GTA5Class("person",               24, 11, "human", 6, True, False, (220, 20, 60)),
        GTA5Class("rider",                25, 12, "human", 6, True, False, (255, 0, 0)),
        GTA5Class("car",                  26, 13, "vehicle", 7, True, False, (0, 0, 142)),
        GTA5Class("truck",                27, 14, "vehicle", 7, True, False, (0, 0, 70)),
        GTA5Class("bus",                  28, 15, "vehicle", 7, True, False, (0, 60, 100)),
        GTA5Class("caravan",              29, 255, "vehicle", 7, True, True, (0, 0, 90)),
        GTA5Class("trailer",              30, 255, "vehicle", 7, True, True, (0, 0, 110)),
        GTA5Class("train",                31, 16, "vehicle", 7, True, False, (0, 80, 100)),
        GTA5Class("motorcycle",           32, 17, "vehicle", 7, True, False, (0, 0, 230)),
        GTA5Class("bicycle",              33, 18, "vehicle", 7, True, False, (119, 11, 32)),
        GTA5Class("license plate",        -1, -1, "vehicle", 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    id_to_color = np.array([c.color for c in classes])
    
    def __init__(
        self,
        root: str = "dataset",
        split: str = "train",
        mode: str = "fine",
        cropsize = (512, 1024),
        target_type: Union[List[str], str] = "semantic",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform, target_transform)
        print("root: ", root)
        self.mode = "gtFine" if mode == "fine" else "gtCoarse"
        self.images_dir = os.path.join(self.root,"images")
        self.targets_dir = os.path.join(self.root, "labels")
        self.target_type = target_type
        self.split = split
        self.images = []
        self.targets = []

        self.to_tensor = transforms.Compose([
            v2.RandomResizedCrop(size=cropsize),
            transforms.ToTensor()
            ])


        self.trans_train = Compose([
            RandomCrop(cropsize)
            ])
        verify_str_arg(mode, "mode", ("fine", "coarse"))
        if mode == "fine":
            valid_modes = ("train", "test", "val")
        else:
            valid_modes = ("train", "train_extra", "val")
        msg = "Unknown value '{}' for argument split if mode is '{}'. Valid values are {{{}}}."
        msg = msg.format(split, mode, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        if not isinstance(target_type, list):
            self.target_type = [target_type]
        [
            verify_str_arg(value, "target_type", ("instance", "semantic", "polygon", "color" ))
            for value in self.target_type
        ]

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                target_types = []
                for t in self.target_type:
                    target_name = "{}_{}".format(
                        file_name.split("_leftImg8bit")[0], self._get_target_suffix(self.mode, t)
                    )
                    target_types.append(os.path.join(target_dir, target_name))

                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(target_types)
                

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        image = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.targets[index][0])
        #image , target = self.transforms(image,target)
        #image , target = ExtResize((512,1024))(image,target)
        #image , target = ExtToTensor()(image,target)
        #image, target = self.trans_train((image,target))

        to_tensor = transforms.Compose([
            v2.RandomResizedCrop(size=(512,1024)),
            transforms.ToTensor()
            ])

        image = to_tensor(image)
        #target = np.array(target).astype(np.int64)[np.newaxis,:]
        #target = self.to_tensor(target)
        
        return image, target
    
    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if target_type == "instance":
            return f"{mode}_instanceIds.png"
        elif target_type == "semantic":
            return f"{mode}_labelTrainIds.png"
        elif target_type == "color":
            return f"{mode}_color.png"
        else:
            return f"{mode}_polygons.json"
        
    
    def __len__(self) -> int:
        return len(self.images)