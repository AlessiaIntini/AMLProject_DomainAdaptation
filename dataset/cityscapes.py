import json
import os
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image
from utils.augUtils import *
from torchvision.datasets.utils import iterable_to_str, verify_str_arg
from torchvision.datasets.vision import VisionDataset
import numpy as np

class CityScapes(VisionDataset):

    """Args:
        root (string): Root directory of dataset 

        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="fine"
            otherwise ``train``, ``train_extra`` or ``val``

        mode (string, optional): The quality mode to use, ``fine`` or ``coarse``

        target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``. Can also be a list to output a tuple with all specified target types.

        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version."""

    CityscapesClass = namedtuple(
        "CityscapesClass",
        ["name", "id", "train_id", "category", "category_id", "has_instances", "ignore_in_eval", "color"],
    )

    classes = [
        CityscapesClass("unlabeled",            0, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("ego vehicle",          1, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("out of roi",           3, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("static",               4, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("dynamic",              5, 255, "void", 0, False, True, (111, 74, 0)),
        CityscapesClass("ground",               6, 255, "void", 0, False, True, (81, 0, 81)),
        CityscapesClass("road",                 7, 0, "flat", 1, False, False, (128, 64, 128)),
        CityscapesClass("sidewalk",             8, 1, "flat", 1, False, False, (244, 35, 232)),
        CityscapesClass("parking",              9, 255, "flat", 1, False, True, (250, 170, 160)),
        CityscapesClass("rail track",           10, 255, "flat", 1, False, True, (230, 150, 140)),
        CityscapesClass("building",             11, 2, "construction", 2, False, False, (70, 70, 70)),
        CityscapesClass("wall",                 12, 3, "construction", 2, False, False, (102, 102, 156)),
        CityscapesClass("fence",                13, 4, "construction", 2, False, False, (190, 153, 153)),
        CityscapesClass("guard rail",           14, 255, "construction", 2, False, True, (180, 165, 180)),
        CityscapesClass("bridge",               15, 255, "construction", 2, False, True, (150, 100, 100)),
        CityscapesClass("tunnel",               16, 255, "construction", 2, False, True, (150, 120, 90)),
        CityscapesClass("pole",                 17, 5, "object", 3, False, False, (153, 153, 153)),
        CityscapesClass("polegroup",            18, 255, "object", 3, False, True, (153, 153, 153)),
        CityscapesClass("traffic light",        19, 6, "object", 3, False, False, (250, 170, 30)),
        CityscapesClass("traffic sign",         20, 7, "object", 3, False, False, (220, 220, 0)),
        CityscapesClass("vegetation",           21, 8, "nature", 4, False, False, (107, 142, 35)),
        CityscapesClass("terrain",              22, 9, "nature", 4, False, False, (152, 251, 152)),
        CityscapesClass("sky",                  23, 10, "sky", 5, False, False, (70, 130, 180)),
        CityscapesClass("person",               24, 11, "human", 6, True, False, (220, 20, 60)),
        CityscapesClass("rider",                25, 12, "human", 6, True, False, (255, 0, 0)),
        CityscapesClass("car",                  26, 13, "vehicle", 7, True, False, (0, 0, 142)),
        CityscapesClass("truck",                27, 14, "vehicle", 7, True, False, (0, 0, 70)),
        CityscapesClass("bus",                  28, 15, "vehicle", 7, True, False, (0, 60, 100)),
        CityscapesClass("caravan",              29, 255, "vehicle", 7, True, True, (0, 0, 90)),
        CityscapesClass("trailer",              30, 255, "vehicle", 7, True, True, (0, 0, 110)),
        CityscapesClass("train",                31, 16, "vehicle", 7, True, False, (0, 80, 100)),
        CityscapesClass("motorcycle",           32, 17, "vehicle", 7, True, False, (0, 0, 230)),
        CityscapesClass("bicycle",              33, 18, "vehicle", 7, True, False, (119, 11, 32)),
        CityscapesClass("license plate",        -1, 255, "vehicle", 7, False, True, (0, 0, 142)),
    ]

    # Lists used for decoding and mapping
    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    id_to_color = np.array([c.color for c in classes])


    def __init__(
        self,
        root: str = "/content/Cityscapes/Cityspaces",
        split: str = "train",
        mode: str = "fine",
        target_type: Union[List[str], str] = "semantic",
        transforms: Optional[ExtTransforms] = None,
    ) -> None:
        super().__init__(root, transforms)
        print("root: ", root)
        self.mode = "gtFine" if mode == "fine" else "gtCoarse"
        self.images_dir = os.path.join(self.root,"images",split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.target_type = target_type
        self.split = split
        self.images = []
        self.targets = []

        # Do Some checks
        ##########################################################################################
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
        ##########################################################################################

        # Get the image, appending the path in a list
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
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item.
        """

        # Open the images and apply the transformations
        image = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.targets[index][0])

        image , target = self.transforms(image,target)
        
        return image, target

    def __len__(self) -> int:
        return len(self.images)

    
    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if target_type == "instance":
            return f"{mode}_instanceIds.png"
        elif target_type == "semantic":
            return f"{mode}_labelTrainIds.png"
        elif target_type == "color":
            return f"{mode}_color.png"
        else:
            return f"{mode}_polygons.json"

    @classmethod
    def decode(cls, target):
        target[target == 255] = 19
        return cls.train_id_to_color[target]

    # this function is used to visualize the prediction images
    @classmethod 
    def visualize_prediction(cls,outputs,labels) -> Tuple[Any, Any]:
        """
        Args:
                cls (CityScapes): The class object
                outputs (Tensor): The output of the model
                labels (Tensor): The ground truth labels
        Returns:
                Tuple[Any, Any]: The colorized predictions and the colorized labels
        """
        preds = outputs.max(1)[1].detach().cpu().numpy()
        lab = labels.detach().cpu().numpy()
        colorized_preds = cls.decode(preds).astype('uint8') # To RGB images, (N, H, W, 3), ranged 0~255, numpy array
        colorized_labels = cls.decode(lab).astype('uint8')
        colorized_preds = Image.fromarray(colorized_preds[0]) # to PIL Image
        colorized_labels = Image.fromarray(colorized_labels[0])
        return colorized_preds , colorized_labels
   