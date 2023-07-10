import numpy as np
import torch
from torch.utils.data import Dataset
import os
from torch.utils.data.dataloader import default_collate
import sys
from collections import defaultdict
import json
torch.set_default_tensor_type(torch.FloatTensor)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import AHIBERTools as hiber
import random

HIBER_CLASSES = (
    "bg", "human"
)


class HIBERDataset(Dataset):

    def __len__(self):
        # return len(self.ds)
        return 8
    
    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        self.categories = ["WALK"]
        self.split = split
        self.channel_first = False

        self.ds = hiber.HIBERDataset(root=self.data_dir, categories=self.categories, mode=self.split, channel_first=self.channel_first)

        self.classes = {i: n for i, n in enumerate(HIBER_CLASSES, 1)}

    def __getitem__(self, idx):
        return self.get_image(idx), self.get_target(idx)
    
    def get_image(self, img_id):
        data = self.ds[img_id]
        image = np.transpose(data[0], (2, 0, 1))
        hor = image

        image = np.transpose(data[1], (2, 0, 1))
        ver = image

        return hor, ver
        
    def get_target(self, img_id):
        data = self.ds[img_id]
        if data[6].shape == (0,1248,1640):
            silhouette = np.full((1, 1248, 1640), False)
            data = data[:6] + (silhouette,) + data[7:]

        masks = torch.tensor(data[6])
        masks = torch.nn.functional.interpolate(masks.float().unsqueeze(0), size=(160, 200), mode='bilinear', align_corners=False).squeeze(0)
        masks = masks.round().long()

        fullMasks = torch.tensor(data[6]).float().round().long()
        
        hor_boxes = data[4]
        ver_boxes = data[5]
        hor_boxes = torch.tensor(hor_boxes, dtype=torch.float32)
        ver_boxes = torch.tensor(ver_boxes, dtype=torch.float32)
        
        labels = data[1]
        labels = torch.tensor(labels)

        bone_2d = data[2].copy()
        bone_2d[:,:,0] = bone_2d[:,:,0]/1640*200
        bone_2d[:,:,1] = bone_2d[:,:,1]/1248*160
        bone_2d = torch.tensor(bone_2d)
        bone_2d = torch.clamp(bone_2d, min=0)
        bone_2d[:,:,0] = torch.clamp(bone_2d[:,:,0], max=199)
        bone_2d[:,:,1] = torch.clamp(bone_2d[:,:,1], max=159)

        bone_2d_m = data[2]
        bone_2d_m = torch.tensor(bone_2d_m)
        img_id = torch.tensor(int(data[7]))

        target = dict(image_id=img_id, vboxes=ver_boxes, hboxes=hor_boxes, labels=labels, masks=masks, fullMasks=fullMasks, bone_2d=bone_2d, bone_2d_m=bone_2d_m)

        return target
    
    def collate_fn(self, samples):
        return default_collate(samples)



if __name__ == "__main__":
    root_path = '../../dataset/single/'
    dataset = HIBERDataset(root_path, 'val', channel_first=False) 
    a = dataset.get_target(0)
    
