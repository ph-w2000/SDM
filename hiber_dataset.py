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
        if self.split == "val_small":
            return 66

        return len(self.ds)
    
    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        self.categories = ["WALK"]
        self.split = split
        self.channel_first = False

        if split == "val_small":
            self.ds = hiber.HIBERDataset(root=self.data_dir, categories=self.categories, mode="val", channel_first=self.channel_first)
        else:
            self.ds = hiber.HIBERDataset(root=self.data_dir, categories=self.categories, mode=self.split, channel_first=self.channel_first)

        self.classes = {i: n for i, n in enumerate(HIBER_CLASSES, 1)}

    def __getitem__(self, idx):
        if self.split == "val_small":
            if idx % 6 ==0:
                return self.get_image(0 + idx//590 * 590), self.get_target(0 + idx//590 * 590)
            elif idx % 6 ==1:
                return self.get_image(80 + idx//590 * 590), self.get_target(80 + idx//590 * 590)
            elif idx % 6 ==2:
                return self.get_image(190 + idx//590 * 590), self.get_target(190 + idx//590 * 590)
            elif idx % 6 ==3:
                return self.get_image(300 + idx//590 * 590), self.get_target(300 + idx//590 * 590)
            elif idx % 6 ==4:
                return self.get_image(410 + idx//590 * 590), self.get_target(410 + idx//590 * 590)
            else:
                return self.get_image(520 + idx//590 * 590), self.get_target(520 + idx//590 * 590)

        return self.get_image(idx), self.get_target(idx)
    
    def get_image(self, img_id):
        data = self.ds[img_id]
        if data[0].shape == (160,200,2):
            image = np.transpose(data[0], (2, 0, 1))
            hor = image
        else:
            hor = data[0]

        if data[1].shape == (160,200,2):
            image = np.transpose(data[1], (2, 0, 1))
            ver = image
        else:
            ver = data[0]

        return hor, ver
        
    def get_target(self, img_id):
        data = self.ds[img_id]
        if data[6].shape == (0,1248,1640):
            silhouette = np.full((1, 1248, 1640), False)
            data = data[:6] + (silhouette,) + data[7:]

        mask = torch.tensor(data[6])
        mask = torch.nn.functional.interpolate(mask.float().unsqueeze(0), size=(160, 200), mode='bilinear', align_corners=False).squeeze(0)
        mask = mask.round().long()

        fullMask = torch.tensor(data[6]).float().round().long()

        if mask.shape[0] > 1:
            mask = torch.unsqueeze(torch.any(mask.bool(), dim=0), dim=0)
            fullMask = torch.unsqueeze(torch.any(torch.tensor(data[6]).bool(), dim=0), dim=0)

        
        label = data[1]
        label = torch.tensor(label)

        img_id = torch.tensor(int(data[7]))

        target = dict(image_id=img_id, labels=label, masks=mask, fullMasks=fullMask)

        return target
    
    def collate_fn(self, samples):
        return default_collate(samples)



if __name__ == "__main__":
    root_path = '../../dataset/single/'
    dataset = HIBERDataset(root_path, 'val', channel_first=False) 
    a = dataset.get_target(0)
    
