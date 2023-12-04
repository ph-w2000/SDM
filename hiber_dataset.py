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
        img_id = idx
        if 590 - img_id % 590 == 3:
            ids = [img_id, img_id+1, img_id+2, img_id+2]
        elif 590 - img_id % 590 == 2:
            ids = [img_id, img_id+1, img_id+1, img_id+1]
        elif 590 - img_id % 590 == 1:
            ids = [img_id, img_id, img_id, img_id]
        else:
            ids = [img_id, img_id+1, img_id+2, img_id+3]

        if self.split == "val_small":
            if idx % 6 ==0:
                ids = [0 + idx//6 * 590, 0 + idx//6 * 590 +1, 0 + idx//6 * 590 +2, 0 + idx//6 * 590 +3]
            elif idx % 6 ==1:
                ids = [80 + idx//6 * 590, 80 + idx//6 * 590 +1, 80 + idx//6 * 590 +2, 80 + idx//6 * 590 +3]
            elif idx % 6 ==2:
                ids = [190 + idx//6 * 590, 190 + idx//6 * 590 +1, 190 + idx//6 * 590 +2, 190 + idx//6 * 590 +3]
            elif idx % 6 ==3:
                ids = [300 + idx//6 * 590, 300 + idx//6 * 590 +1, 300 + idx//6 * 590 +2, 300 + idx//6 * 590 +3]
            elif idx % 6 ==4:
                ids = [410 + idx//6 * 590, 410 + idx//6 * 590 +1, 410 + idx//6 * 590 +2, 410 + idx//6 * 590 +3]
            else:
                ids = [520 + idx//6 * 590, 520 + idx//6 * 590 +1, 520 + idx//6 * 590 +2, 520 + idx//6 * 590 +3]

        return self.get_image(ids), self.get_target(ids)
    
    def get_image(self, ids):
        hors = []
        vers = []

        for id in ids:
            data = self.ds[id]
            image = np.transpose(data[0], (2, 0, 1))
            hor = torch.tensor(image)

            image = np.transpose(data[1], (2, 0, 1))
            ver = torch.tensor(image)

            hors.append(hor)
            vers.append(ver)

        hors = torch.stack(hors,dim=1)
        vers = torch.stack(vers,dim=1)

        return hors, vers
        
    def get_target(self, ids):

        img_ids = []
        labels = []
        masks = []
        for id in ids:
            data = self.ds[id]
            if data[6].shape == (0,1248,1640):
                silhouette = np.full((1, 1248, 1640), False)
                data = data[:6] + (silhouette,) + data[7:]

            mask = torch.tensor(data[6])
            mask = torch.nn.functional.interpolate(mask.float().unsqueeze(0), size=(160, 200), mode='bilinear', align_corners=False).squeeze(0)
            mask = mask.round().long()
            
            label = data[1]
            label = torch.tensor(label)

            img_id = torch.tensor(int(data[7]))

            masks.append(mask)
            labels.append(label)
            img_ids.append(img_id)

        img_ids = torch.stack(img_ids,dim=0)
        labels = torch.stack(labels,dim=0)
        masks = torch.stack(masks,dim=1)

        target = dict(image_id=img_ids, labels=labels, masks=masks)

        return target
    
    def collate_fn(self, samples):
        return default_collate(samples)



if __name__ == "__main__":
    root_path = '../../dataset/single/'
    dataset = HIBERDataset(root_path, 'val', channel_first=False) 
    a = dataset.get_target(0)
    
