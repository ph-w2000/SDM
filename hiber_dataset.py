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
        self.categories = ["MULTI"]
        self.split = split
        self.channel_first = False

        self.require_full_masks = False

        self.sequence_length = 4

        if split == "val_small":
            self.ds = hiber.HIBERDataset(root=self.data_dir, categories=self.categories, mode="val", channel_first=self.channel_first)
        else:
            self.ds = hiber.HIBERDataset(root=self.data_dir, categories=self.categories, mode=self.split, channel_first=self.channel_first)

        self.classes = {i: n for i, n in enumerate(HIBER_CLASSES, 1)}

    def generate_number_list(self, n, t):
        if n % 590 <= 590 - t:
            return list(range(n, n + t))
        else:
            result_list = list(range(n, (n // 590) * 590 + 590))
            result_list = result_list + [result_list[-1]] * max(0, t - len(result_list))
            return result_list

    def __getitem__(self, idx):
        img_id = idx

        ids = self.generate_number_list(img_id, self.sequence_length)

        if self.split == "val_small":
            if idx % 6 ==0:
                ids = self.generate_number_list(0 + idx//590 * 590, self.sequence_length)
            elif idx % 6 ==1:
                ids = self.generate_number_list(80 + idx//590 * 590, self.sequence_length)
            elif idx % 6 ==2:
                ids = self.generate_number_list(190 + idx//590 * 590, self.sequence_length)
            elif idx % 6 ==3:
                ids = self.generate_number_list(300 + idx//590 * 590, self.sequence_length)
            elif idx % 6 ==4:
                ids = self.generate_number_list(410 + idx//590 * 590, self.sequence_length)
            else:
                ids = self.generate_number_list(520 + idx//590 * 590, self.sequence_length)

        return self.get_image(ids), self.get_target(ids)
    
    def get_image(self, ids):
        hors = []
        vers = []

        for id in ids:
            data = self.ds[id]
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

            hors.append(torch.tensor(hor))
            vers.append(torch.tensor(ver))

        hors = torch.stack(hors,dim=1)
        vers = torch.stack(vers,dim=1)

        return hors, vers
        
    def get_target(self, ids):

        img_ids = []
        labels = []
        masks = []
        full_masks = []
        for id in ids:
            data = self.ds[id]
            if data[6].shape == (0,1248,1640):
                silhouette = np.full((1, 1248, 1640), False)
                data = data[:6] + (silhouette,) + data[7:]

            mask = torch.tensor(data[6])
            mask = torch.nn.functional.interpolate(mask.float().unsqueeze(0), size=(160, 200), mode='bilinear', align_corners=False).squeeze(0)
            mask = mask.round().long()

            if self.require_full_masks:
                full_mask = torch.tensor(data[6])

            if mask.shape[0] > 1:
                mask = torch.unsqueeze(torch.any(mask.bool(), dim=0), dim=0)
                if self.require_full_masks:
                    full_mask = torch.unsqueeze(torch.any(torch.tensor(data[6]).bool(), dim=0), dim=0)
            
            label = data[1]
            label = torch.tensor(label)

            img_id = torch.tensor(int(data[7]))

            masks.append(mask)
            labels.append(label)
            img_ids.append(img_id)

            if self.require_full_masks:
                full_masks.append(full_mask)

        img_ids = torch.stack(img_ids,dim=0)
        labels = torch.stack(labels,dim=0)
        masks = torch.stack(masks,dim=1)

        if self.require_full_masks:
            full_masks = torch.stack(full_masks,dim=1)

        target = dict(image_id=img_ids, labels=labels, masks=masks, full_masks=full_masks)

        return target
    
    def collate_fn(self, samples):
        return default_collate(samples)



if __name__ == "__main__":
    root_path = '../../dataset/single/'
    dataset = HIBERDataset(root_path, 'val', channel_first=False) 
    a = dataset.get_target(0)
    
