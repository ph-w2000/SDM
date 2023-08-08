# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
from tensorfn import load_config as DiffConfig
import numpy as np
from config.diffconfig import DiffusionConfig, get_model_conf
import torch.distributed as dist
import glob
from models.unet_autoenc import BeatGANsAutoencConfig
from diffusion import create_gaussian_diffusion, make_beta_schedule, ddim_steps
import torchvision.transforms as transforms
import torchvision
from hiber_dataset import HIBERDataset
from torch.utils.data import DataLoader
import resultVisualization

def calculate_iou(array1, array2):

    size = array1.shape[0]
    ious = []

    for s in range(size):
        a1 = array1[s]
        a2 = array2[s]
        intersection = torch.logical_and(a1, a2)
        union = torch.logical_or(a1, a2)
        if torch.sum(union) != 0:
            iou = torch.sum(intersection) / torch.sum(union)
        else:
            iou = 0
        ious.append(iou)

    return torch.mean(iou)

class Predictor():
    def __init__(self):
        """Load the model into memory to make running multiple predictions efficient"""

        conf = DiffConfig(DiffusionConfig, './config/diffusion.conf', show=False)
        self.betas = conf.diffusion.beta_schedule.make()
        
        self.model = get_model_conf().make_model()
        ckpt = torch.load("checkpoints/pidm_deepfashion/last_epoch_41.pt")
        self.model.load_state_dict(ckpt["ema"])
        self.model = self.model.cuda()
        self.model.eval()

        self.diffusion = create_gaussian_diffusion(self.betas, predict_xstart = False)
        
    def predict_pose(
        self,
        loader,
        num_poses=1,
        sample_algorithm='ddim',
        nsteps=100,

        ):
        """Run a single prediction on the model"""
        total_iou = 0
        i = 0

        for ind, (imgs, targets) in enumerate(tqdm(loader)):
            image_hor = imgs[0].float()
            image_ver = imgs[1].float()
            image = torch.cat((image_hor,image_ver), 1)
            mask_GT = targets['masks'].float()

            mask = torch.zeros(image_hor.shape[0], 1, 160, 200)

            val_img = image.cuda()
            val_pose = mask.cuda()
            mask_GT = mask_GT

            filenames = targets["image_id"]

            src = val_img
            tgt_pose = val_pose

            if sample_algorithm == 'ddpm':
                samples = self.diffusion.p_sample_loop(self.model, x_cond = [src, tgt_pose], progress = True, cond_scale = 2)
            elif sample_algorithm == 'ddim':
                noise = torch.randn([mask_GT.shape[0],64,160,200]).cuda()
                seq = range(0, 1000, 1000//nsteps)
                xs, x0_preds = ddim_steps(noise, seq, self.model, self.betas.cuda(), [src, tgt_pose], diffusion=self.diffusion)
                samples = xs[-1].cuda()


            hor = image_hor[0].data.cpu().numpy()
            ver = image_ver[0].data.cpu().numpy()

            scaled_samples = samples.float()
            scaled_GT = mask_GT.float()
            iou = calculate_iou( scaled_samples.data.cpu(),scaled_GT)
            print("IoU ", iou)
            total_iou+=iou
            i+=1

            resultVisualization.visualization(hor,ver, scaled_samples[0].data.cpu().numpy(), scaled_GT[0].numpy(), filenames[0], iou)
            # exit()
        print("Total IoU ", total_iou/i)


if __name__ == "__main__":

    obj = Predictor()
    
    loader = HIBERDataset("../../dataset/HIBER/","val")
    loader = DataLoader(loader, batch_size=8, shuffle=False, num_workers=8)
    
    obj.predict_pose(loader=loader, num_poses=2, sample_algorithm = 'ddim',  nsteps = 50)
    