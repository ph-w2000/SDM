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
import os, glob, cv2, time, shutil
from models.unet_autoenc import BeatGANsAutoencConfig
from diffusion import create_gaussian_diffusion, make_beta_schedule, ddim_steps
import torchvision.transforms as transforms
import torchvision
from hiber_dataset import HIBERDataset
from torch.utils.data import DataLoader
import resultVisualization

def replace_zeros_and_ones_with_random_values(tensor):
    # Get the shape of the input tensor
    shape = tensor.size()

    # Generate random values for 0s from the range [-1, 0]
    random_zeros = -torch.rand(shape)  # Generates random values in the range [-1, 0)

    # Generate random values for 1s from the range (0, 1]
    random_ones = torch.rand(shape)  # Generates random values in the range [0, 1)
    epsilon=1e-7
    random_ones = random_ones.masked_fill(random_ones == 0, epsilon)

    # Use torch.where to replace 0s with random_zeros and 1s with random_ones
    result = torch.where(tensor == 0, random_zeros, random_ones)

    return result

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

        self.model = get_model_conf().make_model()
        ckpt = torch.load("checkpoints/pidm_deepfashion-3090server/last.pt")
        self.model.load_state_dict(ckpt["ema"])
        self.model = self.model.cuda()
        self.model.eval()

        self.betas = conf.diffusion.beta_schedule.make()
        self.diffusion = create_gaussian_diffusion(self.betas, predict_xstart = False)#.to(device)
        self.diffusion.embedding_table.load_state_dict(ckpt["prediction_head_embedding"])
        self.diffusion.conv_seg.load_state_dict(ckpt["prediction_head_conv"])
        
        self.pose_list = glob.glob('data/deepfashion_256x256/target_pose/*.npy')
        self.transforms = transforms.Compose([transforms.Resize((256,256), interpolation=Image.BICUBIC),
                            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])
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
            mask_GT = replace_zeros_and_ones_with_random_values(mask_GT).float()
            bone_2d = targets['bone_2d'].long()
            filenames = targets["image_id"]

            mask = torch.zeros(image_hor.shape[0], 1, 160, 200)
            y_positions = bone_2d[:,:,:,1]
            x_positions = bone_2d[:,:,:,0]
            mask[:,: , y_positions, x_positions] = 1
            mask = mask.float()

            val_img = image.cuda()
            val_pose = mask.cuda()
            mask_GT = mask_GT


            src = val_img
            tgt_pose = val_pose

            if sample_algorithm == 'ddpm':
                samples = self.diffusion.p_sample_loop(self.model, x_cond = [src, tgt_pose], progress = True, cond_scale = 2)
            elif sample_algorithm == 'ddim':
                noise = torch.randn(mask_GT.shape).cuda()
                seq = range(0, 1000, 1000//nsteps)
                xs, x0_preds = ddim_steps(noise, seq, self.model, self.betas.cuda(), [src, tgt_pose], diffusion=self.diffusion)
                samples = xs[-1].cuda()


            hor = image_hor[0].data.cpu().numpy()
            ver = image_ver[0].data.cpu().numpy()

            scaled_samples = samples.float()
            scaled_GT = mask_GT.float()
            iou = calculate_iou( scaled_samples.data.cpu(),scaled_GT)

            total_iou+=iou
            i+=1

            resultVisualization.visualization(hor,ver, scaled_samples[0].data.cpu().numpy(), scaled_GT[0].numpy(), filenames[0], iou)
            # exit()
        print("IoU ", total_iou/i)


if __name__ == "__main__":

    obj = Predictor()
    
    loader = HIBERDataset("../../dataset/HIBER/","val")
    loader = DataLoader(loader, batch_size=8, shuffle=False, num_workers=4)
    
    obj.predict_pose(loader=loader, num_poses=2, sample_algorithm = 'ddim',  nsteps = 50)
    
    # a = torch.randint(0, 1, (2,1,160,200), dtype=torch.int64)
    # b = nn.Embedding(2, 512)
    # gt_down = b(a)
    # print(gt_down.shape)
    # gt_down = gt_down.squeeze(1).permute(0, 3, 1, 2)
    # gt_down = (torch.sigmoid(gt_down) * 2 - 1) * 1

    # print(gt_down.shape)
    
