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

class Predictor():
    def __init__(self):
        """Load the model into memory to make running multiple predictions efficient"""

        conf = DiffConfig(DiffusionConfig, './config/diffusion.conf', show=False)

        self.model = get_model_conf().make_model()
        ckpt = torch.load("checkpoints/pidm_deepfashion/last.pt")
        self.model.load_state_dict(ckpt["ema"])
        self.model = self.model.cuda()
        self.model.eval()

        self.betas = conf.diffusion.beta_schedule.make()
        self.diffusion = create_gaussian_diffusion(self.betas, predict_xstart = False)#.to(device)
        
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
        for ind, (imgs, targets) in enumerate(tqdm(loader)):
            image_hor = imgs[0].float()
            image_ver = imgs[1].float()
            image = torch.cat((image_hor,image_ver), 1)
            labels = targets['masks'].float().repeat(1, 4, 1, 1)
            bone_2d = targets['bone_2d'].long()

            mask = torch.zeros(image_hor.shape[0], 1, 160, 200)
            y_positions = bone_2d[:,:,:,1]
            x_positions = bone_2d[:,:,:,0]
            mask[:,: , y_positions, x_positions] = 1
            mask = mask.float()

            img = torch.cat([image, labels], 0)
            target_img = torch.cat([labels , image], 0)
            target_pose = torch.cat([mask, mask], 0)

            img = img.cuda()
            target_img = target_img.cuda()
            target_pose = target_pose.cuda()

            break

        src = img
        tgt_pose = target_pose

        if sample_algorithm == 'ddpm':
            samples = self.diffusion.p_sample_loop(self.model, x_cond = [src, tgt_pose], progress = True, cond_scale = 2)
        elif sample_algorithm == 'ddim':
            noise = torch.randn(src.shape).cuda()
            seq = range(0, 100, 100//nsteps)
            xs, x0_preds = ddim_steps(noise, seq, self.model, self.betas.cuda(), [src, tgt_pose])
            samples = xs[-1].cuda()


        # samples_grid = torch.cat([src[0],torch.cat([samps for samps in samples], -1)], -1)
        # samples_grid = (torch.clamp(samples_grid, -1., 1.) + 1.0)/2.0
        # pose_grid = torch.cat([torch.zeros_like(src[0,:1,:,:]),torch.cat([samps[:3] for samps in tgt_pose], -1)], -1)

        # output = torch.cat([1-pose_grid, samples_grid], -2)

        # numpy_imgs = output.unsqueeze(0).permute(0,2,3,1).detach().cpu().numpy()
        # fake_imgs = (255*numpy_imgs).astype(np.uint8)
        # Image.fromarray(fake_imgs[0]).save('output.png')

        for k in range(samples.shape[0]-1):
            hor = image_hor[k]
            ver = image_ver[k]

            samples = torch.softmax(samples[k][:1,:,:],dim=0)
            samples = np.where(samples.data.cpu().numpy() > 0.5, 1, 0)

            mask_prediction = samples
            mask_gt = labels[k][:1,:,:]
            
            resultVisualization.visualization(hor,ver, mask_prediction, mask_gt)
            exit()


    def predict_appearance(
        self,
        image,
        ref_img,
        ref_mask,
        ref_pose,
        sample_algorithm='ddim',
        nsteps=100,

        ):
        """Run a single prediction on the model"""

        src = Image.open(image)
        src = self.transforms(src).unsqueeze(0).cuda()
        
        ref = Image.open(ref_img)
        ref = self.transforms(ref).unsqueeze(0).cuda()

        mask = transforms.ToTensor()(Image.open(ref_mask)).unsqueeze(0).cuda()
        pose =  transforms.ToTensor()(np.load(ref_pose)).unsqueeze(0).cuda()


        if sample_algorithm == 'ddpm':
            samples = self.diffusion.p_sample_loop(self.model, x_cond = [src, pose, ref, mask], progress = True, cond_scale = 2)
        elif sample_algorithm == 'ddim':
            noise = torch.randn(src.shape).cuda()
            seq = range(0, 1000, 1000//nsteps)
            xs, x0_preds = ddim_steps(noise, seq, self.model, self.betas.cuda(), [src, pose, ref, mask], diffusion=self.diffusion)
            samples = xs[-1].cuda()


        samples = torch.clamp(samples, -1., 1.)

        output = (torch.cat([src, ref, mask*2-1, samples], -1) + 1.0)/2.0

        numpy_imgs = output.permute(0,2,3,1).detach().cpu().numpy()
        fake_imgs = (255*numpy_imgs).astype(np.uint8)
        Image.fromarray(fake_imgs[0]).save('output.png')

if __name__ == "__main__":

    obj = Predictor()
    
    loader = HIBERDataset("../../dataset/HIBER/","val")
    loader = DataLoader(loader, batch_size=1, shuffle=False, num_workers=1)
    
    obj.predict_pose(loader=loader, num_poses=2, sample_algorithm = 'ddim',  nsteps = 10)
    
    # ref_img = "data/deepfashion_256x256/target_edits/reference_img_0.png"
    # ref_mask = "data/deepfashion_256x256/target_mask/lower/reference_mask_0.png"
    # ref_pose = "data/deepfashion_256x256/target_pose/reference_pose_0.npy"

    # #obj.predict_appearance(image='test.jpg', ref_img = ref_img, ref_mask = ref_mask, ref_pose = ref_pose, sample_algorithm = 'ddim',  nsteps = 50)
