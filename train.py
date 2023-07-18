import os
import warnings

warnings.filterwarnings("ignore")

import time, cv2, torch, wandb
import torch.distributed as dist
from config.diffconfig import DiffusionConfig, get_model_conf
from config.dataconfig import Config as DataConfig
from tensorfn import load_config as DiffConfig
from diffusion import create_gaussian_diffusion, make_beta_schedule, ddim_steps
from tensorfn.optim import lr_scheduler
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import data as deepfashion_data
from model import UNet

def init_distributed():

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    setup_for_distributed(rank == 0)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def is_main_process():
    try:
        if dist.get_rank()==0:
            return True
        else:
            return False
    except:
        return True

def sample_data(loader):
    loader_iter = iter(loader)
    epoch = 0

    while True:
        try:
            yield epoch, next(loader_iter)

        except StopIteration:
            epoch += 1
            loader_iter = iter(loader)

            yield epoch, next(loader_iter)


def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def calculate_iou(array1, array2):

    size = array1.shape[0]
    ious = []

    for s in range(size):
        a1 = array1[s]
        a2 = array2[s]
        intersection = np.logical_and(a1, a2)
        union = np.logical_or(a1, a2)
        if np.sum(union) != 0:
            iou = np.sum(intersection) / np.sum(union)
        else:
            iou = 0
        ious.append(iou)

    return np.mean(iou)


def train(conf, loader, val_loader, model, ema, diffusion, betas, optimizer, scheduler, guidance_prob, cond_scale, device, wandb):

    import time

    i = 0

    loss_list = []
    loss_mean_list = []
    loss_vb_list = []

 
    for epoch in range(500):

        if is_main_process: print ('#Epoch - '+str(epoch))

        start_time = time.time()

        # for batch in tqdm(loader):
        for ind, (imgs, targets) in enumerate(tqdm(loader)):
            i = i + 1

            image_hor = imgs[0].float()
            image_ver = imgs[1].float()
            image = torch.cat((image_hor,image_ver), 1)
            mask_GT = targets['masks'].float()
            bone_2d = targets['bone_2d'].long()

            mask = torch.zeros(image_hor.shape[0], 1, 160, 200)
            B, N, K, _ = bone_2d.shape
            b_indices = torch.arange(B).view(B, 1, 1, 1)
            n_indices = torch.arange(N).view(1, N, 1, 1)
            x_positions = bone_2d[:, :, :, 0:1].long()
            y_positions = bone_2d[:, :, :, 1:].long()
            mask[b_indices, n_indices, y_positions, x_positions] = 1

            img = torch.cat([image, image], 0)
            target_img = torch.cat([mask_GT , mask_GT], 0)
            target_pose = torch.cat([mask, mask], 0)

            img = img.to(device)
            target_img = target_img.to(device)
            target_pose = target_pose.to(device)
            time_t = torch.randint(
                0,
                conf.diffusion.beta_schedule["n_timestep"],
                (img.shape[0],),
                device=device,
            )

            loss_dict = diffusion.training_losses(model, x_start = target_img, t = time_t, cond_input = [img, target_pose], prob = 1 - guidance_prob)
            
            loss = loss_dict['loss'].mean()
            loss_mse = loss_dict['mse'].mean()
            loss_vb = loss_dict['vb'].mean()
        

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            scheduler.step()
            optimizer.step()
            loss = loss_dict['loss'].mean()

            loss_list.append(loss.detach().item())
            loss_mean_list.append(loss_mse.detach().item())
            loss_vb_list.append(loss_vb.detach().item())

            accumulate(
                ema, model.module, 0 if i < conf.training.scheduler.warmup else 0.9999
            )


            if i%args.save_wandb_logs_every_iters == 0 and is_main_process():

                wandb.log({'loss':(sum(loss_list)/len(loss_list)), 
                            'loss_vb':(sum(loss_vb_list)/len(loss_vb_list)), 
                            'loss_mean':(sum(loss_mean_list)/len(loss_mean_list)), 
                            'epoch':epoch,
                            'steps':i})
                loss_list = []
                loss_mean_list = []
                loss_vb_list = []


            if i%args.save_checkpoints_every_iters == 0 and is_main_process():

                if conf.distributed:
                    model_module = model.module

                else:
                    model_module = model

                torch.save(
                    {
                        "model": model_module.state_dict(),
                        "ema": ema.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "conf": conf,
                    },
                    conf.training.ckpt_path + f"/model_{str(i).zfill(6)}.pt"
                )

        if is_main_process():

            print ('Epoch Time '+str(int(time.time()-start_time))+' secs')
            print ('Model Saved Successfully for #epoch '+str(epoch)+' #steps '+str(i))

            if conf.distributed:
                model_module = model.module

            else:
                model_module = model

            torch.save(
                {
                    "model": model_module.state_dict(),
                    "ema": ema.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "conf": conf,
                },
                conf.training.ckpt_path + '/last.pt'
               
            )

        if (epoch)%args.save_wandb_images_every_epochs==0:

            print ('Generating samples at epoch number ' + str(epoch))

            acc = 0

            val_batch = next(val_loader)
            image_hor = val_batch[0][0].float()
            image_ver = val_batch[0][1].float()
            image = torch.cat((image_hor,image_ver), 1)
            mask_GT = val_batch[1]['masks'].float()
            bone_2d = val_batch[1]['bone_2d'].long()

            mask = torch.zeros(image_hor.shape[0], 1, 160, 200)
            y_positions = bone_2d[:,:,:,1]
            x_positions = bone_2d[:,:,:,0]
            mask[:,: , y_positions, x_positions] = 1
            mask = mask.float()

            val_img = image.cuda()
            val_pose = mask.cuda()
            mask_GT = mask_GT.cuda()

            with torch.no_grad():

                if args.sample_algorithm == 'ddpm':
                    print ('Sampling algorithm used: DDPM')
                    samples = diffusion.p_sample_loop(ema, x_cond = [val_img, val_pose], progress = True, cond_scale = cond_scale)
                elif args.sample_algorithm == 'ddim':
                    print ('Sampling algorithm used: DDIM')
                    nsteps = 50
                    noise = torch.randn(mask_GT.shape).cuda()
                    seq = range(0, conf.diffusion.beta_schedule["n_timestep"], conf.diffusion.beta_schedule["n_timestep"]//nsteps)
                    xs, x0_preds = ddim_steps(noise, seq, ema, betas.cuda(), [val_img, val_pose])
                    samples = xs[-1].cuda()

                    acc = calculate_iou( samples.data.cpu().numpy(),mask_GT.data.cpu().numpy())
                    print("IoU: ", acc)
            
            if is_main_process():

                heatmaps_hor = torch.cat([val_img[:,0:1,:,:]], -1)
                heatmaps_samples_hor = [torch.zeros_like(heatmaps_hor) for _ in range(dist.get_world_size())]
                dist.all_gather(heatmaps_samples_hor, heatmaps_hor)

                heatmaps_ver = torch.cat([val_img[:,2:3,:,:]], -1)
                heatmaps_samples_ver = [torch.zeros_like(heatmaps_ver) for _ in range(dist.get_world_size())]
                dist.all_gather(heatmaps_samples_ver, heatmaps_ver)

                prediction = torch.cat([samples], -1)
                prediction_samples = [torch.zeros_like(prediction) for _ in range(dist.get_world_size())]
                dist.all_gather(prediction_samples, prediction)

                MaGT = torch.cat([mask_GT], -1)
                MaGT_samples = [torch.zeros_like(MaGT) for _ in range(dist.get_world_size())]
                dist.all_gather(MaGT_samples, MaGT)

                # import matplotlib.pyplot as plt

                # k = torch.cat(prediction_samples, -2).data.cpu().numpy()
                # k = np.transpose(k[0], (1, 2, 0))
                # # Plot the image
                # plt.imshow(k)
                # plt.axis('off')  # Optional: turn off the axis
                # plt.show()
                # exit()

                wandb.log({'Hor Map':wandb.Image(torch.cat(heatmaps_samples_hor, -2))})
                wandb.log({'Ver Map':wandb.Image(torch.cat(heatmaps_samples_ver, -2))})
                wandb.log({'Prediction':wandb.Image(torch.cat(prediction_samples, -2))})
                wandb.log({'GT':wandb.Image(torch.cat(MaGT_samples, -2))})



def main(settings, EXP_NAME):

    [args, DiffConf, DataConf] = settings

    if is_main_process(): 
        # wandb.init(mode="disabled")
        wandb.init(project="person-synthesis", name = EXP_NAME,  settings = wandb.Settings(code_dir="."))

    if DiffConf.ckpt is not None: 
        DiffConf.training.scheduler.warmup = 0

    DiffConf.distributed = True
    local_rank = int(os.environ['LOCAL_RANK'])
    
    DataConf.data.train.batch_size = args.batch_size//2  #src -> tgt , tgt -> src
    

    val_dataset, train_dataset = deepfashion_data.get_train_val_dataloader(DataConf.data, labels_required = True, distributed = True)
    
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    val_dataset = iter(cycle(val_dataset))

    model = get_model_conf().make_model()
    model = model.to(args.device)
    ema = get_model_conf().make_model()
    ema = ema.to(args.device)

    if DiffConf.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True
        )

    optimizer = DiffConf.training.optimizer.make(model.parameters())
    scheduler = DiffConf.training.scheduler.make(optimizer)

    if DiffConf.ckpt is not None:
        ckpt = torch.load(DiffConf.ckpt, map_location=lambda storage, loc: storage)

        if DiffConf.distributed:
            model.module.load_state_dict(ckpt["model"])

        else:
            model.load_state_dict(ckpt["model"])

        ema.load_state_dict(ckpt["ema"])
        scheduler.load_state_dict(ckpt["scheduler"])

        if is_main_process():  print ('model loaded successfully')

    betas = DiffConf.diffusion.beta_schedule.make()
    diffusion = create_gaussian_diffusion(betas, predict_xstart = False)

    train(
        DiffConf, train_dataset, val_dataset, model, ema, diffusion, betas, optimizer, scheduler, args.guidance_prob, args.cond_scale, args.device, wandb
    )

if __name__ == "__main__":

    init_distributed()

    import argparse

    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--exp_name', type=str, default='pidm_deepfashion')
    parser.add_argument('--DiffConfigPath', type=str, default='./config/diffusion.conf')
    parser.add_argument('--DataConfigPath', type=str, default='./config/data.yaml')
    parser.add_argument('--dataset_path', type=str, default='../../dataset/HIBER/')
    parser.add_argument('--save_path', type=str, default='checkpoints')
    parser.add_argument('--cond_scale', type=int, default=2)
    parser.add_argument('--guidance_prob', type=int, default=0.1)
    parser.add_argument('--sample_algorithm', type=str, default='ddim') # ddpm, ddim
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_wandb_logs_every_iters', type=int, default=50)
    parser.add_argument('--save_checkpoints_every_iters', type=int, default=2000)
    parser.add_argument('--save_wandb_images_every_epochs', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--n_machine', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    print ('Experiment: '+ args.exp_name)

    DiffConf = DiffConfig(DiffusionConfig,  args.DiffConfigPath, args.opts, False)
    DataConf = DataConfig(args.DataConfigPath)

    DiffConf.training.ckpt_path = os.path.join(args.save_path, args.exp_name)
    DataConf.data.path = args.dataset_path

    if is_main_process():

        if not os.path.isdir(args.save_path): os.mkdir(args.save_path)
        if not os.path.isdir(DiffConf.training.ckpt_path): os.mkdir(DiffConf.training.ckpt_path)

    #DiffConf.ckpt = "checkpoints/last.pt"

    main(settings = [args, DiffConf, DataConf], EXP_NAME = args.exp_name)