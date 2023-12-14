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
from hiber_dataset import HIBERDataset
import matplotlib.pyplot as plt
from einops import rearrange
from torch.utils.data import DataLoader, Dataset, SequentialSampler

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
        if torch.all(a2 == 0):
            ious.append(torch.ones(1).squeeze().cuda())
            continue
        intersection = torch.logical_and(a1, a2)
        union = torch.logical_or(a1, a2)
        if torch.sum(union) != 0:
            iou = torch.sum(intersection) / torch.sum(union)
        else:
            iou =  torch.zeros(1).squeeze().cuda()
        ious.append(iou)
    ious = torch.stack(ious)
    return torch.sum(ious, dim=0)


class IntervalSequentialSampler(SequentialSampler):
    def __init__(self, data_source, interval=1):
        super().__init__(data_source)
        self.interval = interval

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        indices = indices[::self.interval]
        return iter(indices)

def test(conf, val_loader, ema, diffusion, betas, cond_scale, wandb):

    import time

    acc = 0
    num = 0

    for ind, (imgs, targets) in enumerate(tqdm(val_loader)):

        image_hor = imgs[0].float()
        image_ver = imgs[1].float()
        b, d = image_hor.shape[0], image_hor.shape[2]
        image = torch.cat((image_hor,image_ver), 1)
        mask_GT = targets['masks'].float()

        mask = torch.zeros(b*d, 1, 160, 200)

        val_img = image.cuda()
        val_pose = mask.cuda()
        mask_GT = mask_GT.cuda()

        # for b in mask_GT.cpu():
        #     for x in range(4):
        #         plt.imshow(b[:,x,:,:].squeeze())
        #         plt.axis('off')  # Turn off axis labels
        #         plt.show()

        mask_GT = rearrange(mask_GT, 'b c d h w -> (b d) c h w')
        val_img = rearrange(val_img, 'b c d h w -> (b d) c h w')
        
        with torch.no_grad():
            if args.sample_algorithm == 'ddpm':
                print ('Sampling algorithm used: DDPM')
                samples = diffusion.p_sample_loop(ema, x_cond = [val_img, val_pose], progress = True, cond_scale = cond_scale)
            elif args.sample_algorithm == 'ddim' and is_main_process():
                print ('Sampling algorithm used: DDIM')
                nsteps = 5

                noise = torch.randn([mask_GT.shape[0],64,160,200]).cuda()
                seq = range(0, conf.diffusion.beta_schedule["n_timestep"], conf.diffusion.beta_schedule["n_timestep"]//nsteps)
                xs, x0_preds = ddim_steps(noise, seq, ema, betas.cuda(), [val_img, val_pose], diffusion=diffusion, d=d)
                samples = xs[-1].cuda()

                scaled_samples = samples.float()
                scaled_GT = mask_GT.float()
                iou = calculate_iou( scaled_samples,scaled_GT).cpu()/(b*d)
                acc += iou
                num +=1
                print("IoU: ", iou)
                
                if is_main_process():

                    prediction = torch.cat([scaled_samples], -1)
                    MaGT = torch.cat([scaled_GT], -1)

                    wandb.log({'Prediction':wandb.Image(wandb.Image(prediction),caption=("IoU "+str(iou)) )})
                    wandb.log({'GT':wandb.Image(MaGT)})
    print("total IoU: " , acc/num)


def main(settings, EXP_NAME):

    [args, DiffConf, DataConf] = settings

    if is_main_process(): 
        # wandb.init(mode="disabled")
        wandb.init(project="person-synthesis", name = EXP_NAME,  settings = wandb.Settings(code_dir="."))

    if DiffConf.ckpt is not None: 
        DiffConf.training.scheduler.warmup = 0

    DiffConf.distributed = True
    local_rank = int(os.environ['LOCAL_RANK'])
    
    DataConf.data.train.batch_size = args.batch_size  #src -> tgt , tgt -> src
    

    val_dataset = HIBERDataset(args.dataset_path, "test")
    custom_sampler = IntervalSequentialSampler(val_dataset, interval=4)
    val_dataset = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=custom_sampler,
        drop_last=False,
        num_workers=getattr(8, 'num_workers', 0),
    )

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
    betas = DiffConf.diffusion.beta_schedule.make()
    diffusion = create_gaussian_diffusion(betas, predict_xstart = False)

    if DiffConf.ckpt is not None:
        ckpt = torch.load(DiffConf.ckpt, map_location=lambda storage, loc: storage)
        print("load", DiffConf.ckpt)
        if DiffConf.distributed:
            model.module.load_state_dict(ckpt["model"])

        else:
            model.load_state_dict(ckpt["model"])

        ema.load_state_dict(ckpt["ema"])
        scheduler.load_state_dict(ckpt["scheduler"])
        diffusion.embedding_table.load_state_dict(ckpt["prediction_head_embedding"])
        diffusion.conv_seg.load_state_dict(ckpt["prediction_head_conv"])

        if is_main_process():  print ('model loaded successfully')

    test(DiffConf, val_dataset, ema, diffusion, betas, args.cond_scale, wandb)

if __name__ == "__main__":

    init_distributed()

    import argparse

    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--exp_name', type=str, default='pidm_deepfashion')
    parser.add_argument('--DiffConfigPath', type=str, default='./config/diffusion.conf')
    parser.add_argument('--DataConfigPath', type=str, default='./config/data.yaml')
    parser.add_argument('--dataset_path', type=str, default='../../../../media/penghui02/T7/')
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

    DiffConf.ckpt = "checkpoints/pidm_deepfashion/last.pt"

    main(settings = [args, DiffConf, DataConf], EXP_NAME = args.exp_name)