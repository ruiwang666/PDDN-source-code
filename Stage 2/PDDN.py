# code is written by Rui Wang (email:rwangbp@connect.ust.hk)
import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from monai.config import print_config
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss
from tqdm import tqdm

from generative.inferers import LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

from models.autoenc import autoenc, encoder

from torch.utils.data import TensorDataset, DistributedSampler
from torch.utils.data import DataLoader
import numpy as np
from torchinfo import summary

import torch.multiprocessing as mp
import torch.distributed as dist

from autoencoder2d.autoencoder import AutoencoderNowcast

print_config()

root_dir = "/mnt/beegfs/rwangbp/"
# checkpoint and save image folder
feature = ""
model_save_path = root_dir + feature
os.makedirs(model_save_path, exist_ok=True)
val_fig_save_path = root_dir + feature
os.makedirs(val_fig_save_path, exist_ok=True)

batch_size = 4
channel = 0  # 0 = Flair
assert channel in [0, 1, 2, 3], "Choose a valid channel"

def load_data():
    # train radar data
    radar_data_train = np.load('')
    
    # train wrf data
    wrf_data_train = np.load('')
    
    radar_data_train = np.concatenate([radar_data_train, wrf_data_train], axis = 1)
    
    # val radar data
    radar_data_val = np.load('')
    
    # val wrf data
    wrf_data_val = np.load('')
    
    radar_data_val = np.concatenate([radar_data_val, wrf_data_val], axis = 1)
    
    # test radar data
    radar_data_test = np.load('')
    
    # test wrf data
    wrf_data_test = np.load('')
    
    radar_data_test = np.concatenate([radar_data_test, wrf_data_test], axis = 1)

    return radar_data_train, radar_data_val, radar_data_test

def cleanup():
    dist.destroy_process_group()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12333'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train_ddp(rank, world_size):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    print(f"Using {device}")

    radar_data_train, radar_data_val, radar_data_test = load_data()

    train_dataset = TensorDataset(torch.tensor(radar_data_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(radar_data_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(radar_data_test, dtype=torch.float32))

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=16, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=16, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=16, persistent_workers=True)
    


    autoencoder1 = AutoencoderNowcast(
        frame_num=1,
        # load checkpoint of 2d autoencoder
        save_ckpt='',
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        in_channels=1,
        block_out_channels=[128, 256, 512, 512, 512],
        act_fn='silu',
        latent_channels=16,
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        norm_num_groups=32,
        layers_per_block=2,
        out_channels=1, ).to(device)

    autoencoder1 = torch.nn.parallel.DistributedDataParallel(autoencoder1, device_ids=[rank], find_unused_parameters=True)
    
    autoencoder2 = AutoencoderNowcast(
        frame_num=1,
        save_ckpt=None,
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        in_channels=19,
        block_out_channels=[128, 256, 512, 512, 512],
        act_fn='silu',
        latent_channels=16,
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        norm_num_groups=32,
        layers_per_block=2,
        out_channels=1, ).to(device)

    autoencoder2 = torch.nn.parallel.DistributedDataParallel(autoencoder2, device_ids=[rank], find_unused_parameters=True)
    


    unet = DiffusionModelUNet(
        with_conditioning=True,
        cross_attention_dim=64,
        spatial_dims=3,
        in_channels=32,
        out_channels=16,
        num_res_blocks=2,
        num_channels=(128, 256, 256, 256),
        attention_levels=(True, True, True, True),
        num_head_channels=(8, 8, 8, 8),
    )
    unet.to(device)

    unet = torch.nn.parallel.DistributedDataParallel(unet, device_ids=[rank], find_unused_parameters=True)

    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0001, beta_end=0.02)

    with torch.no_grad():
        with autocast(enabled=True):
            check_data = first(train_loader)
            z = autoencoder1.module.encode(check_data[0][:1, :1, 1:2].to(device))

    print(f"Scaling factor set to {1/torch.std(z)}")
    scale_factor = 1

    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    optimizer_diff = torch.optim.AdamW(params=unet.parameters(), lr=1e-4, weight_decay=1e-5)

    n_epochs = 15000
    val_interval = 5
    epoch_loss_list = []

    autoencoder1.eval()
    scaler = GradScaler()



    def validate(epoch):

        autoencoder1.eval()
        unet.eval()
        val_batch = next(iter(val_loader))

        img_data = val_batch[0][:, :, :2].to(device)
        
        data_previous_gt = img_data[:, :1, :1]
        data_mask = img_data[:, 1:2, 1:2]
        data_future_gt = img_data[:, :1, 1:2]
        
        data_wrf = img_data[:, 2:, 1:2]
        
        
        with torch.no_grad():
            z_previous_gt = autoencoder1.module.encode(data_previous_gt) * scale_factor
            
        z_condition = torch.cat((data_previous_gt, data_mask, data_wrf), 1)

        noise = torch.randn_like(z_previous_gt)
        noise = noise.to(device)
        scheduler.set_timesteps(num_inference_steps=1000)
        with torch.no_grad():
            synthetic_images = inferer.sample(
                input_noise=noise, conditioning=z_condition, autoencoder_model_radar=autoencoder1.module, autoencoder_model_wrf=autoencoder2.module, diffusion_model=unet.module, scheduler=scheduler
            )

        idx = 0
        img = synthetic_images[idx, 0].detach().cpu().numpy()  # images
        img_gt = data_future_gt[idx, 0].detach().cpu().numpy()
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(75, 30))
        for frame in range(2):
            if (frame // 1) == 0:
                ax = axs[frame // 1]
                ax.imshow(img[frame], cmap="gray")
            else:
                ax = axs[frame // 1]
                ax.imshow(img_gt[frame-1], cmap="gray")

        fig.savefig(os.path.join(val_fig_save_path, f"val_comparison_epoch_{epoch}_rank_{rank}.png"))

    for epoch in range(n_epochs):
        unet.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch[0].to(device)
            optimizer_diff.zero_grad(set_to_none=True)
            

            
            data_previous_gt = images[:, :1, :1]
            data_mask = images[:, 1:2, 1:2]
            data_future_gt = images[:, :1, 1:2]
            
            data_wrf = images[:, 2:, 1:2]
            

            with torch.no_grad():
                z_previous_gt = autoencoder1.module.encode(data_previous_gt) * scale_factor
                
            z_condition = torch.cat((data_previous_gt, data_mask, data_wrf), 1)


            with autocast(enabled=True):
                
                    
                    
                noise = torch.randn_like(z_previous_gt).to(device)

                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                ).long()

                noise_pred = inferer(
                    inputs=data_future_gt, condition=z_condition, autoencoder_model_radar=autoencoder1.module, autoencoder_model_wrf=autoencoder2.module, diffusion_model=unet.module, noise=noise, timesteps=timesteps
                )

                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer_diff)
            scaler.update()

            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= world_size

            epoch_loss += loss.item()

            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
        epoch_loss_list.append(epoch_loss / (step + 1))

        if epoch % val_interval == 0:
            validate(epoch)
            if rank == 0:
                os.makedirs(model_save_path, exist_ok=True)
                torch.save(unet.state_dict(), os.path.join(model_save_path, f"unet_model_epoch_{epoch}.pt"))

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
