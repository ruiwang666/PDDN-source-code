import torch
from torch import nn
from torch.nn import functional as F

from typing import Optional, Tuple, Union, Dict


from prediff.taming import AutoencoderKL


class AutoencoderNowcast(nn.Module):
    def __init__(
        self,
        frame_num: int = 4,
        save_ckpt: str = None,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        in_channels: int = 3,
        out_channels: int = 3,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        layers_per_block: int = 1,
        ):
        super().__init__()
        self.frame_num = frame_num
        self.autoencoder_obs = AutoencoderKL(
            down_block_types=down_block_types,
            in_channels=in_channels,
            block_out_channels=block_out_channels,
            act_fn=act_fn,
            latent_channels=latent_channels,
            up_block_types=up_block_types,
            norm_num_groups=norm_num_groups,
            layers_per_block=layers_per_block,
            out_channels=out_channels, )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
    #    pretrained_ckpt_path = vae_cfg["pretrained_ckpt_path"]
#        if save_ckpt is not None:
        state_dict = torch.load(save_ckpt)
        self.autoencoder_obs.load_state_dict(state_dict)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(-1, self.in_channels, x.shape[-2], x.shape[-1])


        x, x2 = self.autoencoder_obs.forward(x, sample_posterior = True)
        
        x = x.reshape(x.shape[0]//self.frame_num, self.frame_num, self.out_channels, x.shape[-2], x.shape[-1])

        x = x.permute(0, 2, 1, 3, 4)

        return x
        
    def encode(self, xe):
    
        # (batch, channels, frame, height, width)
    
        xe = xe.permute(0, 2, 1, 3, 4) # (batch, frame, channels, height, width)
        
        xe = xe.reshape(-1, self.in_channels, xe.shape[-2], xe.shape[-1]) # (batch × frame, channels, height, width)
        
        xe = self.autoencoder_obs.encode(xe)
        
        xe = xe.sample() # (batch × frame, channels, height, width)
        
        xe = xe.reshape(xe.shape[0]//self.frame_num, self.frame_num, xe.shape[-3], xe.shape[-2], xe.shape[-1]) # (batch, frame, channels, height, width)
        
        xe = xe.permute(0, 2, 1, 3, 4) # (batch, channels, frame, height, width)
        
        return xe
        
    def decode(self, xd):
    
        # (batch, channels, frame, height, width)
        
        xd = xd.permute(0, 2, 1, 3, 4) # (batch, frame, channels, height, width)
        
        xd = xd.reshape(-1, xd.shape[-3], xd.shape[-2], xd.shape[-1]) # (batch × frame, channels, height, width)
        
        xd = self.autoencoder_obs.decode(xd) # (batch × frame, channels, height, width)
        
        xd = xd.reshape(xd.shape[0]//self.frame_num, self.frame_num, xd.shape[-3], xd.shape[-2], xd.shape[-1]) # (batch, frame, channels, height, width)
        
        xd = xd.permute(0, 2, 1, 3, 4) # (batch, channels, frame, height, width)
        
        return xd
