import torch
from torch import nn
import pytorch_lightning as pl

from ..distributions import kl_from_standard_normal, ensemble_nll_normal
from ..distributions import sample_from_standard_normal

import os
import matplotlib.pyplot as plt


class AutoencoderKL(pl.LightningModule):
    def __init__(
        self, 
        encoder, decoder, 
        kl_weight=0.01,     
        encoded_channels=64,
        hidden_width=32,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_width = hidden_width
        self.to_moments = nn.Conv3d(encoded_channels, 2*hidden_width,
            kernel_size=1)
        self.to_decoder = nn.Conv3d(hidden_width, encoded_channels,
            kernel_size=1)
        self.log_var = nn.Parameter(torch.zeros(size=()))
        self.kl_weight = kl_weight

    def encode(self, x):
        h = self.encoder(x)
        (mean, log_var) = torch.chunk(self.to_moments(h), 2, dim=1)
        return (mean, log_var)

    def decode(self, z):
        z = self.to_decoder(z)
        dec = self.decoder(z)
        return dec
        
    def plot_validation_results(self, batch, epoch, save_dir):
        (x, y) = batch
        while isinstance(x, list) or isinstance(x, tuple):
            x = x[0][0]
        (y_pred, _, _) = self.forward(x)
        
        # Select a random sample from the batch
        idx = torch.randint(0, x.size(0), (1,)).item()
        
        # Plot the input and reconstructed frames side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(x[idx, :, 0].detach().cpu().numpy(), cmap='gray')
        ax1.set_title("Input Frame")
        ax1.axis("off")
        ax2.imshow(y_pred[idx, :, 0].detach().cpu().numpy(), cmap='gray')
        ax2.set_title("Reconstructed Frame")
        ax2.axis("off")
        
        # Save the plot to the specified directory
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"validation_epoch_{epoch}.png"))
        plt.close()
        
    def validation_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="val")
        
        # Plot validation results every n epochs
        n = 5  # Change this to the desired number of epochs
        if self.current_epoch % n == 0:
            save_dir = "/project/nowcastdlm/validation_plots"  # Change this to the desired directory
            self.plot_validation_results(batch, self.current_epoch, save_dir)
        
    def predict_step(self, batch):
        (inputs, target) = batch
        (mean, log_var) = self.encode(inputs)
        z = sample_from_standard_normal(mean, log_var)
        dec = self.decode(z)
        return dec

    def forward(self, input, sample_posterior=True):
        (mean, log_var) = self.encode(input)
        if sample_posterior:
            z = sample_from_standard_normal(mean, log_var)
        else:
            z = mean
        dec = self.decode(z)
        return (dec, mean, log_var)

    def _loss(self, batch):
        (x,y) = batch
        while isinstance(x, list) or isinstance(x, tuple):
            x = x[0][0]
        (y_pred, mean, log_var) = self.forward(x)

        rec_loss = (y-y_pred).abs().mean()
        kl_loss = kl_from_standard_normal(mean, log_var)

        total_loss = rec_loss + self.kl_weight * kl_loss

        return (total_loss, rec_loss, kl_loss)

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)[0]
        self.log("train_loss", loss)
        return loss

    @torch.no_grad()
    def val_test_step(self, batch, batch_idx, split="val"):
        (total_loss, rec_loss, kl_loss) = self._loss(batch)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True}
        self.log(f"{split}_loss", total_loss, **log_params)
        self.log(f"{split}_rec_loss", rec_loss.mean(), **log_params)
        self.log(f"{split}_kl_loss", kl_loss, **log_params)

    def validation_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="val")

    def test_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5,
            betas=(0.5, 0.9), weight_decay=1e-3)
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.25, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr,
                "monitor": "val_rec_loss",
                "frequency": 1,
            },
        }
