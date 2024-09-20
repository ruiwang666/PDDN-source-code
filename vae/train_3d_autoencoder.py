import gc
import gzip
import os
import pickle

from fire import Fire
import numpy as np
from omegaconf import OmegaConf

from autoencoder_3d.features import batch, patches, split, transform
from autoencoder_3d.models.autoenc import encoder, training

file_dir = os.path.dirname(os.path.abspath(__file__))


import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torch.utils.data import TensorDataset

from torch import from_numpy





class RadarDataModule(LightningDataModule):
    def __init__(self, data_train, data_val, data_test, num_workers, batch_size=32):
        super().__init__()

        
        self.batch_size = batch_size
        self.num_workers = num_workers


        
        self.train_data = TensorDataset(torch.tensor(data_train, dtype = torch.float32), torch.tensor(data_train, dtype = torch.float32))
        self.val_data = TensorDataset(torch.tensor(data_val, dtype = torch.float32), torch.tensor(data_val, dtype = torch.float32))
        self.test_data = TensorDataset(torch.tensor(data_test, dtype = torch.float32), torch.tensor(data_test, dtype = torch.float32))



    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = self.train_data
            self.val_dataset = self.val_data

        if stage == 'test' or stage is None:
            self.test_dataset = self.test_data

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, num_workers=self.num_workers, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=self.num_workers, batch_size=self.batch_size)


#
#
def setup_model(
    model_dir=None
):
    enc = encoder.SimpleConvEncoder()

    dec = encoder.SimpleConvDecoder()
    (autoencoder, trainer) = training.setup_autoenc_training(
        encoder=enc,
        decoder=dec,
        model_dir=model_dir
    )
    gc.collect()
    return (autoencoder, trainer)


def train(
    batch_size=16,
    sampler_file=None,
    model_dir=None,
    ckpt_path=None
):
    print("Loading data...")

    
    # load radar train data
    radar_data_train = np.load('')

    # load radar val data
    radar_data_val = np.load('')

    # load radar test data
    radar_data_test = np.load('')
    
    
    
    # Instantiate the DataModule with the loaded data
    datamodule = RadarDataModule(data_train = radar_data_train, data_val = radar_data_val, data_test = radar_data_val, num_workers = 8, batch_size=2)

    print("Setting up model...")
    (model, trainer) = setup_model(model_dir=model_dir)

    print("Starting training...")

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


def main(config=None, **kwargs):
    config = OmegaConf.load(config) if (config is not None) else {}
    config.update(kwargs)
    train(**config)


if __name__ == "__main__":
    Fire(main)
