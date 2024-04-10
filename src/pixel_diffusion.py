import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .denoising_diffusion_process import *
from .autoencoder import *


class PixelDiffusion(pl.LightningModule):

    def __init__(self,
                 train_dataset,
                 valid_dataset=None, 
                 batch_size=1,
                 lr=1e-3, 
                 loss_fn=F.mse_loss,
                 schedule="cosine",
                 num_timesteps=1000,
                 sampler=None):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr = lr
        self.batch_size=batch_size
        
        input_dim = 16
        self.model = DenoisingDiffusionConditionalProcess(input_dim, 
                                                          channels=2, 
                                                          dim_mults=(1, 2, 4, 8),
                                                          loss_fn=loss_fn, 
                                                          schedule=schedule, 
                                                          num_timesteps=num_timesteps, 
                                                          sampler=sampler)


    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.output_T(self.model(*args, **kwargs))
    

    def input_T(self, input):
        return input.clip(-1, 1)
    

    def output_T(self, input):
        return input.clip(-1, 1)


    def training_step(self, batch_data, batch_idx):   
        images, _ = batch_data
        loss = self.model.p_loss(self.input_T(images))

        self.log('train_loss', loss, on_step=True, on_epoch=True)

        return loss
    

    def validation_step(self, batch_data, batch_idx):     
        images, _ = batch_data
        loss = self.model.p_loss(self.input_T(images))

        self.log('val_loss',loss, on_step=True, on_epoch=True)
        
        return loss
    

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4)
    

    def val_dataloader(self):
        if self.valid_dataset is not None:
            return DataLoader(self.valid_dataset,
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=4)
        else:
            return None


    def configure_optimizers(self):
        return  torch.optim.AdamW(list(filter(lambda p: p.requires_grad, self.model.parameters())), lr=self.lr)


class PixelDiffusionConditional_v2(PixelDiffusion):
    def __init__(self,
                 train_dataset, 
                 input_dim=16,
                 loc_dim=7,
                 channels=2, 
                 dim_mults=(1, 2, 4, 8),
                 valid_dataset=None, 
                 batch_size=1,
                 lr=1e-3, 
                 loss_fn=F.mse_loss,
                 schedule="cosine",
                 num_timesteps=1000,
                 sampler=None):
        pl.LightningModule.__init__(self)

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.batch_size=batch_size
        self.lr = lr

        self.model = DenoisingDiffusionConditionalProcess(input_dim, 
                                                          loc_dim=loc_dim,
                                                          channels=channels, 
                                                          dim_mults=dim_mults,
                                                          loss_fn=loss_fn, 
                                                          schedule=schedule, 
                                                          num_timesteps=num_timesteps, 
                                                          sampler=sampler)
    
    def training_step(self, batch_data, batch_idx):   
        signal_vec, location_vec, _ = batch_data
        loss = self.model.p_loss(self.input_T(signal_vec), location_vec)

        self.log('train_loss', loss, on_step=True, on_epoch=True)
        
        return loss
            
    def validation_step(self, batch_data, batch_idx):
        signal_vec, location_vec, _ = batch_data

        loss = self.model.p_loss(self.input_T(signal_vec), location_vec)

        self.log('val_loss', loss, on_step=True, on_epoch=True)
        
        return loss


