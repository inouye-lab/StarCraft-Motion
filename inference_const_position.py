import random
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import StarCraftMotionDataset, pad_collate_fn
from model import TransformerModel, Tokenizer
import numpy as np
import argparse
import wandb
import torch_scatter
from datetime import datetime
from sklearn.metrics import confusion_matrix
import pickle
from tqdm import tqdm
from utils.unit_type_mapping import unit_type_mapping, UNIT_TYPE
import einops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model, Loss, and Optimizer
# DataLoader
tokenizer = Tokenizer()
in_test_dataset =  StarCraftMotionDataset(root="/local/scratch/a/bai116/datasets/StarCraftMotion_v0.9/", subset=2, num_frames=10, subsampling_factor=24, split=[0.7,0.15,0.15], seed=1001)
in_test_loader = DataLoader(in_test_dataset, batch_size=64, shuffle=False, collate_fn=pad_collate_fn, num_workers=8, pin_memory=True)
out_test_dataset =  StarCraftMotionDataset(root="/local/scratch/a/bai116/datasets/StarCraftMotion_v0.9/", subset=4, num_frames=10, subsampling_factor=24, split=[0.7,0.15,0.15], seed=1001)
out_test_loader = DataLoader(out_test_dataset, batch_size=64, shuffle=False, collate_fn=pad_collate_fn, num_workers=8, pin_memory=True)

unit_type = unit_type_mapping(UNIT_TYPE)
l2_loss_track = 0.
l1_loss_track = 0.
l2_loss_intention = 0.
l1_loss_intention = 0.
count = 0
mapping = torch.tensor(UNIT_TYPE[2] + UNIT_TYPE[5] + UNIT_TYPE[8]).to(device)

for i, (data, _) in tqdm(enumerate(in_test_loader)):
    if i > 100:
        break
    data = data.to(device)
    data = data.to(torch.float32)

    x = data[:,:9,:,:]
    y = data[:, 1:, :, (2,3,13,14)] / 255
    y_pred = data[:, :-1, :, (2,3,13,14)] / 255 
    mask_condition_1 = (x[:, :, :, 1] != 16) 
    mask_condition_2 = torch.isin(x[:, :, :, 0], mapping)

    # Combine both conditions
    mask = mask_condition_1 & mask_condition_2

    # Expand the mask to match the shape of loss targets
    mask = mask.unsqueeze(dim=3)  # Adjusting shape to match loss dimensions
    
    _id = unit_type[x[:,:,:,0].flatten().cpu().to(int)].to(int)
    
    l2 = ((y_pred[:,:,:,[0,1]]-y[:,:,:,[0,1]]) ** 2).detach().sum().cpu().flatten()
    l1 = torch.abs(y_pred[:,:,:,[0,1]]-y[:,:,:,[0,1]]).detach().sum().cpu().flatten()
    l2_loss_track += l2
    l1_loss_track += l1

    l2 = ((y_pred[:,:,:,[2,3]]-y[:,:,:,[2,3]]) ** 2).detach().sum().cpu().flatten()
    l1 = torch.abs(y_pred[:,:,:,[2,3]]-y[:,:,:,[2,3]]).detach().sum().cpu().flatten()
    l2_loss_intention += l2
    l1_loss_intention += l1
    count += mask.sum().to('cpu')

print(l2_loss_track/count, l1_loss_track/count, l2_loss_intention/count, l1_loss_intention/count)

unit_type = unit_type_mapping(UNIT_TYPE)
l2_loss_track = 0.
l1_loss_track = 0.
l2_loss_intention = 0.
l1_loss_intention = 0.
count = 0
mapping = torch.tensor(UNIT_TYPE[2] + UNIT_TYPE[5] + UNIT_TYPE[8]).to(device)

for i, (data, _) in tqdm(enumerate(out_test_loader)):
    if i > 100:
        break
    data = data.to(device)
    data = data.to(torch.float32)

    x = data[:,:9,:,:]
    y = data[:, 1:, :, (2,3,13,14)] / 255
    y_pred = data[:, :-1, :, (2,3,13,14)] / 255 
    mask_condition_1 = (x[:, :, :, 1] != 16) 
    mask_condition_2 = torch.isin(x[:, :, :, 0], mapping)

    # Combine both conditions
    mask = mask_condition_1 & mask_condition_2

    # Expand the mask to match the shape of loss targets
    mask = mask.unsqueeze(dim=3)  # Adjusting shape to match loss dimensions
    
    _id = unit_type[x[:,:,:,0].flatten().cpu().to(int)].to(int)
    
    l2 = ((y_pred[:,:,:,[0,1]]-y[:,:,:,[0,1]]) ** 2).detach().sum().cpu().flatten()
    l1 = torch.abs(y_pred[:,:,:,[0,1]]-y[:,:,:,[0,1]]).detach().sum().cpu().flatten()
    l2_loss_track += l2
    l1_loss_track += l1

    l2 = ((y_pred[:,:,:,[2,3]]-y[:,:,:,[2,3]]) ** 2).detach().sum().cpu().flatten()
    l1 = torch.abs(y_pred[:,:,:,[2,3]]-y[:,:,:,[2,3]]).detach().sum().cpu().flatten()
    l2_loss_intention += l2
    l1_loss_intention += l1
    count += mask.sum().to('cpu')

print(l2_loss_track/count, l1_loss_track/count, l2_loss_intention/count, l1_loss_intention/count)