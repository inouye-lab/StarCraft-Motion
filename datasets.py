import random
import os
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pandas as pd


class StarCraftMotionDataset(Dataset):
    def __init__(self, root="/local/scratch/a/bai116/datasets/StarCraftMotion_v0.9/", train=True, num_frames=10, subsampling_factor=12, split=0.75, seed=42):
        self.rng = np.random.default_rng(seed=seed)
        random.seed(seed)
        self.root = Path(root)
        metadata = pd.read_csv(self.root / 'metadata.csv')
        # print(metadata)
        self.file_list = metadata['file_name'].to_list()
        idx = np.arange(len(self.file_list), dtype=int)
        data_array = metadata.iloc[:, 1:].to_numpy()
        # _file_names = metadata['']
        self.split = np.array(split)
        self.train = train
        self.num_frames = num_frames
        self.subsampling_factor=subsampling_factor
    
        self.rng.shuffle(idx)
        file_len = len(self.file_list)
        if train:
            indices = idx[:int(file_len*split)]            
        else:
            indices = idx[int(file_len*split):]
        # start to shard!
        self.metadata = []

        for count, idx in enumerate(indices):
            fn = self.file_list[idx]
            meta = data_array[idx]
            total_frame = meta[-2]
            start_shards = np.arange(0, total_frame, self.num_frames*self.subsampling_factor).reshape(-1, 1)
            
            if start_shards[-1,0] + self.num_frames*self.subsampling_factor >= total_frame:
                start_shards = start_shards[:-1]

            if len(start_shards) > 0:
                n_shards = np.arange(0, len(start_shards)).reshape(-1, 1)
                m = np.tile(idx, (len(start_shards), 1))
                
                # file_idx, id_shards, start_shards
                self.metadata.append(np.concatenate([m,n_shards,start_shards], axis=1))
        self.metadata = np.concatenate(self.metadata, axis=0)
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        meta = self.metadata[idx]
        fp_idx, id_shards, start_frame = meta
        filename = self.file_list[fp_idx]
        track = np.load(self.root / 'tracks' / (filename + '.npz'))['track']
        chunk = torch.tensor(track[start_frame:(start_frame+self.subsampling_factor*self.num_frames):self.subsampling_factor])
        delta = torch.diff(chunk, dim=0)
        x = chunk[:-1]
        y_cont = delta[...,[2,3,13,14]] / 255.0
        y_disc = chunk[1:, :, 11].to(int)
        y_disc = (y_disc == 254) * torch.ones_like(y_disc) + (y_disc != 254) * y_disc 

        return x, y_cont, y_disc, meta


def pad_collate_fn(batch):
    x_list = []
    y_cont_list = []
    y_disc_list = []
    metadata = []
    for x, y_cont, y_disc, meta in batch: 
        metadata.append(torch.tensor(meta).reshape(1,-1))
        x = x.swapaxes(0,1)
        y_cont = y_cont.swapaxes(0,1)
        y_disc = y_disc.swapaxes(0,1)
        mask = x[:,0,7] == 0 # filter out non-exist object in this segment
        mask = torch.nonzero(torch.logical_not(mask)).squeeze()  # Find the indices of non-zero elements
        filtered_tensor = x[mask]
        y_cont = y_cont[mask]
        y_disc = y_disc[mask]
        x_list.append(filtered_tensor)
        y_cont_list.append(y_cont)
        y_disc_list.append(y_disc)
    x = pad_sequence(x_list, batch_first=True, padding_value=0).swapaxes(1,2)
    y_cont = pad_sequence(y_cont_list, batch_first=True, padding_value=0).swapaxes(1,2)
    y_disc = pad_sequence(y_disc_list, batch_first=True, padding_value=0).swapaxes(1,2)
    return x.to(torch.float32), y_cont, y_disc, torch.cat(metadata, dim=0)

if __name__ == '__main__':
    dataset =  StarCraftMotionDataset(root="/local/scratch/a/bai116/datasets/StarCraftMotion_v0.9/", train=True, num_frames=10, subsampling_factor=24, split=0.7, seed=1001)
    train_loader = DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=pad_collate_fn)

    for batch, meta in tqdm(train_loader):
        if batch.shape[0] != 10:
            print(batch.shape)

