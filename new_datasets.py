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
    def __init__(self, root, subset=0, num_frames=[10,20], subsampling_factor=[12, 12], split=[0.75, 0.25], seed=42):
        """
        Args:
            root (str): Path to the root with .npz files.
            subset (int): 0: train, 1: test
            num_frame (int): Number of frames per shard after subsampling.
            subsampling_factor (int): Interval to sample frames.
            split (list): Ratio 0, 1, 2.
            seed (int): Seed for reproducibility.
        """
        rng = np.random.default_rng(seed=seed)
        random.seed(seed)
        self.root = Path(root)
        self._file_names = [f.replace('.npz', '') for f in os.listdir(root) if f.endswith('.npz')]
        self.split = np.array(split)
        self.subset = subset
        self.subsampling_factor=subsampling_factor
        self.num_frames = num_frames
        metadata = pd.load_csv(self.root / 'metadata.csv')
        
        file_list = metadata['file_name'].tolist()
        self.rng.shuffle(file_list)
        mask = np.isin(metadata[:, 1], map_idx)
        tem_meta = metadata[mask]
        rng.shuffle(tem_meta)
        self.split = (self.split * len(tem_meta)).astype(int)
        if subset == 0:
            subset_meta = tem_meta[:self.split[0]]
        elif subset == 1:
            subset_meta = tem_meta[self.split[0]:(self.split[0]+self.split[1])]
        elif subset == 2:
            subset_meta = tem_meta[(self.split[0]+self.split[1]):(self.split[0]+self.split[1]+self.split[2])]
        else:
            subset_meta = tem_meta

    
        # start to shard!
        self.metadata = []
        
        for count, _ in enumerate(subset_meta):
            total_frame = subset_meta[count, 2]
            start_shards = np.arange(0, total_frame, self.num_frames*self.subsampling_factor).reshape(-1, 1)
            if start_shards[-1,0] + self.num_frames*self.subsampling_factor >= total_frame:
                start_shards = start_shards[:-1]
            if len(start_shards) > 0:
                n_shards = np.arange(0, len(start_shards)).reshape(-1, 1)
                m = np.tile(subset_meta[count], (len(start_shards), 1))
                # add idx shards
                self.metadata.append(np.concatenate([m,n_shards,start_shards], axis=1))
        self.metadata = np.concatenate(self.metadata, axis=0)




    def __len__(self):
        # Total number of shards across all replays.
        return len(self.metadata)


    def __getitem__(self, idx):
        meta = self.metadata[idx]
        fp_idx = meta[0].astype(int)
        start_frame = meta[-1].astype(int)
        # if self.cache:
        #     track = self.cache_track[fp_idx]
        # else:
        filename = self.replay_map[fp_idx]
        track = np.load(self.root / 'tracks' / (filename + '.npz'))['track']
        chuck = torch.tensor(track[start_frame:(start_frame+self.subsampling_factor*self.num_frames):self.subsampling_factor])
        return chuck, meta

def pad_collate_fn(batch):
    tensor_list = []
    for tensor, meta in batch:
        tensor = tensor.swapaxes(0,1)
        mask = tensor[:,0,7] == 0 # filter out non-exist object in this segment
        mask = torch.nonzero(torch.logical_not(mask))  # Find the indices of non-zero elements
        filtered_tensor = tensor[mask[:,0]]
        tensor_list.append(filtered_tensor)
    v = pad_sequence(tensor_list, batch_first=True, padding_value=0)
    
    return v.swapaxes(1,2), meta

if __name__ == '__main__':
    dataset =  StarCraftMotionDataset(root="/local/scratch/a/bai116/datasets/StarCraftMotion_v0.9/", subset=0, num_frames=10, subsampling_factor=24, split=[0.7,0.15,0.15], seed=1001)
    train_loader = DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=pad_collate_fn)

    for batch, meta in tqdm(train_loader):
        if batch.shape[0] != 10:
            print(batch.shape)

    # for batch, meta in train_loader:
    #     if batch.shape[0] != 10:
    #         print(batch.shape)
    #         break
        
