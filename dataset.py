import random
import os

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class StarCraftMotionDataset(Dataset):
    def __init__(self, root, train=True, split=0.8, seed=42):
        """
        Args:
            root (str): Path to the root with .npy files.
            train (boolean): true for training and false for testing splits.
            split_ratio (float): Ratio of training data (e.g., 0.8 means 80% training, 20% testing).
            seed (int): Seed for reproducibility.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.file_names = [f for f in os.listdir(root) if f.endswith('.npy')]
        self.split = split
        self.train = train

        # Set seed for reproducibility
        random.seed(seed)
        
        # Shuffle files
        random.shuffle(self.file_names)
        
        # Split into train and test sets based on split_ratio
        split_index = int(len(self.file_names) * split)
        if self.train:
            self.file_names = self.file_names[:split_index]
        else:
            self.file_names = self.file_names[split_index:]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root, self.file_names[idx])
        data = np.load(file_path)

        return torch.tensor(data, dtype=torch.float32)

def pad_collate_fn(batch):
    """
    Custom collate function to handle varying sizes of data by padding tensors to the same size.
    
    Args:
        batch (list of tensors): A batch of tensors with varying N sizes, but the same T and A.
    
    Returns:
        Padded batch of shape (B, N_max, T, A).
    """
    # Find the maximum N in the batch (shape: N, T, A)
    tensor_list = []
    for tensor in batch:
        mask_1 = tensor[:,0,7] == 0 # filter out non-exist object in this segment
        mask_2 = torch.any(torch.abs(tensor[:,1:,2] - tensor[:,:-1,2]) + torch.abs(tensor[:,1:,3] - tensor[:,:-1,3]) > 20, dim=1)
        mask = torch.nonzero(torch.logical_not(torch.logical_or(mask_1, mask_2)))  # Find the indices of non-zero elements
        
        filtered_tensor = tensor[mask[:,0]]

        tensor_list.append(filtered_tensor)

    v = pad_sequence(tensor_list, batch_first=True, padding_value=0)
    return v