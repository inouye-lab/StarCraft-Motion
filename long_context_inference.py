import random
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from dataset import StarCraftMotionDataset, pad_collate_fn
from model import TransformerModel, Tokenizer
import numpy as np
import argparse
import wandb
import torch_scatter
from datetime import datetime



def main(args):
    # Hyperparameters
    hidden_dim = 128
    num_layers = 2
    batch_size = 4
    learning_rate = 1e-3
    hparam = vars(args)
    # wandb
    
    wandb_project = "StarCraftMotion"

    # setup WanDB
    if not args.no_wandb:
        wandb.init(project=wandb_project,
                    entity='inouye-lab',
                    config=hparam)
        wandb.run.log_code()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Model, Loss, and Optimizer
    # DataLoader
    tokenizer = Tokenizer()
    test_dataset = StarCraftMotionDataset("/local/scratch/a/bai116/datasets/StarCraftMotion_v0.2/{}".format(hparam['testset_folder']), train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
    model = TransformerModel(n_embd=hidden_dim, n_ctx=9, n_heads=4, n_layer=num_layers)
    model.to(device)
    criterion_con = nn.MSELoss(reduction='sum') 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    path = hparam['data'] + hparam['pretrained_model']
    model.load(optimizer=optimizer, path=path, device=device)
    model.eval()
    test_loss = 0
    test_data = 0.
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            data = torch.swapaxes(data, 1, 2)
            x = data[:, 0:5, :, :]
            y = (data[:, 1:, :, 2:6]-data[:, :-1, :, 2:6]) / 255
            tokens, padding_mask = tokenizer.tokenize(x)
            pred = model(tokens, padding_mask)
            unit_count = torch.nonzero(data[:,0,:,0] != 0).shape[0]
            test_loss += criterion_con(pred, y).item()
            test_data += unit_count
            torch.cuda.empty_cache()
    unit_type_loss = unit_type_loss / unit_count
    torch.save(unit_type_loss, 'data/results.pt')
    
    wandb.log({"test_loss": test_loss / test_data})

    

    # DataLoader
    # train_dataset = StarCraftMotionDataset(num_samples, num_categories, max_len, embedding_dim)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='StarCraft-Motion Inference')
    parser.add_argument('--data', default="/local/scratch/a/bai116/datasets/StarCraftMotion_v0.2/")
    parser.add_argument('--pretrained_model', type=float)
    parser.add_argument('--testset_folder', default="config.json")
    parser.add_argument('--no_wandb', default=False, action="store_true")
    parser.add_argument('--len-frame-pred', default=1, type=int)

    args = parser.parse_args()
    main(args)