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

def main(args):
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
    in_test_dataset =  StarCraftMotionDataset(root="/local/scratch/a/bai116/datasets/StarCraftMotion_v0.9/", subset=2, num_frames=args.num_frames, subsampling_factor=24, split=[0.7,0.15,0.15], seed=1001)
    in_test_loader = DataLoader(in_test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_fn, num_workers=args.num_workers, pin_memory=True)
    
    out_test_dataset =  StarCraftMotionDataset(root="/local/scratch/a/bai116/datasets/StarCraftMotion_v0.9/", subset=4, num_frames=args.num_frames, subsampling_factor=24, split=[0.7,0.15,0.15], seed=1001)
    out_test_loader = DataLoader(out_test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_fn, num_workers=args.num_workers, pin_memory=True)
    
    path = 'data/' + hparam['pretrained_model']

    y_count_pred = None
    y_disc_pred = None

    single_model = TransformerModel(n_embd=args.hidden_dim, n_ctx=9, n_heads=args.num_heads, n_layer=args.num_layers)
    optimizer = optim.Adam(single_model.parameters(), lr=0.001)
    single_model.load_model(optimizer=optimizer, path=path, device=device)
    model = nn.DataParallel(single_model)
    model = model.to(device)
    model.eval()

    l2_loss_track = np.zeros(args.num_frames-9)
    l1_loss_track = np.zeros(args.num_frames-9)
    l2_loss_intention = np.zeros(args.num_frames-9)
    l1_loss_intention = np.zeros(args.num_frames-9)
    count = np.zeros(args.num_frames-9)
    conf_mat = [np.zeros((26,26)) for _ in range(args.num_frames-9)]
    for i, (data, meta) in tqdm(enumerate(in_test_loader)):
        if i > 1:
            break
        data = data.to(torch.float32)
        data = data.to(device)

        x = data[:, 0:9]
        y = (data[:, 1:, :, [2,3,13,14]]-data[:, :-1, :, [2,3,13,14]]) / 255
        y_disc = data[:,1:, :, 11].to(int)
        for count in range(args.num_frames-9):
            tokens, padding_mask = tokenizer(x)
            continuous, intention = model(tokens, padding_mask)
            next_token = torch.cat((data[:,9,:,[0,1]], continuous[:,-1,:,[0,1]], data[:,9,:,4:]), dim=2).unsqueeze(dim=1)
            if y_count_pred is not None:
                y_count_pred = torch.cat((y_count_pred, continuous[:, -1, :, 0:4].unsqueeze(dim=1)), dim=1)
            else:
                y_count_pred = continuous[:,-1, :, 0:4].unsqueeze(dim=1)
            if y_disc_pred is not None:
                y_disc_pred = torch.cat((y_disc_pred, intention[:,-1].unsqueeze(dim=1)))
            else:
                y_disc_pred = intention[:,-1].unsqueeze(dim=1)
            x = torch.cat((x[:,1:], next_token), dim=1)

        # track prediction
        count += y_disc.shape[0] * y_disc.shape[2] * np.ones(args.num_frames-9)
        l2_loss_track += ((y_count_pred[:,:,:,[0,1]]-y[:,10:,:,[0,1]]) ** 2).sum(dim=(0,2,3)).detach().cpu().numpy()
        l1_loss_track += torch.abs((y_count_pred[:,:,:,[0,1]]-y[:,10:,:,[0,1]])).sum(dim=(0,2,3)).detach().cpu().numpy()
    
        l2_loss_intention += ((y_count_pred[:,:,:,[2,3]]-y[:,10:,:,[2,3]]) ** 2).sum(dim=(0,2,3)).detach().cpu().numpy()
        l1_loss_intention += torch.abs((y_count_pred[:,:,:,[2,3]]-y[:,10:,:,[2,3]])).sum(dim=(0,2,3)).detach().cpu().numpy()

        intention_pred = y_disc_pred.argmax(dim=-1) # B * T * U
        for j in range(args.num_frames-9):
            y_true = y_disc[:, j, :]
            y_true[y_true == 254] = 1
            y_true[y_true == 255] = 25 
            conf_mat[j] += confusion_matrix(y_true.cpu().flatten(), intention_pred[:,j,:].cpu().flatten(), labels=list(range(26)))

    pickle.dump([l2_loss_track/count, l1_loss_track/count, l2_loss_intention/count, l1_loss_intention/count, conf_mat], open("result_indomain.pkl",'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='StarCraft-Motion Baseline')
    parser.add_argument('--no_wandb', default=False, action="store_true")
    parser.add_argument('--seed', default=1001, type=int)
    # data
    parser.add_argument('--data', default="/local/scratch/a/bai116/datasets/StarCraftMotion_v0.9/", type=str)
    parser.add_argument('--num_frames', default=60, type=int)
    parser.add_argument('--subsampling_factor', default=24, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    # model
    parser.add_argument('--pretrained_model', type=str)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--spatial_only', default=False, action="store_true")
    parser.add_argument('--temporal_only', default=False, action="store_true")
    parser.add_argument('--future_window', type=int)



    args = parser.parse_args()
    main(args)