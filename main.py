import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import StarCraftMotionDataset, pad_collate_fn
from model import TransformerModel, Tokenizer, SimpleMLP
import numpy as np
import argparse
import wandb
import torch_scatter
from datetime import datetime
from tqdm import tqdm
from utils.unit_type_mapping import unit_type_mapping, UNIT_TYPE


if not os.path.exists('data'):
    os.makedirs('data')

def main(args):
    # wandb
    wandb_project = "StarCraftMotion"
    # setup WanDB
    if not args.no_wandb:
        wandb.init(project=wandb_project,
                    entity='inouye-lab',
                    config=vars(args))
        wandb.run.log_code()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    # Model, Loss, and Optimizer
    # DataLoader
    # train_dataset = StarCraftMotionDataset(num_samples, num_categories, max_len, embedding_dim)
    train_dataset =  StarCraftMotionDataset(root=args.data, train=True, num_frames=args.num_frames, subsampling_factor=args.subsampling_factor, split=0.75, seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_fn, num_workers=args.num_workers, pin_memory=True)
    tokenizer = Tokenizer()
    test_dataset =  StarCraftMotionDataset(root=args.data, train=False, num_frames=args.num_frames, subsampling_factor=args.subsampling_factor, split=0.75, seed=args.seed)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_fn, num_workers=args.num_workers, pin_memory=True)
    if not (args.temporal or args.spatial):
        single_model = SimpleMLP(n_embd=args.hidden_dim)
    else:
        single_model = TransformerModel(n_embd=args.hidden_dim, n_ctx=args.num_frames-1, n_heads=args.num_heads, n_layer=args.num_layers, spatial=args.spatial, temporal=args.temporal)

        
    model = nn.DataParallel(single_model)
    model = single_model
    model.to(device)
    criterion_con = nn.MSELoss(reduction='sum') 
    criterion_dis = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    step = 0
    mapping = torch.tensor(UNIT_TYPE[2] + UNIT_TYPE[5] + UNIT_TYPE[8]).to(device)
    for e in range(args.num_epochs):
        model.train()
        training_loss = 0
        classification_loss = 0
        count = 0
        for iterates, (x, y_cont, y_disc, meta) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training with Epoch {}".format(e)):
            step += 1
            x = x.to(device)
            y_cont = y_cont.to(device)
            y_disc = y_disc.to(device)
            tokens, padding_mask = tokenizer(x)
            y_cont_hat, y_disc_hat = model(tokens, padding_mask)
            loss = criterion_con(y_cont_hat.reshape(-1,4), y_cont.reshape(-1,4))
            loss_intention = criterion_dis(y_disc_hat.reshape(-1, y_disc_hat.shape[-1]), (y_disc.squeeze()).reshape(-1))
            optimizer.zero_grad()
            (loss+loss_intention).backward()
            training_loss += loss.item()
            classification_loss += loss_intention.item()
            mask = y_cont[:,:,:,0] != 0
            unit_count = mask.nonzero().shape[0]
            count += unit_count
            optimizer.step()
            wandb.log({"loss": training_loss/count, "intention_loss": classification_loss/count}, step=iterates+e*len(train_loader))
        
        # test here: test + debug + add per type error.
            if iterates % 2000 == 0 and iterates != 0:
                model.eval()
                with torch.no_grad():
                    
                    test_loss = 0
                    test_data = 0.
                    unit_type_loss = torch.zeros(512).to(device)
                    unit_count = torch.zeros(512).to(device)
                    for j, (x, y_cont, y_disc, meta) in enumerate(test_loader):
                        if j > 10:
                            break
                        x = x.to(device)
                        y_cont = y_cont.to(device)
                        y_disc = y_disc.to(device)            
                        tokens, padding_mask = tokenizer(x)
                        y_cont_hat, y_disc_hat = model(tokens, padding_mask)
        
                        mask_condition_1 = (x[:, :, :, 1] != 16) 
                        mask_condition_2 = torch.isin(x[:, :, :, 0], mapping)

                        # Combine both conditions
                        mask = mask_condition_1 & mask_condition_2

                        # Expand the mask to match the shape of loss targets
                        mask = mask.unsqueeze(dim=3)  # Adjusting shape to match loss dimensions
                        unit_count = torch.nonzero((x*mask)[:,0,:,0] != 0).shape[0]
                        test_loss += criterion_con(y_cont_hat*mask, y_cont*mask).item()
                        test_data += unit_count
                        torch.cuda.empty_cache()
                    unit_type_loss = unit_type_loss / unit_count
                    single_model.save_model(optimizer, 'data/{}_{}.pt'.format(wandb.run.name, iterates))
                    wandb.log({"test_loss": test_loss / test_data}, step=iterates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='StarCraft-Motion Baseline')
    parser.add_argument('--no_wandb', default=False, action="store_true")
    parser.add_argument('--seed', default=1001, type=int)
    # data
    parser.add_argument('--data', default="/local/scratch/a/bai116/datasets/StarCraftMotion_v0.9/", type=str)
    parser.add_argument('--num_frames', default=10, type=int)
    parser.add_argument('--subsampling_factor', default=24, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=32, type=int)
    # model
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--spatial', default=0)
    parser.add_argument('--temporal', default=0)
    # training
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)

    args = parser.parse_args()
    main(args)
