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

# Hyperparameters
num_heads = 4
hidden_dim = 128
num_layers = 2
batch_size = 4
num_epochs = 1
learning_rate = 1e-3


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
    # train_dataset = StarCraftMotionDataset(num_samples, num_categories, max_len, embedding_dim)
    train_dataset = StarCraftMotionDataset("/local/scratch/a/bai116/datasets/StarCraftMotion_v0.2/processed", train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    tokenizer = Tokenizer()
    test_dataset = StarCraftMotionDataset("/local/scratch/a/bai116/datasets/StarCraftMotion_v0.2/processed", train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
    model = TransformerModel(n_embd=hidden_dim, n_ctx=9, n_heads=4, n_layer=num_layers)
    # model = nn.DataParallel(model)
    model.to(device)
    criterion_con = nn.MSELoss(reduction='sum') 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    step = 0
    for e in range(num_epochs):
        model.train()
        training_loss = 0
        count = 0
        for iterates, data in enumerate(train_loader):
            data = data.to(device)
            step += 1
            data = data[:, :, 1:, :]     # Remove the first frame
            data = torch.swapaxes(data, 1, 2)
            x = data[:, :-1, :, :]
            y = (data[:, 1:, :, 2:6] - data[:, :-1, :, 2:6]) / 255
            # y_cont = (data[:, 1:, :, 2:6] - data[:, :-1, :, 2:6]) / 255
            # y_disc = 
            tokens, padding_mask = tokenizer(x)
            
            # Data tokenization and preprocessing
           
            continuous = model(tokens, padding_mask)
            loss = criterion_con(continuous, y)
            # print(x.shape)
            # print(loss.item())
            # print_gpu_usage()
            optimizer.zero_grad()
            loss.backward()
            training_loss += loss.item()
            mask = y[:,:,:,0] != 0
            unit_count = mask.nonzero().shape[0]
            count += unit_count
            optimizer.step()
            torch.cuda.empty_cache()
            print(loss.item() / unit_count)
            print(training_loss / count)
            print("-----")
            wandb.log({"loss": training_loss/count}, step=iterates)
        
        # test here: test + debug + add per type error.
            if iterates % 10 == 0 and iterates != 0: 
                model.eval()
                test_loss = 0
                test_data = 0.
                unit_type_loss = torch.zeros(512).to(device)
                unit_count = torch.zeros(512).to(device)
                with torch.no_grad():
                    for j, data in enumerate(test_loader):
                        if j > 100:
                            break
                        data = data.to(device)
                        # data = data[:, :, 1:, :]     # Remove the first frame
                        data = torch.swapaxes(data, 1, 2)
                        x = data[:, :-1, :, :]
                        y = (data[:, 1:, :, 2:6]-data[:, :-1, :, 2:6]) / 255
                        tokens, padding_mask = tokenizer.tokenize(x)
                        pred = model(tokens, padding_mask)
                        unit_count = torch.nonzero(data[:,0,:,0] != 0).shape[0]
                        test_loss += criterion_con(pred, y).item()
                        test_data += unit_count
                        torch.cuda.empty_cache()
                unit_type_loss = unit_type_loss / unit_count
                torch.save(unit_type_loss, 'data/results.pt')
                model.save_model(optimizer, 'data/model_{}.pt'.format(datetime.now().strftime("%Y%m%d%H%M%S")))
                wandb.log({"test_loss": test_loss / test_data}, step=iterates)


def print_gpu_usage():
    if torch.cuda.is_available():
        print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached GPU memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    else:
        print("CUDA is not available on this device.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='StarCraft-Motion Baseline')
    # parser.add_argument('--config_file', help='config file', default="config.json")
    parser.add_argument('--no_wandb', default=False, action="store_true")
    # parser.add_argument('--seed', default=1001, type=int)
    # parser.add_argument('--num_clients', default=1, type=int)
    # parser.add_argument('--batch_size', default=16, type=int)
    # parser.add_argument('--iid', default=1, type=float)
    # parser.add_argument('--server_method', default='FedAvg')
    # parser.add_argument('--fraction', default=1, type=float)
    # parser.add_argument('--f', default=10, type=int)
    # parser.add_argument('--num_rounds', default=20, type=int)
    # parser.add_argument('--dataset', default='PACS')
    # parser.add_argument('--split_scheme', default='official')
    # parser.add_argument('--client_method', default='ERM')
    # parser.add_argument('--local_epochs', default=1, type=int)
    # parser.add_argument('--n_groups_per_batch', default=2, type=int)
    # parser.add_argument('--optimizer', default='torch.optim.Adam')
    # parser.add_argument('--lr', default=3e-5, type=float)
    # parser.add_argument('--momentum', default=0, type=float)
    # parser.add_argument('--weight_decay', default=0, type=float)
    # parser.add_argument('--eps', default=1e-8, type=float)
    # parser.add_argument('--hparam1', default=1, type=float, help="irm: lambda; rex: lambda; fish: meta_lr; mixup: alpha; mmd: lambda; coral: lambda; groupdro: groupdro_eta; fedprox: mu; feddg: ratio; fedadg: alpha; fedgma: mask_threshold; fedsr: l2_regularizer;")
    # parser.add_argument('--hparam2', default=1, type=float, help="fedsr: cmi_regularizer; irm: penalty_anneal_iters; fedadg: second_local_epochs")
    # parser.add_argument('--hparam3', default=0, type=float)
    # parser.add_argument('--hparam4', default=0, type=float)
    # parser.add_argument('--hparam5', default=0, type=float)

    args = parser.parse_args()
    main(args)
