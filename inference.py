import argparse
import pickle
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torch_scatter

from sklearn.metrics import confusion_matrix

from utils.unit_type_mapping import unit_type_mapping
from model import TransformerModel, Tokenizer, SimpleMLP
from datasets import StarCraftMotionDataset, pad_collate_fn
from utils.unit_type_mapping import unit_type_mapping, UNIT_TYPE


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Model, Loss, and Optimizer
    # DataLoader
    tokenizer = Tokenizer()
    in_test_dataset =  StarCraftMotionDataset(root="/local/scratch/a/bai116/datasets/StarCraftMotion_v0.9/", subset=2, num_frames=10, subsampling_factor=24, split=[0.7,0.15,0.15], seed=1001)
    in_test_loader = DataLoader(in_test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_fn, num_workers=args.num_workers, pin_memory=True)
    
    out_test_dataset =  StarCraftMotionDataset(root="/local/scratch/a/bai116/datasets/StarCraftMotion_v0.9/", subset=4, num_frames=10, subsampling_factor=24, split=[0.7,0.15,0.15], seed=1001)
    out_test_loader = DataLoader(out_test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_fn, num_workers=args.num_workers, pin_memory=True)
    
    path = 'data/' + args.pretrained_model

    if not (args.temporal or args.spatial):
        single_model = SimpleMLP(n_embd=args.hidden_dim)
    else:
        single_model = TransformerModel(n_embd=args.hidden_dim, n_ctx=args.num_frames-1, n_heads=args.num_heads, n_layer=args.num_layers, spatial=args.spatial, temporal=args.temporal)
        

    optimizer = optim.Adam(single_model.parameters(), lr=0.001)
    single_model.load_model(optimizer=optimizer, path=path, device=device)
    model = nn.DataParallel(single_model)
    model = model.to(device)
    model.eval()

    l2_loss_track = torch.zeros(4)
    l1_loss_track = torch.zeros(4)
    l2_loss_intention = torch.zeros(4)
    l1_loss_intention = torch.zeros(4)
    count = torch.zeros(4)
    mapping = torch.tensor(UNIT_TYPE[2] + UNIT_TYPE[5] + UNIT_TYPE[8]).to(device)
    conf_mat = torch.zeros((4,26,26))
    for i, (data, meta) in tqdm(enumerate(in_test_loader)):
        if i >= 400:
            break
        data = data.to(torch.float32)
        data = data.to(device)

        x = data[:, :-1]
        mask_condition_1 = (x[:, :, :, 1] != 16) 
        mask_condition_2 = torch.isin(x[:, :, :, 0], mapping)

        # Combine both conditions
        mask = mask_condition_1 & mask_condition_2

        # Expand the mask to match the shape of loss targets
        mask = mask.unsqueeze(dim=3)  # Adjusting shape to match loss dimensions    
        y = (data[:, 1:, :, [2,3,13,14]]-data[:, :-1, :, [2,3,13,14]]) / 255
        y_goal = data[:,1:, :, 11].to(int)
        y_goal = (y_goal == 254) * torch.ones_like(y_goal).to(device) + (y_goal != 254) * y_goal
        tokens, padding_mask = tokenizer(x)
        y_pred, y_disc = model(tokens, padding_mask)
        intention_pred = y_disc.argmax(dim=-1) # B * T * U
        # track prediction
        for j,t in enumerate([0,1,4,8]):
            mask_idx = mask[:,t].flatten().nonzero().to('cpu')
            if mask[:,t].sum() == 0:
                continue
            l2 = ((y_pred[:,t,:,[0,1]]*mask[:,t]-y[:,t,:,[0,1]]*mask[:,t]) ** 2).detach().sum().cpu()
            l1 = torch.abs(y_pred[:,t,:,[0,1]]*mask[:,t]-y[:,t,:,[0,1]]*mask[:,t]).detach().sum().cpu()
            l2_loss_track[j] += l2
            l1_loss_track[j] += l1

            l2 = ((y_pred[:,t,:,[2,3]]*mask[:,t]-y[:,t,:,[2,3]]*mask[:,t]) ** 2).detach().sum().cpu()
            l1 = torch.abs(y_pred[:,t,:,[2,3]]*mask[:,t]-y[:,t,:,[2,3]]*mask[:,t]).detach().sum().cpu()
            l2_loss_intention[j] += l2
            l1_loss_intention[j] += l1
            count[j] += mask[:,t].sum().to('cpu')
            conf_mat[j] += confusion_matrix(y_goal[:,t].cpu().flatten()[mask_idx], intention_pred[:,t].cpu().flatten()[mask_idx], labels=list(range(26)))

    pickle.dump([l2_loss_track, l1_loss_track, l2_loss_intention, l1_loss_intention, count, conf_mat], open("result_indomain_{}.pkl".format(args.pretrained_model),'wb'))


    l2_loss_track = torch.zeros(4)
    l1_loss_track = torch.zeros(4)
    l2_loss_intention = torch.zeros(4)
    l1_loss_intention = torch.zeros(4)
    count = torch.zeros(4)
    mapping = torch.tensor(UNIT_TYPE[2] + UNIT_TYPE[5] + UNIT_TYPE[8]).to(device)
    conf_mat = torch.zeros((4,26,26))
    for i, (data, meta) in tqdm(enumerate(out_test_loader)):
        if i >= 400:
            break
        data = data.to(torch.float32)
        data = data.to(device)

        x = data[:, :-1]
        mask_condition_1 = (x[:, :, :, 1] != 16) 
        mask_condition_2 = torch.isin(x[:, :, :, 0], mapping)

        # Combine both conditions
        mask = mask_condition_1 & mask_condition_2

        # Expand the mask to match the shape of loss targets
        mask = mask.unsqueeze(dim=3)  # Adjusting shape to match loss dimensions    
        y = (data[:, 1:, :, [2,3,13,14]]-data[:, :-1, :, [2,3,13,14]]) / 255
        y_goal = data[:,1:, :, 11].to(int)
        y_goal = (y_goal == 254) * torch.ones_like(y_goal).to(device) + (y_goal != 254) * y_goal
        tokens, padding_mask = tokenizer(x)
        y_pred, y_disc = model(tokens, padding_mask)
        intention_pred = y_disc.argmax(dim=-1) # B * T * U
        # track prediction
        for j,t in enumerate([0,1,4,8]):
            mask_idx = mask[:,t].flatten().nonzero().to('cpu')
            if mask[:,t].sum() == 0:
                continue
            l2 = ((y_pred[:,t,:,[0,1]]*mask[:,t]-y[:,t,:,[0,1]]*mask[:,t]) ** 2).detach().sum().cpu()
            l1 = torch.abs(y_pred[:,t,:,[0,1]]*mask[:,t]-y[:,t,:,[0,1]]*mask[:,t]).detach().sum().cpu()
            l2_loss_track[j] += l2
            l1_loss_track[j] += l1

            l2 = ((y_pred[:,t,:,[2,3]]*mask[:,t]-y[:,t,:,[2,3]]*mask[:,t]) ** 2).detach().sum().cpu()
            l1 = torch.abs(y_pred[:,t,:,[2,3]]*mask[:,t]-y[:,t,:,[2,3]]*mask[:,t]).detach().sum().cpu()
            l2_loss_intention[j] += l2
            l1_loss_intention[j] += l1
            count[j] += mask[:,t].sum().to('cpu')
            conf_mat[j] += confusion_matrix(y_goal[:,t].cpu().flatten()[mask_idx], intention_pred[:,t].cpu().flatten()[mask_idx], labels=list(range(26)))

    pickle.dump([l2_loss_track, l1_loss_track, l2_loss_intention, l1_loss_intention, count, conf_mat], open("result_outdomain_{}.pkl".format(args.pretrained_model),'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='StarCraft-Motion Baseline')
    parser.add_argument('--seed', default=1001, type=int)
    # data
    parser.add_argument('--data', default="/local/scratch/a/bai116/datasets/StarCraftMotion_v0.9/", type=str)
    parser.add_argument('--num_frames', default=10, type=int)
    parser.add_argument('--subsampling_factor', default=24, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    # model
    parser.add_argument('--pretrained_model', type=str)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--spatial', default=0)
    parser.add_argument('--temporal', default=0)
    # # training
    # parser.add_argument('--num_epochs', default=2, type=int)
    # parser.add_argument('--learning_rate', default=1e-3, type=float)


    args = parser.parse_args()
    main(args)