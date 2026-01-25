import collections
import json
import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae_dact import RQVAE
import argparse
import torch.nn.functional as F
import os

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

# Attention: Remember to change the dataset name and checkpoint path
# if you want to generate codes for other datasets.
def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=3000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )
    parser.add_argument('--eval_step', type=int, default=50, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument('--lr_scheduler_type', type=str, default="linear", help='scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epochs')
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.003], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")

    parser.add_argument("--device", type=str, default="cuda", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument('--layers', type=int, nargs='+', default=[512,256,128,64], help='hidden sizes of every layer')
    parser.add_argument('--save_limit', type=int, default=5, help='save limit for ckpt')
    
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt/Beauty", help="please specify output directory for model")

    parser.add_argument("--rqvae_path", type=str, default="./ckpt/Beauty/Nov-04-2025_07-00-54/best_collision_model.pth", help="")
    parser.add_argument("--old_rqvae_path", type=str, default="./ckpt/Beauty/Nov-04-2025_07-00-54/best_collision_model.pth", help="")
    parser.add_argument("--data_path", type=str, default="../data/Beauty/item_emb_0.7.parquet", help="Input data path.")
    parser.add_argument("--output_file", type=str, default="../data/Beauty/Beauty_t5_rqvae_0.7.npy", help="Input data path.")
    parser.add_argument("--cf_path", type=str, default="../data/Beauty/Beauty_t5_rqvae_0.7.npy", help="Input data path.")
    parser.add_argument("--old_code", type=str, default="../data/Beauty/Beauty_t5_rqvae_0.7.npy", help="Input data path.")
    parser.add_argument("--first_only", type=int, default=0, help="")

    return parser.parse_args()

def load_model(args, data, ckpt_path):
    model = RQVAE(in_dim=data.dim,
                    num_emb_list=args.num_emb_list,
                    e_dim=args.e_dim,
                    layers=args.layers,
                    dropout_prob=args.dropout_prob,
                    bn=args.bn,
                    loss_type=args.loss_type,
                    quant_loss_weight=args.quant_loss_weight,
                    kmeans_init=args.kmeans_init,
                    kmeans_iters=args.kmeans_iters,
                    sk_epsilons=args.sk_epsilons,
                    sk_iters=args.sk_iters,
                    )

    model.load_ckpt(ckpt_path)
    model = model.to(device)
    model.eval()
    return model

if __name__=="__main__":
    args=parse_args()
    ckpt_path = args.rqvae_path
    output_file = args.output_file
    device = torch.device("cuda")

    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    state_dict = ckpt["state_dict"]


    data = EmbDataset(args.data_path)

    model = load_model(args, data, ckpt_path)

    data_loader = DataLoader(data,num_workers=args.num_workers,
                                batch_size=args.batch_size, shuffle=False,
                                pin_memory=True)

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]
    old_codes_array = np.load(args.old_code)
    for d in tqdm(data_loader):
        d, emb_idx=d[0],d[1]
        d = d.to(device)
        _,_,indices,_,_,_,_,_,_ = model(d)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))

            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)
  
    print("All indices number: ",len(all_indices))
    print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    print("Collision Rate",(tot_item-tot_indice)/tot_item)


    all_indices_dict = {}
    for item, indices in enumerate(all_indices.tolist()):
        all_indices_dict[item] = list(indices)

    codes = []
    for key, value in all_indices_dict.items():
        code = [int(item.split('_')[1].strip('>')) for item in value]
        codes.append(code)

    codes_array = np.array(codes)
    codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))

    # Resolve duplicates by incrementing the last dimension
    unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = unique_codes[counts > 1]

    if len(duplicates) > 0:
        print("Resolving duplicates in codes...")
        for duplicate in duplicates:
            duplicate_indices = np.where((codes_array == duplicate).all(axis=1))[0]
            for i, idx in enumerate(duplicate_indices):
                codes_array[idx, -1] = i  # Increment the last digit for resolving duplicates

    new_unique_codes, new_counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = new_unique_codes[new_counts > 1]

    if len(duplicates) > 0:
        print("There still have duplicates:", duplicates)
    else:
        print("There are no duplicates in the codes after resolution.")

    final_codes_array = codes_array.copy()
    if args.first_only==1:
        print(f"Loading old codes from {args.old_code} for stability...")
        
        if len(old_codes_array)<len(codes_array):
            codes_array = codes_array[:len(old_codes_array)]
        for i in range(len(codes_array)):
            if codes_array[i][0] == old_codes_array[i][0]:
                final_codes_array[i] = old_codes_array[i]
        print("Code stabilization completed: reused old codes where first layer matched.")

    change_num=0
    for i in range(len(old_codes_array)):
        if (final_codes_array[i] != old_codes_array[i]).any():
            change_num+=1
    print(f"Change num is {change_num}, while total num is {len(final_codes_array)}，with a ratio of{change_num/len(final_codes_array)*100}%.")
    np.save(output_file, final_codes_array)
    print(f"Saving codes to {output_file}")