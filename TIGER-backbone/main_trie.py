from dataset import GenRecDataset
from dataloader import GenRecDataLoader
import torch
from transformers import T5ForConditionalGeneration, T5Config
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import argparse
import os
import random
import time
from openpyxl import Workbook, load_workbook
from datetime import datetime
import os
from generation_trie import *

class TIGER(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(TIGER, self).__init__()
        t5config = T5Config(
            num_layers=config['num_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            num_heads=config['num_heads'],
            d_kv=config['d_kv'],
            dropout_rate=config['dropout_rate'],
            vocab_size=config['vocab_size'],
            pad_token_id=config['pad_token_id'],
            eos_token_id=config['eos_token_id'],
            decoder_start_token_id=config['pad_token_id'],
            feed_forward_proj=config['feed_forward_proj'],
        )
        self.model = T5ForConditionalGeneration(t5config)
    
    @property
    def n_parameters(self):
        num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
        total_params = num_params(self.parameters())
        emb_params = num_params(self.model.get_input_embeddings().parameters())
        return f'#Embedding params: {emb_params}, #Total params: {total_params}'

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss, outputs.logits
    
    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,  num_beams: int = 20, **kwargs):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=5, 
            num_beams=num_beams,
            num_return_sequences=num_beams,
            **kwargs
        )

def calculate_metrics(preds, labels, topk_list):
    expanded_labels = labels.unsqueeze(1).expand_as(preds) 
    match = (preds == expanded_labels).all(dim=-1) 
    
    metrics = {}
    
    for k in topk_list:
        match_k = match[:, :k]
        
        hit = (match_k.sum(dim=1) > 0).float() 
        metrics['Recall@' + str(k)] = hit.mean().item()

        ranks = torch.arange(1, k + 1).to(preds.device).float()
        weights = 1.0 / torch.log2(ranks + 1)
        
        dcg = (match_k.float() * weights).sum(dim=1)
        
        metrics['NDCG@' + str(k)] = dcg.mean().item()

    return metrics

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        input_ids = batch['history'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)

        optimizer.zero_grad()
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def validate(model, valid_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch['history'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target'].to(device)

            loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += loss.item()
            
    return total_loss / len(valid_loader)

def evaluate(model, eval_loader, topk_list, beam_size, device, trie=None):
    model.eval()
    accumulated_metrics = {f'Recall@{k}': 0.0 for k in topk_list}
    accumulated_metrics.update({f'NDCG@{k}': 0.0 for k in topk_list})
    total_samples = 0
    
    constraint_fn = None
    if trie is not None:
        constraint_fn = prefix_allowed_tokens_fn(trie)
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['history'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target'].to(device)

            preds = model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                num_beams=beam_size,
                prefix_allowed_tokens_fn=constraint_fn 
            )
            
            preds = preds[:, 1:]
            
            preds = preds.reshape(input_ids.shape[0], beam_size, -1)
            
            batch_metrics = calculate_metrics(preds, labels, topk_list)
            
            batch_size = input_ids.shape[0]
            total_samples += batch_size
            for k, v in batch_metrics.items():
                accumulated_metrics[k] += v * batch_size 

    avg_metrics = {k: v / total_samples for k, v in accumulated_metrics.items()}
    
    avg_recalls = {k: v for k, v in avg_metrics.items() if 'Recall' in k}
    avg_ndcgs = {k: v for k, v in avg_metrics.items() if 'NDCG' in k}

    
    return avg_recalls, avg_ndcgs

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--infer_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=141)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_decoder_layers', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--d_kv', type=int, default=64)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--vocab_size', type=int, default=1025+480)
    parser.add_argument('--pad_token_id', type=int, default=0)
    parser.add_argument('--eos_token_id', type=int, default=0)
    parser.add_argument('--feed_forward_proj', type=str, default='relu')
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--dataset_path', type=str, default='../data/Beauty')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--code_path', type=str, required=True)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--log_path', type=str, default='./logs/tiger_0.7.log')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--save_path', type=str, default='./ckpt/tiger_0.7.pth')
    parser.add_argument('--early_stop', type=int, default=15)
    parser.add_argument('--topk_list', type=list, default=[5,10,20])
    parser.add_argument('--beam_size', type=int, default=20)
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/tiger_60.pth')
    parser.add_argument('--current_epoch', type=int, default=131)
    parser.add_argument('--evaluate_epoch', type=int, default=70)
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--data', type=str, default='_0.6_0.7.parquet')
    parser.add_argument('--desc', type=str, default=None)
    
    config = vars(parser.parse_args())
    config['dataset_path']=f"../data/{config['dataset']}"

    logging.basicConfig(
        filename=config['log_path'],
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    if config['desc'] and config['desc']!='None':
        logging.info(config['desc'])
    logging.info(f"Config: {config}")

    # Model
    model = TIGER(config)
    logging.info(model.n_parameters)

    # Load Checkpoint
    if config['ckpt_path'] and config['ckpt_path']!='None':
        print(f"Loading ckpt: {config['ckpt_path']}")
        ckpt = torch.load(config['ckpt_path'], map_location=config['device'])
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)

    set_seed(config['seed'])
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Datasets
    train_dataset = GenRecDataset(config['dataset_path']+ f"/train{config['data']}", config['code_path'], 'train', config['max_len'])
    validation_dataset = GenRecDataset(config['dataset_path'] + f"/valid{config['data']}", config['code_path'], 'evaluation', config['max_len'])
    test_dataset = GenRecDataset(config['dataset_path'] + f"/test{config['data']}", config['code_path'], 'evaluation', config['max_len'])
    
    print(f"Train: {len(train_dataset)}, Valid: {len(validation_dataset)}, Test: {len(test_dataset)}")
    
    train_dataloader = GenRecDataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    validation_dataloader = GenRecDataLoader(validation_dataset, batch_size=config['infer_size'], shuffle=False)
    test_dataloader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])

    print("Building Trie...")
    all_codes = list(test_dataset.item_to_code.values())
    start_token = config['pad_token_id']
    eos_token = config['eos_token_id']
    
    trie_sequences = []
    for code in all_codes:
        code_list = list(code) if isinstance(code, (np.ndarray, list)) else list(code)
        seq = [start_token] + code_list + [eos_token]
        trie_sequences.append(seq)
        
    item_trie = Trie(trie_sequences)
    print("Trie built.")

    best_loss = 10000.0
    early_stop_counter = 0
    best_epoch = 0

    # === Evaluation Only ===
    if config['type'] == 'test':
        recalls, ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device, trie=item_trie)
        print(f"Test Recalls: {recalls}")
        print(f"Test NDCGs: {ndcgs}")
        logging.info(f"Test Recalls: {recalls}")
        logging.info(f"Test NDCGs: {ndcgs}")

    # === Training ===
    elif config['type'] == 'train':
        for epoch in tqdm(range(config['current_epoch'], config['num_epochs'])):
            logging.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
            
            # Train
            train_loss = train(model, train_dataloader, optimizer, device)
            logging.info(f"Train Loss: {train_loss:.4f}")
            
            # Save Checkpoint (Every Epoch)
            torch.save(model.state_dict(), config['save_path'])
            
            # Evaluate logic
            if epoch >= config['evaluate_epoch']:
                valid_loss = validate(model, validation_dataloader, device)
                logging.info(f"Valid Loss: {valid_loss:.4f}")

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_epoch = epoch
                    early_stop_counter = 0
                    torch.save(model.state_dict(), config['save_path']) # Save Best
                    logging.info(f"Best model saved (Loss: {best_loss:.4f})")
                else:
                    early_stop_counter += 1
                    logging.info(f"Early stop counter: {early_stop_counter}")
                    if early_stop_counter >= config['early_stop']:
                        logging.info("Early stopping triggered.")
                        break
        
        # Load Best and Final Test
        print("Loading best model for final testing...")
        model.load_state_dict(torch.load(config['save_path']))
        recalls, ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device, trie=item_trie)
        logging.info(f"Final Test Recalls: {recalls}")
        logging.info(f"Final Test NDCGs: {ndcgs}")
        print(f"Final Epoch: {best_epoch+1}")
        print(f"Final Test Recalls: {recalls}")
        print(f"Final Test NDCGs: {ndcgs}")
        print(f"Save Path: {config['save_path']}")
        