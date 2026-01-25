import copy
import random
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
import logging
import json
import numpy as np
import pandas as pd


all_prompt = {
    "seqrec": [
        {
            "instruction": "What would user be likely to purchase next after buying items {history} ?",
            "response": "{target}"
        }
    ]
}

class SeqRecDataset(Dataset):
    def __init__(self, args, mode="train",prompt_sample_num=None,sample_num=None):
        super().__init__()
        self.args = args
        self.mode = mode
        self.dataset = args.dataset
        self.data_path = args.data_path 
        self.index_file = args.index_file 
        self.data_file = args.data_file

        self.max_his_len = args.max_his_len
        self.add_prefix = getattr(args, 'add_prefix', False) 
        self.his_sep = getattr(args, 'his_sep', " ")        
        
        self.special_token_for_answer = getattr(args, 'special_token_for_answer', "|start_of_answer|")
        self.tasks = getattr(args, 'tasks', 'seqrec')
        self.prompt_sample_num = getattr(args, 'train_prompt_sample_num', 1)
        
        self.prompt_list = all_prompt.get(self.tasks, [])
        if not self.prompt_list:
            raise ValueError(f"No prompts found for task: {self.tasks}")

        self.indices = self._load_indices()
        
        self.item_to_text = self._build_item_to_text()
        
        self.data = self._load_data()
        
        self.new_tokens = self._extract_new_tokens()

    def _load_indices(self):
        index_path = os.path.join(self.data_path, self.dataset, self.dataset + self.index_file)
        
        print(f"Loading indices from .npy: {index_path}")
        
        indices = np.load(index_path, allow_pickle=True)
        
        prefix_templates = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>", "<f_{}>"]
        convert_indices = {}

        for i in range(indices.shape[0]):
            item_id = str(i + 1) # 假设 Item ID 从 1 开始
            code = indices[i]
            token_list = []
            
            for level, value in enumerate(code):
                if level >= len(prefix_templates):
                    raise ValueError(f"Level ({level+1}) is invalid!")
                
                token = prefix_templates[level].format(int(value))
                token_list.append(token)
            
            convert_indices[item_id] = token_list
            
        return convert_indices

    def _build_item_to_text(self):
        mapping = {}
        for item_id, tokens in self.indices.items():
            mapping[str(item_id)] = "".join(tokens)
        return mapping

    def _extract_new_tokens(self):
        new_tokens = set()
        for tokens in self.indices.values():
            for t in tokens:
                new_tokens.add(t)
        return sorted(list(new_tokens))

    def _load_data(self):
        path = os.path.join(self.data_path, self.dataset, f"{self.mode}{self.data_file}")
        print(f"Loading data from: {path}")
        try:
            df = pd.read_parquet(path)
            if 'history' not in df.columns or 'target' not in df.columns:
                 raise ValueError(f"Parquet file must contain 'history' and 'target' columns.")
            return df
        except Exception as e:
            print(f"Error loading parquet file: {e}")
            return pd.DataFrame()

    def get_new_tokens(self):
        return self.new_tokens

    def __len__(self):
        return len(self.data)

    def get_all_items(self):
        all_items = set()
        for item_id, token_list in self.indices.items():
            all_items.add("".join(token_list))

        return all_items

    def __getitem__(self, index):
        row = self.data.iloc[index]
        
        history_ids = row['history']
        target_id = row['target']
        
        history_tokens = []
        for iid in history_ids:
            if str(iid) in self.item_to_text:
                history_tokens.append(self.item_to_text[str(iid)])
        
        target_token = self.item_to_text.get(str(target_id), "")
        
        if self.max_his_len > 0:
            history_tokens = history_tokens[-self.max_his_len:]
        
        if self.add_prefix:
            history_str_list = [
                str(k + 1) + ". " + item_str 
                for k, item_str in enumerate(history_tokens)
            ]
            history_str = self.his_sep.join(history_str_list)
        else:
            history_str = self.his_sep.join(history_tokens)

        prompt_idx = 0
        current_prompt = self.prompt_list[prompt_idx]
        
        data_dict = {
            "history": history_str,
            "dataset": self.dataset,
            "target": target_token
        }
        
        # Input (Instruction)
        input_text = current_prompt['instruction'].format(**data_dict)+self.special_token_for_answer
        
        response_text = current_prompt['response'].format(**data_dict)
        
        return {
            "input_ids": input_text,
            "labels": response_text
        }


class BaseDataset(Dataset):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)
        self.phase=args.phase
        self.max_his_len = args.max_his_len
        self.his_sep = args.his_sep
        self.index_file = args.index_file
        self.add_prefix = args.add_prefix
        self.ft = args.ft
        self.post = args.post
        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None

    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)

    def get_new_tokens(self):

        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))
        if self.args.special_token_for_answer:
            self.new_tokens.append("|start_of_answer|")
        return self.new_tokens

    def get_all_items(self):


        if self.all_items is not None:
            return self.all_items
        warm_items = np.load(os.path.join(self.data_path,"phase%s"%(self.phase), "warm_item.npy"), allow_pickle=True).tolist()
        cold_items = np.load(os.path.join(self.data_path,"phase%s"%(self.phase), "cold_item.npy"), allow_pickle=True).tolist()
        self.all_items = set()
        for i_id in warm_items:
            self.all_items.add("".join(self.indices[str(i_id)]))
        for i_id in cold_items:
            self.all_items.add("".join(self.indices[str(i_id)]))

        return self.all_items
    
    def get_warm_items(self):
        self.warm_items = set()
        warm_items = np.load(os.path.join(self.data_path,"phase%s"%(self.phase), "warm_item.npy"), allow_pickle=True).tolist()
        self.warm_ids = warm_items
        for i_id in warm_items:
            self.warm_items.add("".join(self.indices[str(i_id)]))
        return self.warm_items


    def get_cold_items(self):

        self.cold_items = set()
        cold_items = np.load(os.path.join(self.data_path,"phase%s"%(self.phase), "cold_item.npy"), allow_pickle=True).tolist()
        self.cold_ids = cold_items
        for i_id in cold_items:
            self.cold_items.add("".join(self.indices[str(i_id)]))
        return self.cold_items

    def get_prefix_allowed_tokens_fn(self, tokenizer):
        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            for index in self.indices.values():
                for i, token in enumerate(index):
                    token_id = tokenizer(token)["input_ids"][0]
                    if i not in self.allowed_tokens.keys():
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)
            self.allowed_tokens[len(self.allowed_tokens.keys())] = set([tokenizer.eos_token_id])
        sep = [0]

        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            reversed_sent = sentence[::-1]
            for i in range(len(reversed_sent)):
                if reversed_sent[i:i + len(sep)] == sep[::-1]:
                    # print(list(self.allowed_tokens[i]))
                    return list(self.allowed_tokens[i])

        return prefix_allowed_tokens_fn

    def _process_data(self):

        raise NotImplementedError




class ItemFeatDataset(BaseDataset):

    def __init__(self, args, task="item2index", prompt_sample_num=1, sample_num=-1):
        super().__init__(args)

        self.task = task.lower()
        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num

        # self.prompts = all_prompt[self.task]
        self.prompt = {}
        self.prompt["instruction"] = "Which item has the title: \"{title}\"?"
        self.prompt["response"] = "{item}"
        # load data
        self.phase=args.phase
        self.post = args.post
        self.ft=args.ft
        self._load_data()
        self.special_token_for_answer = args.special_token_for_answer
        self.feat_data = self._process_data()




    def _load_data(self):

        if self.post and  "warmtrain" in self.post:
            print("warmtrain")
            with open(os.path.join(self.data_path, "phase%s"%(0),self.dataset + self.index_file), 'r') as f:
                self.indices = json.load(f)
        else:
            with open(os.path.join(self.data_path, "phase%s"%(self.phase),self.dataset + self.index_file), 'r') as f:
                self.indices = json.load(f)
    
        with open(os.path.join(self.data_path, self.dataset + ".item.json"), 'r') as f:
            self.item_feat = json.load(f)


    def _process_data(self):

        feat_data = []
        if self.ft:
            train_items = np.load(os.path.join(self.data_path,"phase%s"%(self.phase-1), "cold_item.npy"), allow_pickle=True).tolist()
        else:
            train_items = np.load(os.path.join(self.data_path,"phase%s"%(self.phase), "warm_item.npy"), allow_pickle=True).tolist()
        for iid in self.item_feat:
            if int(iid) not in train_items:
                continue
            feat = self.item_feat[iid]
            index = "".join(self.indices[iid])
            feat["item"] = index
            feat["title"] = feat["title"].strip().strip(".!?,;:`")
            feat_data.append(feat)

        return feat_data


    def __len__(self):
        return len(self.feat_data) * self.prompt_sample_num

    def _get_text_data(self, data):

        instruction = self.prompt["instruction"].format(**data)+self.special_token_for_answer
        response = self.prompt["response"].format(**data)

        return instruction,response 

    def __getitem__(self, index):

        idx = index
        d = self.feat_data[idx]

        input, output = self._get_text_data(d)
        return dict(input_ids=input, labels=output)



class FusionSeqRecDataset(BaseDataset):
        
    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)
        self.items=None
        self.mode = mode
        self.prompt_id = prompt_id
        self.prompt = "What would user be likely to purchase next after buying items {history} ?"
        self.special_token_for_answer = args.special_token_for_answer
        self.sample_num = sample_num


        # load data
        self._load_data()

        self._remap_items()
        
        # load data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.inter_data = self._process_valid_data()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        elif self.mode == "test_warm":
            self.warm_items = self.get_warm_items()
            self.inter_data = self._process_test_warm_data()
        elif self.mode == "test_cold":
            self.cold_items = self.get_cold_items()
            self.inter_data = self._process_test_cold_data()
        elif self.mode == "test_old_warm":
            self.old_warm_items,self.new_warm_items=self.get_newold_warm_items()
            self.inter_data = self._process_test_old_warm_data()

        elif self.mode == "test_new_warm":
            self.old_warm_items,self.new_warm_items=self.get_newold_warm_items()
            self.inter_data = self._process_test_new_warm_data()
        else:
            raise NotImplementedError


    def _load_data(self):
        self.warm_data = {}
        self.train_data = {}

        for phase_idx in range(self.phase):
            
            train_data = np.load(os.path.join(self.data_path,"phase%s"%(phase_idx), "training_dict.npy"), allow_pickle=True).item()
            for uid in train_data:
                if uid not in self.warm_data:
                    self.warm_data[uid] = []
                self.warm_data[uid].extend(train_data[uid])

            valid_data = np.load(os.path.join(self.data_path,"phase%s"%(phase_idx), "validation_dict.npy"), allow_pickle=True).item()
            for uid in valid_data:
                if uid not in self.warm_data:
                    self.warm_data[uid] = []
                if len(valid_data[uid]):
                    self.warm_data[uid].extend(valid_data[uid])
    
        train_data = np.load(os.path.join(self.data_path,"phase%s"%(self.phase), "training_dict.npy"), allow_pickle=True).item()
        for uid in train_data:
            if uid not in self.train_data:
                self.train_data[uid] = []
            self.train_data[uid].extend(train_data[uid])
        if self.phase==0:
            for uid in self.train_data:
                self.warm_data[uid] = []
        if not self.ft:
            for uid in self.warm_data:
                self.train_data[uid] = self.warm_data[uid]+self.train_data[uid]
                self.warm_data[uid] = []
        cnt =0
        for uid in self.train_data:
            for iid in self.train_data[uid]:
                cnt+=1
        print(f"train {cnt}")

        cnt =0
        for uid in self.warm_data:
            for iid in self.warm_data[uid]:
                cnt+=1
        print(f"warm {cnt}")
            
        self.valid_data = np.load(os.path.join(self.data_path,"phase%s"%(self.phase), "validation_dict.npy"), allow_pickle=True).item()
        
        for uid in self.valid_data:
            for iid in self.valid_data[uid]:
                cnt+=1
        print(f"valid {cnt}")
        # exit()
        
        self.test_data = np.load(os.path.join(self.data_path,"phase%s"%(self.phase), "testing_dict.npy"), allow_pickle=True).item()
        
        if self.post and  "warmtrain" in self.post:
            print("warmtrain")
            with open(os.path.join(self.data_path, "phase%s"%(0),self.dataset + self.index_file), 'r') as f:
                self.indices = json.load(f)
        else:
            with open(os.path.join(self.data_path, "phase%s"%(self.phase),self.dataset + self.index_file), 'r') as f:
                self.indices = json.load(f)


    def _remap_items(self):

        self.remapped_warm = dict()
        for uid, items in self.warm_data.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_warm[uid] = new_items

        self.remapped_train = dict()
        for uid, items in self.train_data.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_train[uid] = new_items
        
        self.remapped_valid = dict()
        for uid, items in self.valid_data.items():
            new_items = ["".join(self.indices[str(i)]) for i in items] if len(items) else []
            self.remapped_valid[uid] = new_items

        self.remapped_test = dict()
        for uid, items in self.test_data.items():
            new_items = ["".join(self.indices[str(i)]) for i in items] if len(items) else []
            self.remapped_test[uid] = new_items

    def _process_train_data(self):

        inter_data = []
        for uid in self.remapped_train:
            warm_items = self.remapped_warm[uid]
            items = self.remapped_train[uid]# input of each training sample
            if len(items)>1: # a training user should at least have two interactions
                if self.args.subseq:
                    for i in range(0, len(items)):
                        one_data = dict()
                        if len(warm_items)>=1 or i>=1:
                            
                            one_data["item"] = items[i]
                            history = warm_items+items[:i]
                            if self.max_his_len > 0:
                                history = history[-self.max_his_len:]
                            if self.add_prefix:
                                history = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(history)]
                            one_data["inters"] = self.prompt.format(history=",".join(history))+ self.special_token_for_answer
                            inter_data.append(one_data)
                            #inter_data.append(one_data)
                else:
                    
                    one_data = dict()
                    one_data["item"] = items[-1]
                    history = warm_items+items[:-1]
                    if self.max_his_len > 0:
                        history = history[-self.max_his_len:]
                    if self.add_prefix:
                        history = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(history)]
                    one_data["inters"] = self.prompt.format(history=",".join(history)) + self.special_token_for_answer
                    inter_data.append(one_data)

        return inter_data
    
    def _process_valid_data(self):

        inter_data = []
        for uid in self.remapped_valid:
            items = self.remapped_valid[uid]
            train_items = self.remapped_train[uid]
            warm_items = self.remapped_warm[uid]
            if len(items):
                one_data = dict()
                # one_data["user"] = uid
                one_data["item"] = items[0]
                history = warm_items+train_items
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
                one_data["inters"] = self.prompt.format(history=",".join(history)) + self.special_token_for_answer
                inter_data.append(one_data)

        return inter_data

    def _process_test_data(self):

        inter_data = []
        for uid in self.remapped_test:
            items = self.remapped_test[uid]
            train_items = self.remapped_train[uid]
            warm_items = self.remapped_warm[uid]
            valid_items = self.remapped_valid[uid]
            if len(items):
                one_data = dict()
                one_data["item"] = items
                history = warm_items+train_items+valid_items
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
                one_data["inters"] = self.prompt.format(dataset=self.dataset,history=",".join(history)) + self.special_token_for_answer
                inter_data.append(one_data)

        if self.sample_num > 0:
            inter_data = inter_data[:self.sample_num]

        return inter_data
    

    def _process_test_warm_data(self):
        warm_cnt = 0
        inter_data = []
        for uid in self.remapped_test:
            items = self.remapped_test[uid]
            train_items = self.remapped_train[uid]
            warm_items = self.remapped_warm[uid]
            valid_items = self.remapped_valid[uid]
            ids = self.test_data[uid]
            if len(items):
                one_data = dict()
                gold = []
                for iid,item in zip(ids,items):
                    if iid in self.warm_ids:
                        gold.append(item)
                        warm_cnt += 1
                one_data["item"] = gold
                history = warm_items+train_items + valid_items
                if len(gold):
                    if self.max_his_len > 0:
                        history = history[-self.max_his_len:]
                    if self.add_prefix:
                        history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
                    one_data["inters"] = self.prompt.format(dataset=self.dataset,history=",".join(history)) + self.special_token_for_answer
                    inter_data.append(one_data)

        if self.sample_num > 0:
            inter_data = inter_data[:self.sample_num]
        print("warm interaction in test:", warm_cnt)
        return inter_data
    

    def _process_test_cold_data(self):
        inter_data = []
        cold_cnt = 0
        for uid in self.remapped_test:
            items = self.remapped_test[uid]
            train_items = self.remapped_train[uid]
            warm_items = self.remapped_warm[uid]
            valid_items = self.remapped_valid[uid]
            ids = self.test_data[uid]
            if len(items):
                one_data = dict()
                gold = []
                for iid,item in zip(ids,items):
                    if iid in self.cold_ids:
                        gold.append(item)
                        warm_cnt += 1
                one_data["item"] = gold
                history = warm_items+train_items + valid_items
                if len(gold):
                    if self.max_his_len > 0:
                        history = history[-self.max_his_len:]
                    if self.add_prefix:
                        history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
                    one_data["inters"] = self.prompt.format(dataset=self.dataset,history=",".join(history)) + self.special_token_for_answer
                    inter_data.append(one_data)

        if self.sample_num > 0:
            inter_data = inter_data[:self.sample_num]
        print("cold interaction in test:", cold_cnt)
        return inter_data
    
    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

    def __len__(self):
        return len(self.inter_data)

    def __getitem__(self, index):
        d = self.inter_data[index]
        return dict(input_ids=d["inters"], labels=d["item"])