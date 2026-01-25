import torch
import copy
import argparse
from dataclasses import dataclass

import transformers
import math
from torch.utils.data import Sampler
import torch.distributed as dist
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, T5Tokenizer, T5Config, T5ForConditionalGeneration


class Collator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        # print(self.tokenizer.model_max_length)

    def __call__(self, batch):
        
        input_texts = [d["input_ids"] for d in batch]
        label_texts = [d["labels"] for d in batch]

        inputs = self.tokenizer(input_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)

        labels = self.tokenizer(label_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)
        inputs['labels'] = labels['input_ids']
        inputs['labels'][inputs['labels'] == self.tokenizer.pad_token_id] = -100

        return inputs

class Collator_DecoderOnly_manual(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        print("*** only train response:",  self.only_train_response)
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id


    def __call__(self, batch):

        if not self.only_train_response:

            full_texts = [d["input_ids"] + d["labels"] + self.tokenizer.eos_token for d in batch]
            inputs = self.tokenizer(
                text = full_texts,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_attention_mask=True,
            )
            labels = copy.deepcopy(inputs["input_ids"])
            inputs["labels"] = labels
        else:

            input_texts = [d["input_ids"] for d in batch]
            full_texts = [d["input_ids"] + d["labels"] + self.tokenizer.eos_token for d in batch]

            inputs = self.tokenizer(
                text = full_texts,
                text_target = input_texts,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_attention_mask=True,
            )
            labels = copy.deepcopy(inputs["input_ids"])


            labels[labels == self.tokenizer.pad_token_id] = -100

            labels[torch.where(inputs["labels"] != self.tokenizer.pad_token_id)] = -100


            inputs["labels"] = labels
            
        return inputs

class Collator_DecoderOnly(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

            
    def __call__(self, batch):

        input_texts = [d["input_ids"] + d["labels"]+ self.tokenizer.eos_token for d in batch]

        inputs = self.tokenizer(input_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=512,
                                truncation=True,
                                return_attention_mask=True)

        inputs['labels'] = copy.deepcopy(inputs["input_ids"])
        
        return inputs

class TestCollator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        targets = [d["labels"] for d in batch]

        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        return (inputs, targets)

