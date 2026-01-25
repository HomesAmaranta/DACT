import argparse
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import sys
from typing import List, Dict

import torch

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(gpu_name)

import transformers
from peft import PeftModel 
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Qwen2Tokenizer, Qwen2Config, Qwen2ForCausalLM, GenerationConfig
from utils import *
from collator import TestCollator
from evaluate import get_topk_results, get_metrics_results
from generation_trie import Trie

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)


class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)


def prefix_allowed_tokens_fn(candidate_trie, tokenizer, special_token_for_answer="|start_of_answer|"):

    sep = tokenizer(special_token_for_answer)["input_ids"]
    bos = [1]
    def prefix_allowed_tokens(batch_id, sentence):
        for i in range(len(sentence),-1,-1):
            if sentence[i-len(sep):i].tolist() == sep:
                if i == len(sentence):
                    sentence_ = bos
                else:
                    sentence_ = [1] + sentence[i:].tolist()
        trie_out = candidate_trie.get(sentence_)
        return trie_out

    return prefix_allowed_tokens


def get_greedy_prefix_allowed_tokens_fn(indices, tokenizer, special_token_for_answer="|start_of_answer|"):

    allowed_tokens = {}
    for index in indices.values(): # 
        for i, token in enumerate(index): # 
            token_id = tokenizer(token)["input_ids"][0] # 
            if i not in allowed_tokens.keys(): # 
                allowed_tokens[i] = set() # 
            allowed_tokens[i].add(token_id) # 
    allowed_tokens[len(allowed_tokens.keys())] = set([tokenizer.eos_token_id]) #

    sep = tokenizer(special_token_for_answer)["input_ids"][1:]
    def prefix_allowed_tokens_fn(batch_id, sentence):
        sentence = sentence.tolist()
        reversed_sent = sentence[::-1]
        for i in range(len(reversed_sent)):
            if reversed_sent[i:i + len(sep)] == sep[::-1]:
                try:
                    return list(allowed_tokens[i])
                except:
                     print("problem here")

    return prefix_allowed_tokens_fn

def get_topk_results(predictions, scores, targets, k, all_items=None, special_token_for_answer="|start_of_answer|"):
    results = []
    B = len(targets)
    predictions = [_.split(special_token_for_answer)[-1] for _ in predictions]
    if all_items is not None:
        cnt = 0
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000
                cnt += 1
    batch_pred = []
    breakpoint()
    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]

        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        results = sorted(pairs, key=lambda x: x[1], reverse=True)
        pred = [r[0] for r in results]
        batch_pred.append(pred)
    return batch_pred



def test(args):

    set_seed(args.seed)
    print(vars(args))

    device_map = {"": args.gpu_id}
    device = torch.device("cuda",args.gpu_id)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.ckpt_path)
    tokenizer.padding_side = "left"

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    load_8bit = True 
    dtype = torch.bfloat16 
    bf16 = True 


    if not args.lora:
        args.base_model = args.ckpt_path

    model = Qwen2ForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        load_in_8bit=load_8bit,
        device_map=device_map,
    )
    model.resize_token_embeddings(len(tokenizer))

    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        modules_to_save=args.lora_modules_to_save.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)

    if args.ckpt_path:
        checkpoint_name = os.path.join(
            args.ckpt_path, "adapter_model.bin"
        )  # only LoRA model - LoRA config above has to fit
        args.ckpt_path = False  # So the trainer won't try loading its state
        if os.path.exists(checkpoint_name):
        
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            if local_rank == 0:
                print(f"Checkpoint {checkpoint_name} not found")

    print("** model loaded")
    model.eval()

    prompt_ids = [0]

    test_data = load_test_dataset(args)
    if args.subset_test:
        args.sample_num = 200 #2000
        test_data = load_test_dataset(args)

    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()

    if args.greedy_trie:
        prefix_allowed_tokens = get_greedy_prefix_allowed_tokens_fn(test_data.indices,tokenizer)
    else:
        candidate_trie = Trie(
            [
                [1] + 
                tokenizer.encode(candidate)
                + [tokenizer.eos_token_id]
                for candidate in all_items
            ]
        )
        prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie, tokenizer, args.special_token_for_answer)


    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                             shuffle=False, num_workers=0, pin_memory=True)

    print("data num:", len(test_data))

    import time 
    
    # all performance
    with torch.no_grad():
        for prompt_id in prompt_ids:

            test_loader.dataset.set_prompt(prompt_id)
            total = 0

            all_pred_list = []
            all_gold_list = []
            st_all = time.time()
            for step, batch in enumerate(tqdm(test_loader)):
                
                inputs = batch[0].to(device)
                targets = batch[1]
                if step<=1:
                    print(inputs)
                total += len(targets)
                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=args.max_new_token, #10
                    prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                    do_sample=False
                )
                
                output_ids = output["sequences"]
                scores = output["sequences_scores"]
                output = tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )

                topk_res = get_topk_results(output,scores,targets,args.num_beams,
                                            all_items=all_items if args.filter_items else None)
                if step<=1:
                    print(topk_res)

                all_pred_list.extend(topk_res)
                all_gold_list.extend(targets)

            test_results = computeTopNAccuracy(all_gold_list, all_pred_list, topN=[5, 10, 20])
            print("=== End ===")
            print("=== All performance")
            print_results(None, None, test_results)
            print(f"All time costs: {round(time.time()-st_all, 2)}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_train_args(parser)
    parser = parse_test_args(parser)
    # parser = parse_llama_args(parser)

    args = parser.parse_args()
    test(args)
