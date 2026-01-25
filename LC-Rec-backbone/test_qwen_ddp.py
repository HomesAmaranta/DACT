import argparse
import json
import os
import sys
from typing import List, Dict
import pandas as pd
import torch
import pickle
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(gpu_name)

import transformers
from peft import PeftModel 
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Qwen2Tokenizer, Qwen2Config, Qwen2ForCausalLM, BitsAndBytesConfig
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

from test_qwen import Trie, prefix_allowed_tokens_fn, get_greedy_prefix_allowed_tokens_fn

def test(args):

    set_seed(args.seed)
    args.ckpt_path = find_path(args.ckpt_path)
    print(vars(args))

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        print(vars(args))

    dist.init_process_group(backend="nccl", world_size=world_size, rank=local_rank)

    device_map = {"": local_rank}
    device = torch.device("cuda",local_rank)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.ckpt_path)
    tokenizer.padding_side = "left"

    load_8bit = True 
    # load_8bit = False
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
        )  
        args.ckpt_path = False  
        if os.path.exists(checkpoint_name):
        
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name, map_location="cpu")
            set_peft_model_state_dict(model, adapters_weights)
            del adapters_weights
        else:
            if local_rank == 0:
                print(f"Checkpoint {checkpoint_name} not found")
                
    model = DistributedDataParallel(model, device_ids=[local_rank])
    model.eval()
    prompt_ids = [0]

    test_data = load_test_dataset(args)
    if args.subset_test:
        args.sample_num = 200
        test_data = load_test_dataset(args)
    ddp_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=local_rank, shuffle=False, drop_last=False)

    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()

    col_dict = {}
    indices = test_data.indices
    for item_id, token_list in indices.items():
        token_str = "".join(token_list)
        if token_str not in col_dict:
            col_dict[token_str] = 0
        col_dict[token_str] += 1

    all_len = len(indices)              
    unique_codes = len(col_dict)       
    collision_rate = (all_len - unique_codes) / all_len
    print("col rate",len(col_dict),(all_len-len(col_dict))/all_len)

    candidate_trie = Trie(
            [
                [1] + 
                tokenizer.encode(candidate)
                + [tokenizer.eos_token_id]
                for candidate in all_items
            ]
        )
    prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie, tokenizer)


    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                              num_workers=2, pin_memory=True,shuffle=False)#,

    if local_rank == 0:
        print("data num:", len(test_data))
        

    import time 
    # all performance
    with torch.no_grad(): 
        for prompt_id in prompt_ids:

            total = 0

            all_pred_list = []
            all_gold_list = []
            
            st_all = time.time()
            local_all_pred_list = []
            local_all_gold_list = []
            for step, batch in enumerate(tqdm(test_loader)):
           

                inputs = batch[0].to(device)
                targets = batch[1]
                if step%world_size!=local_rank:
                    continue
                total += len(targets)
                output = model.module.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=args.max_new_token, 
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
                                            all_items=all_items)
                local_all_pred_list.extend(topk_res)
                local_all_gold_list.extend(targets)
            dist.barrier()
            res_gather_list = [None for _ in range(world_size)]
            dist.all_gather_object(obj=local_all_pred_list, object_list=res_gather_list)
            target_gather_list = [None for _ in range(world_size)]
            dist.all_gather_object(obj=local_all_gold_list, object_list=target_gather_list)
            
            if local_rank == 0:
                for ga_res in res_gather_list:
                    all_pred_list.extend(ga_res)

                for ga_tar in target_gather_list:
                    all_gold_list.extend(ga_tar)
            if local_rank == 0:
                print("=== End ===%s"%(local_rank))
                test_results = get_metrics_results(all_pred_list, ["hit@5","hit@10","hit@20","ndcg@5","ndcg@10","ndcg@20"])
                print(test_results)
                
        dist.barrier()
        
        if local_rank == 0:
            print("=== End ===")
            print("=== All performance")
            print(f"All time costs: {round(time.time()-st_all, 2)}s")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_test_args(parser)
    parser = parse_dataset_args(parser)
    parser.add_argument("--data_file", type=str, default=None)
    
    args = parser.parse_args()

    args = parser.parse_args()

    test(args)