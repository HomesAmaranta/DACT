import json
import logging
import os
import re
import random
import datetime

import math

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from data import SeqRecDataset,ItemFeatDataset


def parse_global_args(parser):


    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base_model", type=str, default="t5-small",help="basic model path")
    parser.add_argument("--ft", type=int, default=0, help="Random seed")
    parser.add_argument("--post", type=str, default=None, help="Random seed")
    parser.add_argument("--phase", type=str, default=None,help="phase path")
    parser.add_argument("--output_dir", type=str, default="/ckpt/games/", help="The output directory")
    parser.add_argument("--verbose", action="store_true", help='whether or not print the time costs')
    return parser

def parse_dataset_args(parser):
    parser.add_argument("--data_path", type=str, default="../../data/", help="data directory")
    parser.add_argument("--tasks", type=str, default="seqrec", help="Downstream tasks, separate by comma")
    parser.add_argument("--dataset", type=str, default="beauty", help="Dataset name")
    parser.add_argument("--index_file", type=str, default=".llamaindex-sk4-sk.json", help="the item indices file")

    # arguments related to sequential task
    parser.add_argument("--max_his_len", type=int, default=20,
                        help="the max number of items in history sequence, -1 means no limit")
    parser.add_argument("--add_prefix", action="store_true", default=False,
                        help="whether add sequential prefix in history")
    parser.add_argument("--his_sep", type=str, default=", ", help="The separator used for history")
    parser.add_argument("--only_train_response", action="store_true", default=False,
                        help="whether only train on responses")

    parser.add_argument("--train_prompt_sample_num", type=str, default="1",
                        help="the number of sampling prompts for each task")
    parser.add_argument("--train_data_sample_num", type=str, default="-1",
                        help="the number of sampling prompts for each task")

    # arguments related for evaluation
    parser.add_argument("--valid_prompt_id", type=int, default=0,
                        help="The prompt used for validation")
    parser.add_argument("--sample_valid", action="store_true", default=True,
                        help="use sampled prompt for validation")
    parser.add_argument("--valid_prompt_sample_num", type=int, default=2,
                        help="the number of sampling validation sequential recommendation prompts")

    return parser

def parse_train_args(parser):

    parser.add_argument("--optim", type=str, default="adamw_torch", help='The name of the optimizer')
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--logging_step", type=int, default=2)
    parser.add_argument("--model_max_length", type=int, default=1024)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--ft_ckpt_path", type=str,
                        default=None,
                        help="The checkpoint path")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj", help="separate by comma")
    parser.add_argument("--lora_modules_to_save", type=str,
                        default="embed_tokens,lm_head", help="separate by comma")
    
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="either training checkpoint or final adapter")

    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--save_and_eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_and_eval_steps", type=int, default=200)
    parser.add_argument("--fp16",  action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--deepspeed", type=str, default="./config/ds_z3_bf16.json")

    parser.add_argument("--subseq", action="store_true")
    return parser

def parse_test_args(parser):

    parser.add_argument("--ckpt_path", type=str,
                        default="/./ckpt/games/",
                        help="The checkpoint path")
    parser.add_argument("--filter_items", action="store_true", default=False,
                        help="whether filter illegal items")
    parser.add_argument("--greedy_trie", action="store_true")
    parser.add_argument("--results_file", type=str,
                        default="./results/test-ddp.json",
                        help="result output path")

    parser.add_argument("--test_batch_size", type=int, default=5)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--sample_num", type=int, default=-1,
                        help="test sample number, -1 represents using all test data")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID when testing with single GPU")
    parser.add_argument("--test_prompt_ids", type=str, default="0",
                        help="test prompt ids, separate by comma. 'all' represents using all")
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
                        help="test metrics, separate by comma")
    parser.add_argument("--test_task", type=str, default="SeqRec")
    parser.add_argument("--max_new_token", type=int, default=10)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--subset_test", action="store_true")

    parser.add_argument("--eval_after_train", action="store_true")
    parser.add_argument("--special_token_for_answer", default="|start_of_answer|")

    return parser


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def ensure_dir(dir_path):

    os.makedirs(dir_path, exist_ok=True)


def load_datasets(args):

    tasks = args.tasks.split(",")

    train_prompt_sample_num = [int(_) for _ in args.train_prompt_sample_num.split(",")]
    assert len(tasks) == len(train_prompt_sample_num), "prompt sample number does not match task number"
    train_data_sample_num = [int(_) for _ in args.train_data_sample_num.split(",")]
    assert len(tasks) == len(train_data_sample_num), "data sample number does not match task number"

    train_datasets = []
    for task, prompt_sample_num,data_sample_num in zip(tasks,train_prompt_sample_num,train_data_sample_num):
        if task.lower() == "seqrec":
            dataset = SeqRecDataset(args, mode="train")
            #item_dataset = ItemFeatDataset(args, prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)
        else:
            raise NotImplementedError
        train_datasets.append(dataset)
        #train_datasets.append(dataset)

    train_data = ConcatDataset(train_datasets)

    valid_data = SeqRecDataset(args,"valid",args.valid_prompt_sample_num)

    return train_data, valid_data

def load_test_dataset(args):

    if args.test_task.lower() == "seqrec":
        test_data = SeqRecDataset(args, mode="test", sample_num=args.sample_num)
        # test_data = SeqRecTestDataset(args, sample_num=args.sample_num)
    else:
        raise NotImplementedError

    return test_data

def load_test_warm_dataset(args):
    if args.test_task.lower() == "seqrec":
        test_data = SeqRecDataset(args, mode="test_warm", sample_num=args.sample_num)
    else:
        raise NotImplementedError
    return test_data

def load_test_cold_dataset(args):
    if args.test_task.lower() == "seqrec":
        test_data = SeqRecDataset(args, mode="test_cold", sample_num=args.sample_num)
    else:
        raise NotImplementedError
    return test_data

def prefix_allowed_tokens_fn(candidate_trie):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        trie_out = candidate_trie.get(sentence)
        return trie_out

    return prefix_allowed_tokens

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def computeTopNAccuracy(GroundTruth, predictedIndices, topN, col_dict,rank=None):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []
    print("User length:", len(predictedIndices))
    
    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        user_length = 0
        cnt = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                user_length += 1
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        if predictedIndices[i][j] not in col_dict:
                            print("error")
                            exit()
                        else:
                            col = float(col_dict[predictedIndices[i][j]])
                            col = 1
                        dcg += 1.0/math.log2(j + 2)/col

                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))/col
                            mrrFlag = False
                        userHit += 1/col
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR
        
        precision.append(round(sumForPrecision / user_length, 4))
        recall.append(round(sumForRecall / user_length, 4))
        NDCG.append(round(sumForNdcg / user_length, 4))
        MRR.append(round(sumForMRR / user_length, 4))
        
    return precision, recall, NDCG, MRR


def print_results(loss, valid_result, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None: 
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(

                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]]), 
                            '-'.join([str(x) for x in valid_result[2]]), 
                            '-'.join([str(x) for x in valid_result[3]])))
    if test_result is not None: 
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]]), 
                            '-'.join([str(x) for x in test_result[2]]), 
                            '-'.join([str(x) for x in test_result[3]])))

def find_path(base_path):
    if base_path=="None" or base_path is None:
        return base_path
    if re.fullmatch(r".*checkpoint-\d+$", base_path) and os.path.isdir(base_path):
        return base_path

    if not os.path.isdir(base_path):
        base_path=base_path.replace("2e-5","1e-4")
    if not os.path.isdir(base_path):
        raise ValueError(f"Path is not a directory or does not exist: {base_path}")

    checkpoints = []
    for item in os.listdir(base_path):
        full_path = os.path.join(base_path, item)
        if os.path.isdir(full_path) and re.fullmatch(r"checkpoint-\d+", item):
            checkpoints.append((int(item.split("-")[-1]), full_path))

    if not checkpoints:
        raise ValueError(f"No 'checkpoint-<number>' subdirectory found in: {base_path}")

    latest = max(checkpoints, key=lambda x: x[0])[1]
    print(f"Resuming from checkpoint: {latest}")
    return latest