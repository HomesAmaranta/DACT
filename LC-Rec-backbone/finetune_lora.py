import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import sys
from typing import List
# import wandb
import torch
import math
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(gpu_name)
import copy


os.environ["WANDB_MODE"]="disabled"  

# from fastchat.train.llama2_flash_attn_monkey_patch import (
#     replace_llama_attn_with_flash_attn,
# )

# replace_llama_attn_with_flash_attn()

import transformers
from transformers import Qwen2Tokenizer, Qwen2Config, Qwen2ForCausalLM, EarlyStoppingCallback

from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)


from utils import *
from collator import Collator, Collator_DecoderOnly, Collator_DecoderOnly_manual, TestCollator
from torch.utils.data import DataLoader

def train(args):

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print(vars(args))

    # if ddp:
    device_map = {"": local_rank}
    args.resume_from_checkpoint=find_path(args.resume_from_checkpoint)
    print("resume_from_checkpoint:",args.resume_from_checkpoint)
    config = Qwen2Config.from_pretrained(args.base_model)
    if args.resume_from_checkpoint=="None" or args.resume_from_checkpoint is None:
        tokenizer = Qwen2Tokenizer.from_pretrained(args.base_model,
                                                model_max_length=args.model_max_length,
                                                padding_side="left",)
    else:
        print("Loading tokenizer from checkpoint...")
        tokenizer = Qwen2Tokenizer.from_pretrained(args.resume_from_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    train_data, valid_data = load_datasets(args)
    print("before add tokenizer len",len(tokenizer))
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    print("after add tokenizer len",len(tokenizer))
    
    config.vocab_size = len(tokenizer)
    if local_rank == 0:
        print("add {} new token.".format(add_num))
        print("data num:", len(train_data))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)

    collator = Collator_DecoderOnly_manual(args, tokenizer)

    load_8bit = True 
    dtype =torch.bfloat16
    bf16 = True 

    model = Qwen2ForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        load_in_8bit=load_8bit,
        device_map=device_map,
        # attn_implementation="flash_attention_2",
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
    if args.resume_from_checkpoint:
        temp=args.resume_from_checkpoint
        checkpoint_name = os.path.join(
            args.resume_from_checkpoint, "adapter_model.bin"
        )  # only LoRA model - LoRA config above has to fit
        args.resume_from_checkpoint = False  # So the trainer won't try loading its state
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            if local_rank == 0:
                print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name,map_location="cpu")
            if add_num>0:
                print("Add tokens num is ",add_num)
                pop_list =[] 
                for _ in adapters_weights.keys():
                    if "embed_tokens" in _:
                        l = adapters_weights[_].shape[0]
                        model.model.model.embed_tokens.modules_to_save.default.weight.data[:l]=copy.deepcopy(adapters_weights[_])
                        pop_list.append(_)
                    if "lm_head" in _:
                        l = adapters_weights[_].shape[0]
                        model.model.lm_head.modules_to_save.default.weight.data[:l]=copy.deepcopy(adapters_weights[_])
                        pop_list.append(_)
                
                for _ in pop_list:
                    adapters_weights.pop(_)
                set_peft_model_state_dict(model, adapters_weights)
            else:
                print("No add tokens.")
                set_peft_model_state_dict(model, adapters_weights)
            del adapters_weights
        else:
            if local_rank == 0:
                print(f"Checkpoint {checkpoint_name} not found")
            if temp!="None":
                print("error")
                exit()
    for n, p in model.named_parameters():
        if "original_module" in n and any(module_name in n for module_name in config.modules_to_save):
            p.requires_grad = False

    if local_rank == 0:
        model.print_trainable_parameters()

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    one_epoch = math.ceil(len(train_data)/args.per_device_batch_size/args.gradient_accumulation_steps/4)
    steps = int(one_epoch/5)
    

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            seed=args.seed,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=10,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            report_to=None,
            bf16=bf16,
            logging_steps=args.logging_step,
            optim=args.optim,
            gradient_checkpointing=True,
            eval_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=steps,
            save_steps=steps,
            output_dir=args.output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            eval_delay=1 if args.save_and_eval_strategy=="epoch" else  one_epoch,
            save_safetensors=False
        ),
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=10)],
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    if args.epochs:
        try:
            trainer.save_state()
            trainer.save_model(output_dir=args.output_dir)
            if int(os.environ.get("LOCAL_RANK"))==0: 
                opt_file = os.path.join(find_path(args.output_dir), "optimizer.pt")
                if os.path.exists(opt_file):
                    os.remove(opt_file)
        except:
            if int(os.environ.get("LOCAL_RANK"))==0: 
                model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                print(f"The best model is saved at {args.output_dir}")

    if local_rank == 0 and args.eval_after_train:
        from test_qwen import Trie, prefix_allowed_tokens_fn, get_greedy_prefix_allowed_tokens_fn, get_topk_results

        print("** model loaded")
        model.eval()

        prompt_ids = [0]

        test_data = load_test_dataset(args)
        if args.subset_test:
            args.sample_num = 200
            test_data = load_test_dataset(args)

        tokenizer.padding_side = "left"
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
            prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie, tokenizer)


        test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                                shuffle=False, num_workers=0, pin_memory=True)

        print("data num:", len(test_data))

        import time
    print(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_test_args(parser)
    parser = parse_dataset_args(parser)
    parser.add_argument("--data_file", type=str, default=None)
    args = parser.parse_args()

    train(args)
