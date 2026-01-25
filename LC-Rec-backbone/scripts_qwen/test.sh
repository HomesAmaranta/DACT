export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1 

lora="--lora"
DATASET=Tools
phase=0.7_test

for lr in 3e-4
    do
        (
        CKPT_PATH= # your ckpt path
        logfile=../log/Qwen/test/${DATASET}/phase$phase/Reformer-LC-Rec-Qwen2.5-1.5B-${lr}lr-0wd-lora-qvoud-64r-128a-bf16-int8-log.txt 

        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=5881 ../test_qwen_ddp.py \
            --dataset $DATASET \
            --base_model Qwen/Qwen2.5-1.5B-Instruct \
            --ckpt_path $CKPT_PATH \
            --test_batch_size 32 \
            --num_beams 20 \
            --lora_r 8 \
            --lora_alpha 32 \
            --lora_modules_to_save "embed_tokens,lm_head" \
            --index_file _0.7_dact.npy \
            --data_file _0.7.parquet \
            ${lora} \
            --phase $phase \
            --ft 0 \
            --lora
        )
    done

