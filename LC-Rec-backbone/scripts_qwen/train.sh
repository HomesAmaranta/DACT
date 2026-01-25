export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1 
DATASET=Tools
lora="--lora"
only_train_response="--only_train_response"
model_class=Qwen2.5-1.5B
subset=""
ft=1
ckpt_name= # your ckpt path
index_name=_0.7_dact.npy
data_file=_0.7.parquet
post_name=test
lr=2e-5
seed= # your seed

model_path=Qwen/Qwen2.5-1.5B-Instruct # or write your cached model path
phase=0.7
for wd in 0
do
    suffix=${model_class}-${lr}lr-${wd}wd-${suffix}
    logfile=../log/Qwen//${DATASET}/phase${phase}/${index_name}/train_${suffix}-log.txt 
    logdir=../log/Qwen//${DATASET}/phase${phase}/${index_name}
    if [ ! -d "$logdir" ]; then
        mkdir -p "$logdir"
        echo "Directory $logdir created."
    else
        echo "Directory $logdir already exists."
    fi
    OUTPUT_DIR=../ckpt/Qwen/${DATASET}/phase${phase}/${index_name}/${suffix}
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=5881 ../finetune_lora.py \
        --base_model $model_path \
        --output_dir $OUTPUT_DIR \
        --subseq \
        --dataset $DATASET \
        --per_device_batch_size 16 \
        --gradient_accumulation_steps 2 \
        --learning_rate $lr \
        --epochs 50 \
        --lora_r 8 \
        --lora_alpha 32 \
        --lora_target_modules "q_proj,v_proj,o_proj,up_proj,down_proj" \
        --weight_decay $wd \
        --save_and_eval_strategy steps \
        --warmup_steps 200 \
        --lora_modules_to_save "embed_tokens,lm_head" \
        --index_file ${index_name} \
        --special_token_for_answer "|start_of_answer|" \
        --test_batch_size 4 \
        --resume_from_checkpoint ${ckpt_name} \
        --num_beams 20 \
        --phase ${phase} \
        --ft 1 \
        --post $post_name \
        --seed ${seed} \
        --data_file ${data_file} \
        ${subset} \
        ${only_train_response} 
done

