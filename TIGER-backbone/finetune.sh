dataset=Tools

s=0.6
e=0.7
suffix=_dact
seed= # your seed
if [ $s == 0.6 ]; then
    ckpt_path_shu="_cf"
else
    ckpt_path_shu=$suffix
fi

ckpt_path="./ckpt/tiger_${dataset}_${s}${ckpt_path_shu}.pth" # 预训练模型的路径

save_path="./ckpt/tiger_${dataset}_${e}${suffix}.pth"
code_path="../data/${dataset}/${dataset}_${e}${suffix}.npy"
log_path="./logs/tiger_${dataset}_${e}${suffix}.log"
type=train
data="_${e}.parquet"
current_epoch=0
num_epochs=200
evaluate_epoch=0
desc=finetune

echo "python main_trie.py --type $type \
            --ckpt_path $ckpt_path \
            --code_path $code_path \
            --log_path $log_path \
            --data $data \
            --desc $desc \
            --save_path $save_path \
            --current_epoch $current_epoch \
            --num_epochs $num_epochs \
            --evaluate_epoch $evaluate_epoch"

python main_trie.py --type $type \
            --ckpt_path $ckpt_path \
            --code_path $code_path \
            --log_path $log_path \
            --data $data \
            --desc $desc \
            --save_path $save_path \
            --current_epoch $current_epoch \
            --num_epochs $num_epochs \
            --evaluate_epoch $evaluate_epoch \
            --dataset $dataset \
            --seed $seed