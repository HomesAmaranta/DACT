# 对物品（新/老）做冷启动，不包含冲撞处理，且没有旧码本
dataset=Tools
s=0.6
e=0.7
rqvae_path=./ckpt/${dataset}/${e}_dact/last.pth
output_file=../data/${dataset}/${dataset}_${e}_dact.npy
old_code=../data/${dataset}/${dataset}_${s}_cf.npy
data_path=../data/${dataset}/item_emb_${e}.parquet
first_only=1

python get_model_indices_dact.py --rqvae_path $rqvae_path \
                            --data_path $data_path \
                            --output_file $output_file \
                            --old_code $old_code \
                            --first_only $first_only 

