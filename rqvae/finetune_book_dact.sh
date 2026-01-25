dataset=Tools
s=0.6
e=0.7
dir=0.7_dact
data_path=../data/${dataset}/item_emb_${e}.parquet
cf_loss=1
cf_path=../data/${dataset}/item_cf_${e}.pth

stable_loss=1
stable_weight=1
kl_loss_all=1
kl_temp=1
rqvae_path=./ckpt/${dataset}/0.6_cf/epoch_19999_collision_0.0578_model.pth

epochs=5000

python main.py --data_path $data_path \
               --cf_loss $cf_loss \
               --cf_path $cf_path \
               --stable_loss $stable_loss \
               --stable_weight $stable_weight \
               --rqvae_path $rqvae_path \
               --epochs $epochs \
               --kl_loss_all $kl_loss_all \
               --kl_temp $kl_temp \
               --dataset $dataset \
               --dir $dir \
               --lr 0.0001