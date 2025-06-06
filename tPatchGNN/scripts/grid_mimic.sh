export CUDA_VISIBLE_DEVICES=0

 
model='simple'
 
mkdir -p gridsearch


# Model: simple, Best Epoch:229, Seed: 1, lr: 0.001, batch_size: 32, nhead: 1, tedim: 10, nlayer: 1, K: 2, hid_dim: 16, preconvdim:16

export CUDA_VISIBLE_DEVICES=2 # Assigns physical GPU 0 to be logical GPU 0 for this subshell
  for seed_val in $(seq 120 150); do
    python run_models.py \
      --model $model \
      --align variate \
      --dataset mimic --state 'def' --history 24 \
      --patience 50 --batch_size 32 --lr 0.001 \
      --nhead 1 --tf_layer 1 --nlayer 1 --K 2 \
      --hid_dim 16 --preconvdim 16 --te_dim 10 \
      --outlayer Linear --seed $seed_val --gpu 0
  done


# for batch_size in 32; do
# for k in 4; do
# for te_dim in 10 20; do
# for nlayer in 1 2; do
# for preconvdim in 16 32; do

# export CUDA_VISIBLE_DEVICES=3
# python run_models.py \
#     --model $model \
#     --align variate \
#     --dataset mimic --state 'def' --history 24 \
#     --patience 50 --batch_size $batch_size --lr 1e-3 \
#     --nhead 1 --tf_layer 1 --nlayer $nlayer --K $k \
#     --te_dim $te_dim --node_dim 10 --hid_dim 64 --preconvdim $preconvdim \
#     --outlayer Linear --seed 1 --gpu 0   
 
# done
# done
# done
# done
# done



# for batch_size in 32 64; do
# for k in 2 4; do
# for te_dim in 10 20; do
# for nlayer in 1 2; do
# for preconvdim in 16 32; do


 
    

# export CUDA_VISIBLE_DEVICES=1
# python run_models.py \
#     --model $model \
#     --align variate \
#     --dataset mimic --state 'def' --history 24 \
#     --patience 50 --batch_size $batch_size --lr 1e-3 \
#     --nhead 1 --tf_layer 1 --nlayer $nlayer --K $k \
#     --te_dim $te_dim --node_dim 10 --hid_dim 32 --preconvdim $preconvdim \
#     --outlayer Linear --seed 1 --gpu 0  >> gridsearch/mimic.txt 2>&1 

 

# done
# done
# done
# done
# done



 