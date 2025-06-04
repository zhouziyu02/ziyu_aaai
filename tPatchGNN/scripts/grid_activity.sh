export CUDA_VISIBLE_DEVICES=0

 
model='simple'
 
mkdir -p gridsearch


# export CUDA_VISIBLE_DEVICES=1
# python run_models.py \
#     --model $model \
#     --align variate \
#     --dataset activity --state 'def' --history 3000 \
#     --patience $patience --batch_size 32 --lr 1e-3 \
#     --patch_size 300 --stride 300 --nhead 1 --nlayer 1 \
#     --te_dim 10 --hid_dim 32 \
#     --outlayer Linear --seed $seed --gpu $gpu
 
# Model: simple, Best Epoch:293, Seed: 1, lr: 0.001, batch_size: 32, nhead: 1, tedim: 10, nlayer: 1, K: 4, hid_dim: 16, preconvdim:32
# MSE: 0.00254, MAE: 0.03067, MAPE: 7.07%
# Model FLOPs: 15221248.00, Parameter count (thop): 4561.00, Manual param count: 5135
# Average training time per iteration: 0.0170s
# Time now: 2025-06-02 02:36:49, Time for training: 10m 6s
export CUDA_VISIBLE_DEVICES=2 # Assigns physical GPU 0 to be logical GPU 0 for this subshell
for seed_val in $(seq 51 52); do
    python run_models.py \
      --model $model \
      --align variate \
      --dataset activity --state 'def' --history 3000 \
      --patience 50 --batch_size 32 --lr 0.001 \
      --nhead 1 --tf_layer 1 --nlayer 1 --K 4 \
      --hid_dim 16 --preconvdim 32 --te_dim 10 \
      --outlayer Linear --seed $seed_val --gpu 0
done

# for batch_size in 64; do
# for k in 2 4; do
# for te_dim in 10 20; do
# for nlayer in 1 2; do
# for preconvdim in 16 32; do
# for hid_dim in 16 32; do

# export CUDA_VISIBLE_DEVICES=1
# python run_models.py \
#     --model $model \
#     --align variate \
#     --dataset activity --state 'def' --history 3000 \
#     --patience 50 --batch_size $batch_size --lr 1e-3 \
#     --nhead 1 --tf_layer 1 --nlayer $nlayer --K $k \
#     --te_dim $te_dim --hid_dim $hid_dim --preconvdim $preconvdim \
#     --outlayer Linear --seed 1 --gpu 0

# done
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



 