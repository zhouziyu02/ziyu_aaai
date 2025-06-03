### tPatchGNN ###   bash ./scripts/run_all.sh
export CUDA_VISIBLE_DEVICES=0
patience=50
gpu=0
seed=1
model='simple'
mkdir -p logs
# tPatchGNN
#simple
# original scripts
# for seed in {1..5}
# do
#     python run_models.py \
#     --dataset physionet --state 'def' --history 24 \
#     --patience $patience --batch_size 32 --lr 1e-3 \
#     --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
#     --te_dim 10 --node_dim 10 --hid_dim 64 \
#     --outlayer Linear --seed $seed --gpu $gpu


#     python run_models.py \
#     --dataset mimic --state 'def' --history 24 \
#     --patience $patience --batch_size 32 --lr 1e-3 \
#     --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
#     --te_dim 10 --node_dim 10 --hid_dim 64 \
#     --outlayer Linear --seed $seed --gpu $gpu


    # python run_models.py \
    # --model tPatchGNN \
    # --dataset activity --state 'def' --history 3000 \
    # --patience $patience --batch_size 32 --lr 1e-3 \
    # --patch_size 300 --stride 300 --nhead 1 --tf_layer 1 --nlayer 1 \
    # --te_dim 10 --node_dim 10 --hid_dim 32 \
    # --outlayer Linear --seed $seed --gpu $gpu 


#     python run_models.py \
#     --dataset ushcn --state 'def' --history 24 \
#     --patience $patience --batch_size 192 --lr 1e-3 \
#     --patch_size 2 --stride 2 --nhead 1 --tf_layer 1 --nlayer 1 \
#     --te_dim 10 --node_dim 10 --hid_dim 32 \
#     --outlayer Linear --seed $seed --gpu $gpu
# done






# export CUDA_VISIBLE_DEVICES=0
# python run_models.py \
#     --model $model \
#     --align patch \
#     --dataset activity --state 'def' --history 3000 \
#     --patience $patience --batch_size 32 --lr 1e-3 \
#     --patch_size 300 --stride 300 --nhead 1 --nlayer 2 \
#     --te_dim 10 --hid_dim 32 \
#     --outlayer Linear --seed $seed --gpu $gpu


# export CUDA_VISIBLE_DEVICES=1
# python run_models.py \
#     --model $model \
#     --align variate \
#     --dataset physionet --state 'def' --history 24 \
#     --patience $patience --batch_size 32 --lr 1e-3 \
#     --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
#     --te_dim 10 --node_dim 10 --hid_dim 64 \
#     --outlayer Linear --seed $seed --gpu $gpu > logs/physionet.txt 2>&1 &







 
# export CUDA_VISIBLE_DEVICES=1
# python run_models.py \
#     --model $model \
#     --align variate \
#     --dataset mimic --state 'def' --history 24 \
#     --patience $patience --batch_size 32 --lr 1e-3 \
#     --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
#     --te_dim 10 --node_dim 10 --hid_dim 64 \
#     --outlayer Linear --seed $seed --gpu $gpu   


# export CUDA_VISIBLE_DEVICES=1
# python run_models.py \
#     --model $model \
#     --align variate \
#     --dataset activity --state 'def' --history 3000 \
#     --patience $patience --batch_size 32 --lr 1e-3 \
#     --patch_size 300 --stride 300 --nhead 1 --nlayer 1 \
#     --te_dim 10 --hid_dim 32 \
#     --outlayer Linear --seed $seed --gpu $gpu

# Model: simple, Seed: 1, lr: 0.001, batch_size: 256, nhead: 1, nlayer: 1, K: 4, hid_dim: 128

# export CUDA_VISIBLE_DEVICES=3
# python run_models.py \
#     --model $model \
#     --align variate \
#     --dataset ushcn --state 'def' --history 24 \
#     --patience 50 --batch_size 256 --lr 1e-3 \
#     --nhead 1 --tf_layer 1 --nlayer 1 \
#     --te_dim 10 --node_dim 10 --hid_dim 128 --K 4 \
#     --preconvdim 16 \
#     --outlayer Linear --seed 1 --gpu 0

Model: simple, Seed: 1, lr: 0.001, batch_size: 128, nhead: 1, nlayer: 2, K: 2, hid_dim: 64, preconvdim:8

(
  export CUDA_VISIBLE_DEVICES=3 # Assigns physical GPU 0 to be logical GPU 0 for this subshell
  for seed_val in $(seq 1 50); do
    python run_models.py \
      --model $model \
      --align variate \
      --dataset ushcn --state 'def' --history 24 \
      --patience 50 --batch_size 256 --lr 0.001 \
      --nhead 1 --tf_layer 1 --nlayer 1 --K 4 \
      --hid_dim 128 --preconvdim 16 --te_dim 10 \
      --outlayer Linear --seed $seed_val --gpu 0 >> gridsearch/ushcn.txt 2>&1
    # No '&' here, so runs sequentially on the assigned device
  done
)  
 

# Wait for all backgrounded groups of tasks to complete
wait
 
# export CUDA_VISIBLE_DEVICES=1
# python run_models.py \
#     --model $model \
#     --align variate \
#     --dataset physionet --state 'def' --history 24 \
#     --patience $patience --batch_size 32 --lr 1e-3 \
#     --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
#     --te_dim 10 --node_dim 10 --hid_dim 64 \
#     --outlayer Linear --seed $seed --gpu $gpu




# data,pred_len,loss,batch_size,lr,patch_size,stride,nlayer,hid_dim,K,random_dim
# activity,,0.0028559379279613495,32,0.0015123477154516524,300,300,1,32,4,64
