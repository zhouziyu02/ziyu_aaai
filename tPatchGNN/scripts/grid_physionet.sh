export CUDA_VISIBLE_DEVICES=1

 
model='simple'
 
mkdir -p gridsearch


 
# export CUDA_VISIBLE_DEVICES=2 # Assigns physical GPU 0 to be logical GPU 0 for this subshell
#   for seed_val in $(seq 51 100); do
#     python run_models.py \
#       --model $model \
#       --align variate \
#       --dataset physionet --state 'def' --history 24 \
#       --patience 50 --batch_size 16 --lr 0.001 \
#       --nhead 1 --tf_layer 1 --nlayer 2 --K 2 \
#       --hid_dim 64 --preconvdim 16 --te_dim 20 \
#       --outlayer Linear --seed $seed_val --gpu 0 
#     # No '&' here, so runs sequentially on the assigned device
#   done
 

 
export CUDA_VISIBLE_DEVICES=3 # Assigns physical GPU 1 to be logical GPU 0 for this subshell
  for seed_val in $(seq 101 150); do
    python run_models.py \
      --model $model \
      --align variate \
      --dataset physionet --state 'def' --history 24 \
      --patience 50 --batch_size 16 --lr 0.001 \
      --nhead 1 --tf_layer 1 --nlayer 2 --K 2 \
      --hid_dim 64 --preconvdim 16 --te_dim 20 \
      --outlayer Linear --seed $seed_val --gpu 0 
    # No '&' here, so runs sequentially on the assigned device
  done
 
# Wait for all backgrounded groups of tasks to complete
 
 

# export CUDA_VISIBLE_DEVICES=3
# python run_models.py \
#       --model $model \
#       --align variate \
#       --dataset physionet --state 'def' --history 24 \
#       --patience 50 --batch_size 16 --lr 0.001 \
#       --nhead 1 --tf_layer 1 --nlayer 2 --K 2 \
#       --hid_dim 64 --preconvdim 16 --te_dim 20 \
#       --outlayer Linear --seed 1 --gpu 0  
 
 
# for batch_size in 16 32 64 128 256; do
# for k in 2 4 8; do
# for te_dim in 10 20 30; do
# for nlayer in 1 2 3; do

# export CUDA_VISIBLE_DEVICES=0
# python run_models.py \
#     --model $model \
#     --align variate \
#     --dataset physionet --state 'def' --history 24 \
#     --patience 50 --batch_size $batch_size --lr 1e-3 \
#     --nhead 1 --tf_layer 1 --nlayer $nlayer --K $k \
#     --te_dim $te_dim  --hid_dim 16 \
#     --outlayer Linear --seed 1 --gpu 0  >> gridsearch/physionet.txt 2>&1 &

# export CUDA_VISIBLE_DEVICES=2
# python run_models.py \
#     --model $model \
#     --align variate \
#     --dataset physionet --state 'def' --history 24 \
#     --patience 50 --batch_size $batch_size --lr 1e-3 \
#     --nhead 1 --tf_layer 1 --nlayer $nlayer --K $k \
#     --te_dim $te_dim  --hid_dim 32 \
#     --outlayer Linear --seed 1 --gpu 0  >> gridsearch/physionet.txt 2>&1 &

# export CUDA_VISIBLE_DEVICES=3
# python run_models.py \
#     --model $model \
#     --align variate \
#     --dataset physionet --state 'def' --history 24 \
#     --patience 50 --batch_size $batch_size --lr 1e-3 \
#     --nhead 1 --tf_layer 1 --nlayer $nlayer --K $k \
#     --te_dim $te_dim  --hid_dim 64 \
#     --outlayer Linear --seed 1 --gpu 0  >> gridsearch/physionet.txt 2>&1

# # Wait for all background jobs (on GPU 0 and GPU 2) to complete
# # before starting the next iteration for these GPUs.
# # The job on GPU 3 runs in the foreground, so the script naturally waits for it.
# wait

# done
# done
# done
# done