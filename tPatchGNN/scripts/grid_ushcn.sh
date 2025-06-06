export CUDA_VISIBLE_DEVICES=0

 
model='simple'
 
mkdir -p gridsearch

# Model: simple, Seed: 1, lr: 0.001, batch_size: 256, nhead: 1, nlayer: 1, K: 4, hid_dim: 128
#Model: simple, Seed: 1, lr: 0.001, batch_size: 128, nhead: 1, nlayer: 2, K: 2, hid_dim: 64, preconvdim:8


# export CUDA_VISIBLE_DEVICES=3 # Assigns physical GPU 0 to be logical GPU 0 for this subshell
#   for seed_val in $(seq 10 50); do
#     python run_models.py \
#       --model $model \
#       --align variate \
#       --dataset ushcn --state 'def' --history 24 \
#       --patience 50 --batch_size 256 --lr 0.001 \
#       --nhead 1 --tf_layer 1 --nlayer 1 --K 4 \
#       --hid_dim 128 --preconvdim 16 --te_dim 10 \
#       --outlayer Linear --seed $seed_val --gpu 0 >> gridsearch/ushcn.txt 2>&1
#   done

# Wait for all backgrounded groups of tasks to complete


export CUDA_VISIBLE_DEVICES=3 # Assigns physical GPU 0 to be logical GPU 0 for this subshell
  for seed_val in $(seq 82 120); do
    python run_models.py \
      --model $model \
      --align variate \
      --dataset ushcn --state 'def' --history 24 \
      --patience 50 --batch_size 128 --lr 0.001 \
      --nhead 1 --tf_layer 1 --nlayer 2 --K 2 \
      --hid_dim 64 --preconvdim 8 --te_dim 10 \
      --outlayer Linear --seed $seed_val --gpu 0
  done


# for batch_size in 128 192 256; do
# for k in 2 4 8; do
# for te_dim in 10 20; do
# for nlayer in 1 2; do
# for preconvdim in 8 16 32; do


# export CUDA_VISIBLE_DEVICES=0
# python run_models.py \
#     --model $model \
#     --align variate \
#     --dataset ushcn --state 'def' --history 24 \
#     --patience 40 --batch_size $batch_size --lr 1e-3 \
#     --nhead 1 --tf_layer 1 --nlayer $nlayer --K $k \
#     --te_dim $te_dim  --hid_dim 128 --preconvdim $preconvdim \
#     --outlayer Linear --seed 1 --gpu 0 >> gridsearch/ushcn.txt 2>&1 &


# export CUDA_VISIBLE_DEVICES=0
# python run_models.py \
#     --model $model \
#     --align variate \
#     --dataset ushcn --state 'def' --history 24 \
#     --patience 40 --batch_size $batch_size --lr 1e-3 \
#     --nhead 1 --tf_layer 1 --nlayer $nlayer --K $k \
#     --te_dim $te_dim  --hid_dim 16 --preconvdim $preconvdim \
#     --outlayer Linear --seed 1 --gpu 0 >> gridsearch/ushcn.txt 2>&1 &

# export CUDA_VISIBLE_DEVICES=1
# python run_models.py \
#     --model $model \
#     --align variate \
#     --dataset ushcn --state 'def' --history 24 \
#     --patience 40 --batch_size $batch_size --lr 1e-3 \
#     --nhead 1 --tf_layer 1 --nlayer $nlayer --K $k \
#     --te_dim $te_dim  --hid_dim 32 --preconvdim $preconvdim \
#     --outlayer Linear --seed 1 --gpu 0 >> gridsearch/ushcn.txt 2>&1 &

# export CUDA_VISIBLE_DEVICES=1
# python run_models.py \
#     --model $model \
#     --align variate \
#     --dataset ushcn --state 'def' --history 24 \
#     --patience 40 --batch_size $batch_size --lr 1e-3 \
#     --nhead 1 --tf_layer 1 --nlayer $nlayer --K $k \
#     --te_dim $te_dim  --hid_dim 64 --preconvdim $preconvdim \
#     --outlayer Linear --seed 1 --gpu 0 >> gridsearch/ushcn.txt 2>&1



# # Wait for all background jobs (on GPU 0 and GPU 2) to complete
# # before starting the next iteration for these GPUs.
# # The job on GPU 3 runs in the foreground, so the script naturally waits for it.
# wait

done
done
done
done
done