export CUDA_VISIBLE_DEVICES=0
python run_models.py \
      --model KFNet \
      --align variate \
      --dataset activity --state 'def' --history 3000 \
      --patience 50 --batch_size 32 --lr 0.001 \
      --nhead 1 --tf_layer 1 --nlayer 1 --K 4 \
      --hid_dim 16 --preconvdim 32 --te_dim 10 \
      --outlayer Linear --seed 1 --gpu 0