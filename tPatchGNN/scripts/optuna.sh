
patience=50
gpu=0
model='simple'
mkdir -p optuna

export CUDA_VISIBLE_DEVICES=1
python run_models.py \
    --model $model \
    --align variate \
    --dataset activity --state 'def' --history 3000 \
    --patience $patience \
    --nhead 1 \
    --use_hyperParam_optim 1 \
    --outlayer Linear --seed 1 --gpu $gpu > optuna/activity.txt 2>&1
