import os
import sys
sys.path.append("..")
from thop import profile
import time
import datetime
 
import numpy as np

from random import SystemRandom

import torch
import torch.nn as nn
import torch.optim as optim
# import optuna
 
from lib.parse_datasets import parse_datasets
from model.tPatchGNN import *
 
from model.tPatchGNN import tPatchGNN
from model.KFNet import KFNet



def build_model(args):
    if args.model == 'tPatchGNN':
        model = tPatchGNN(args)
    elif args.model == 'KFNet':    
        model = KFNet(args)
    else:
        raise ValueError(f"Unknown model name: {args.model}")
    return model


def train_main(args, optunaTrialReport=None):
 
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")

    args.device = device

    utils.setup_seed(args.seed)

    experimentID = args.load
    if experimentID is None:
        experimentID = int(SystemRandom().random() * 100000)
    ckpt_path = os.path.join(args.save, f"experiment_{experimentID}.ckpt")

     
    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2):]
    input_command = " ".join(input_command)

    
    data_obj = parse_datasets(args)
    input_dim = data_obj["input_dim"]
    args.ndim = input_dim

    
    model = build_model(args).to(args.device)
    
      
    dummy_batch = utils.get_next_batch(data_obj["train_dataloader"])
    dummy_tp_to_predict = dummy_batch["tp_to_predict"]
    dummy_observed_data = dummy_batch["observed_data"]
    dummy_observed_tp = dummy_batch["observed_tp"]
    dummy_observed_mask = dummy_batch["observed_mask"]

    flops, params_thop = profile(
        model,
        inputs=(dummy_tp_to_predict, dummy_observed_data, dummy_observed_tp, dummy_observed_mask),
        verbose=False
    )
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

     
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

     
    num_batches = data_obj["n_train_batches"]
    best_val_mse = np.inf
    test_res = None   
    best_iter = 0
    inference_times = []  

    global_train_start = time.time()
    total_iters = 0
 
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        iter_count = 0
        epoch_train_losses = []
        time_now = time.time()

        model.train()
        for i in range(num_batches):
            optimizer.zero_grad()
            batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
            train_res = compute_all_losses(model, batch_dict)   
            loss = train_res["loss"]
            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.item())
            iter_count += 1
            total_iters += 1
 
            if (i + 1) % 10 == 0:
                speed = (time.time() - time_now) / iter_count
                left_time = speed * (num_batches - (i + 1))
                print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                iter_count = 0
                time_now = time.time()

        epoch_time = time.time() - epoch_start_time
        train_loss_avg = np.mean(epoch_train_losses)
        print(f"Epoch: {epoch + 1} cost time: {epoch_time:.2f}s")

         
        model.eval()
        with torch.no_grad():
            val_res = evaluation(model, data_obj["val_dataloader"], data_obj["n_val_batches"])
             
            if val_res["mse"] < best_val_mse:
                best_val_mse = val_res["mse"]
                best_iter = epoch
                
                 
                inference_start = time.time()
                test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])
                inference_end = time.time()
                inference_times.append(inference_end - inference_start)
                
            vali_loss = val_res["loss"]
            test_loss = test_res["loss"] if test_res is not None else np.inf

         

        print(f"Epoch: {epoch + 1}, Steps: {num_batches} "
              f"| Train Loss: {train_loss_avg:.7f} "
              f"Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")

         
        if (epoch - best_iter) >= args.patience:
            no_improvement = epoch - best_iter
            print(f"Early stopping triggered. No improvement for {no_improvement} epoch(s).")
            break

     
    global_training_time = time.time() - global_train_start
    avg_iter_time = global_training_time / total_iters
    avg_inference_time = np.mean(inference_times) if inference_times else 0.0

    if args.dataset == 'physionet':
        final_result_path = "seed_physionet.txt"
    elif args.dataset == 'ushcn':
        final_result_path = "seed_ushcn.txt"
    elif args.dataset == 'mimic':
        final_result_path = "seed_mimic.txt"
    elif args.dataset == 'activity':
        final_result_path = "seed_activity.txt"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
        
 
    with open("efficiency.txt", "a", encoding="utf-8") as f:
        f.write(f"Model: {args.model}, Dataset: {args.dataset}, Best Epoch:{best_iter}, Seed: {args.seed}, lr: {args.lr}, batch_size: {args.batch_size}, nhead: {args.nhead}, tedim: {args.te_dim}, nlayer: {args.nlayer}, K: {args.K}, hid_dim: {args.hid_dim}, preconvdim:{args.preconvdim}\n")
        if test_res is not None:
            f.write("MSE: {:.5f}, MAE: {:.5f}, MAPE: {:.2f}%\n".format(
                test_res["mse"], test_res["mae"], test_res["mape"] * 100
            ))
        else:
            f.write("No test result available.\n")

        f.write("Model FLOPs: {:.2f}, Parameter count (thop): {:.2f}, Manual param count: {}\n".format(
            flops, params_thop, param_count
        ))
        f.write("Average training time per iteration: {:.4f}s\n".format(avg_iter_time))
        f.write("Average inference time: {:.4f}s\n".format(avg_inference_time))

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        minutes = int(global_training_time // 60)
        seconds = int(global_training_time % 60)
        f.write("Time now: {}, Time for training: {}m {}s\n".format(
            current_time, minutes, seconds
        ))
        f.write('\n')
 
    if test_res is not None:
        final_test_mse = test_res["mse"]
    else:
        final_test_mse = float('inf')

    return final_test_mse
