import os
import sys
sys.path.append("..")

import time
import datetime
import argparse
import numpy as np
import pandas as pd
import random
from random import SystemRandom
from sklearn import model_selection

import torch
import torch.nn as nn
import torch.optim as optim
# from lib.Tuner import Tuner
import lib.utils as utils
from lib.parse_datasets import parse_datasets
from model.tPatchGNN import *

from exp import train_main
parser = argparse.ArgumentParser('IMTS Forecasting')

parser.add_argument('--state', type=str, default='def')
parser.add_argument('-n',  type=int, default=int(1e8), help="Size of the dataset")
parser.add_argument('--hop', type=int, default=1, help="hops in GNN")
# parser.add_argument('--ndim', type=int, default=1, help="hops in GNN")
parser.add_argument('--nhead', type=int, default=1, help="heads in Transformer")
parser.add_argument('--tf_layer', type=int, default=1, help="# of layer in Transformer")
parser.add_argument('--nlayer', type=int, default=1, help="# of layer in TSmodel")
parser.add_argument('--epoch', type=int, default=1000, help="training epoches")
parser.add_argument('--patience', type=int, default=10, help="patience for early stop")
parser.add_argument('--history', type=int, default=24, help="number of hours (months for ushcn and ms for activity) as historical window")
parser.add_argument('-ps', '--patch_size', type=float, default=24, help="window size for a patch")
parser.add_argument('--stride', type=float, default=24, help="period stride for patch sliding")
parser.add_argument('--logmode', type=str, default="a", help='File mode of logging.')

parser.add_argument('--lr',  type=float, default=5e-4, help="Starting learning rate.")
parser.add_argument('--w_decay', type=float, default=0.0, help="weight decay.")
parser.add_argument('-b', '--batch_size', type=int, default=32)

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('--seed', type=int, default=1, help="Random seed")
parser.add_argument('--dataset', type=str, default='physionet', help="Dataset to load. Available: physionet, mimic, ushcn")

parser.add_argument('--K', type=int, default = 4 , help = '')


parser.add_argument('--random_dim', type=int, default = 64 , help = '')
 

parser.add_argument('--pred_len', type = int, required = False  , default = 1000, help = 'number_of_jobs for optuna')

parser.add_argument('--enc_in', type = int, required = False  , default = 12, help = 'number_of_jobs for optuna')




parser.add_argument('--preconvdim', type = int, required = True , default = 16, help = 'number_of_jobs for optuna')

parser.add_argument('--n_jobs', type = int, required = False, choices = [1, 2, 3, 4], default = 1, help = 'number_of_jobs for optuna')
 
parser.add_argument('--quantization', type=float, default=0.0, help="Quantization on the physionet dataset.")
parser.add_argument('--model', type=str, default='tPatchGNN', help="Model name")
parser.add_argument('--outlayer', type=str, default='Linear', help="Model name")
parser.add_argument('-hd', '--hid_dim', type=int, default=64, help="Number of units per hidden layer")
parser.add_argument('-td', '--te_dim', type=int, default=10, help="Number of units for time encoding")
parser.add_argument('-nd', '--node_dim', type=int, default=10, help="Number of units for node vectors")
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use.')
parser.add_argument('--align', type=str, default='patch', help='.')
args = parser.parse_args()
args.npatch = int(np.ceil((args.history - args.patch_size) / args.stride)) + 1 # (window size for a patch)

def main():
    train_main(args)

if __name__ == '__main__':

        train_main(args)  
