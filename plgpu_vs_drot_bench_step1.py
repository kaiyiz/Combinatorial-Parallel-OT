"""
This is a script that conducts the experiments shown in our paper. 
A GPU based implementation of our algorithm is compared with DROT methods (step 1).
"""
import argparse
import numpy as np
import pandas as pd
import torch
import ot
import ot.gpu
import os
import time

from transport import transport_torch as transport_pure_gpu
from matching import matching_torch_v1 as matching_pure_gpu
from load_data import load_DA_SB_cost

parser = argparse.ArgumentParser()
parser.add_argument('--nexp', type=int, default=30)
parser.add_argument('--n', type=int, default=10000)
parser.add_argument('--dataset_name', type=str, default='synthetic_OT')
parser.add_argument('--metric', type=str, default='sqeuclidean')
parser.add_argument('--delta_num', type=int, default=10)
parser.add_argument('--delta_low', type=float, default=0.001)
parser.add_argument('--delta_high', type=float, default=1)
parser.add_argument('--normalize_cost', type=int, default=1)
parser.add_argument('--is_transport', type=int, default=1)
parser.add_argument('--scale_factor', type=int, default=1)
parser.add_argument('--nlp_name', type=str, default=None)
parser.add_argument('--nlp_portion_size', type=int, default=100)
args = parser.parse_args()
args_dict = vars(args)
args_str = str(args_dict)
print("-------------------------------------step1-------------------------------------")
print(args_str)

NUM_EXPERIMENTS = int(args.nexp)
n = int(args.n)
norm_cost = args.normalize_cost
is_transport = args.is_transport
dataset_name = args.dataset_name
scale_factor = args.scale_factor
metric = args.metric
delta_num = int(args.delta_num)
delta_low = float(args.delta_low)
delta_high = float(args.delta_high)
nlp_name = args.nlp_name
nlp_portion_size = int(args.nlp_portion_size)

delta_tryouts = np.flip(np.logspace(np.log10(delta_low), np.log10(delta_high), num=delta_num))
bench_res_mean = []
bench_res_std = []
col = ['delta', 'reg_choose', 'emd_time', 'drot_gpu_time', 'pl_gpu_time', 'emd_loss', 'drot_gpu_loss', 'pl_gpu_loss', 'pl_gpu_iter']
bench_df = pd.DataFrame(columns=col)
final_cost_filename = f"./pr_cost_drot_{NUM_EXPERIMENTS}_{n}_{norm_cost}_{dataset_name}_{is_transport}_{scale_factor}_{metric}_{delta_num}_{delta_low}_{delta_high}_{nlp_name}_{nlp_portion_size}.npy"
final_iter_filename = f"./pr_iter_drot_{NUM_EXPERIMENTS}_{n}_{norm_cost}_{dataset_name}_{is_transport}_{scale_factor}_{metric}_{delta_num}_{delta_low}_{delta_high}_{nlp_name}_{nlp_portion_size}.npy"
final_time_filename = f"./pr_time_drot_{NUM_EXPERIMENTS}_{n}_{norm_cost}_{dataset_name}_{is_transport}_{scale_factor}_{metric}_{delta_num}_{delta_low}_{delta_high}_{nlp_name}_{nlp_portion_size}.npy"

pl_gpu_time = []
pl_gpu_iter = []
pl_gpu_loss = []

for delta in delta_tryouts:
    print("delta = {}".format(delta))
    for i in range(NUM_EXPERIMENTS):
        DA, SB, cost = load_DA_SB_cost(dataset_name, n=n, norm_cost=norm_cost, metric = metric, scale_factor = scale_factor, seed=i, nlp_i=i+1, nlp_name=nlp_name, nlp_portion_size=nlp_portion_size)
        C = cost.max()

        ######test on gpu###########

        with torch.no_grad():
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # print(device)
            cost_tensor = torch.tensor(cost, device=device, requires_grad=False)
            DA_tensor = torch.tensor(DA, device=device, requires_grad=False)
            SB_tensor = torch.tensor(SB, device=device, requires_grad=False)
            delta_tensor = torch.tensor([delta], device=device, requires_grad=False)
            C_tensor = torch.tensor([C], device=device, requires_grad=False)
            if is_transport:
                torch.cuda.synchronize()
                start = time.perf_counter()
                Mb, yA, yB, ot_pyt_loss, iteration = transport_pure_gpu(DA_tensor, SB_tensor, cost_tensor, delta_tensor, device=device)
                end = time.perf_counter()
            else:
                torch.cuda.synchronize()
                start = time.perf_counter()
                Mb, yA, yB, ot_pyt_loss, iteration = matching_pure_gpu(cost_tensor, C_tensor, delta_tensor, device=device)
                end = time.perf_counter()
                ot_pyt_loss = ot_pyt_loss/n
            print("pl loss = {}".format(ot_pyt_loss.cpu().numpy()))
            print("pl takes time = {}s".format(end-start))
            pl_gpu_time.append(end-start)
            pl_gpu_loss.append(ot_pyt_loss.cpu().numpy())
            pl_gpu_iter.append(iteration)

# save the final cost
np.save(final_cost_filename, pl_gpu_loss)
np.save(final_iter_filename, pl_gpu_iter)
np.save(final_time_filename, pl_gpu_time)