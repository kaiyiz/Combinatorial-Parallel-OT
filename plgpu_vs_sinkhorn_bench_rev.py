"""
This is a script that conducts the experiments shown in our paper. 
A GPU based implementation of our algorithm is compared with sinkorn methods in reverse process.
"""
import argparse
import numpy as np
import pandas as pd
import torch
import ot
import os
import time

from transport import transport_torch as transport_pure_gpu
from matching import matching_torch_v1 as matching_pure_gpu
from search_parameter import get_pl_delta_matching
from search_parameter import get_pl_delta_transport
from load_data import load_DA_SB_cost

parser = argparse.ArgumentParser()
parser.add_argument('--nexp', type=int, default=30)
parser.add_argument('--n', type=int, default=1000)
parser.add_argument('--dataset_name', type=str, default='synthetic_OT')
parser.add_argument('--metric', type=str, default='sqeuclidean')
parser.add_argument('--reg_num', type=int, default=10)
parser.add_argument('--reg_low', type=float, default=0.01)
parser.add_argument('--reg_high', type=float, default=1)
parser.add_argument('--normalize_cost', type=int, default=1)
parser.add_argument('--is_transport', type=int, default=1)
parser.add_argument('--scale_factor', type=int, default=1)
parser.add_argument('--nlp_name', type=str, default=None)
parser.add_argument('--nlp_portion_size', type=int, default=100)
args = parser.parse_args()
print(args)

NUM_EXPERIMENTS = int(args.nexp)
n = int(args.n)
norm_cost = args.normalize_cost
is_transport = args.is_transport
dataset_name = args.dataset_name
scale_factor = args.scale_factor
metric = args.metric
reg_num = int(args.reg_num)
reg_low = float(args.reg_low)
reg_high = float(args.reg_high)
nlp_name = args.nlp_name
nlp_portion_size = int(args.nlp_portion_size)

reg_tryouts = np.flip(np.logspace(np.log10(reg_low), np.log10(reg_high), num=reg_num))
bench_res_mean = []
bench_res_std = []
col = ['reg', 'choose_delta', 'emd_time', 'sink_gpu_time', 'pl_gpu_time', 'emd_loss', 'sink_gpu_loss', 'pl_gpu_loss', 'pl_gpu_iter']
bench_df = pd.DataFrame(columns=col)

for reg in reg_tryouts:
    print(reg)
    emd_time = []
    emd_loss = []
    sink_gpu_time = []
    sink_gpu_loss = []
    pl_gpu_time = []
    pl_gpu_iter = []
    pl_gpu_loss = []
    pl_gpu_delta_choose = []

    for i in range(NUM_EXPERIMENTS):
        DA, SB, cost = load_DA_SB_cost(dataset_name, n=n, norm_cost=norm_cost, metric = metric, scale_factor = scale_factor, seed=i, nlp_i=i+1, nlp_name=nlp_name, nlp_portion_size=nlp_portion_size)
        C = cost.max()
        ######test on cpu###########
        start = time.time()
        ot_loss_emd = ot.emd2(SB, DA, cost, processes=1, numItermax=100000000)
        end = time.time()
        emd_time.append(end-start)
        emd_loss.append(ot_loss_emd)

        ######test on gpu###########

        with torch.no_grad():
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            cost_tensor = torch.tensor(cost, device=device, requires_grad=False)
            DA_tensor = torch.tensor(DA, device=device, requires_grad=False)
            SB_tensor = torch.tensor(SB, device=device, requires_grad=False)

            torch.cuda.synchronize() 
            start = time.perf_counter()
            ot_pal_loss_sinkhorn = ot.sinkhorn2(SB_tensor, DA_tensor, cost_tensor, reg, method='sinkhorn', numItermax=100000000)
            end = time.perf_counter()
            sink_gpu_time.append(end-start)
            sink_gpu_loss.append(ot_pal_loss_sinkhorn.cpu().numpy())

            if is_transport:
                delta = get_pl_delta_transport(DA_tensor, SB_tensor, cost_tensor, ot_pal_loss_sinkhorn.cpu().numpy(), device = device, d = 1e-5, show_pl_time = True)
                print("delta choose = {}".format(delta))
                delta_tensor_ = torch.tensor([delta], device=device, requires_grad=False)
                cost_tensor_ = cost_tensor.clone()
                DA_tensor_ = DA_tensor.clone()
                SB_tensor_ = SB_tensor.clone()
                torch.cuda.synchronize()
                start = time.perf_counter()
                Mb, yA, yB, ot_pyt_loss, iteration = transport_pure_gpu(DA_tensor_, SB_tensor_, cost_tensor_, delta_tensor_, device=device)
                end = time.perf_counter()
            else:
                C_tensor_ = torch.tensor([C], device=device, requires_grad=False)
                delta = get_pl_delta_matching(cost_tensor, C_tensor_, ot_pal_loss_sinkhorn, n, device = device, d = 1e-5, show_pl_time = True)
                print("delta choose = {}".format(delta))
                delta_tensor_ = torch.tensor([delta], device=device, requires_grad=False)
                cost_tensor_ = cost_tensor.clone()
                torch.cuda.synchronize()
                start = time.perf_counter()
                Mb, yA, yB, ot_pyt_loss, iteration = matching_pure_gpu(cost_tensor_, C_tensor_, delta_tensor_, device=device)
                end = time.perf_counter()
                ot_pyt_loss = ot_pyt_loss/n
            pl_gpu_delta_choose.append(delta)
            pl_gpu_time.append(end-start)
            pl_gpu_loss.append(ot_pyt_loss.cpu().numpy())
            pl_gpu_iter.append(iteration)

    print("synthetic data (OT)")
    print("problem size {}".format(n))
    print("*********test on cpu************")    
    print("emd standard took {}({}) seconds with loss {}({})".format(np.mean(emd_time), np.std(emd_time), np.mean(emd_loss), np.std(emd_loss)))
    print("*********test on gpu************")  
    print("sinkhorn-gpu took {}({}) seconds with loss {}({})".format(np.mean(sink_gpu_time), np.std(sink_gpu_time), np.mean(sink_gpu_loss), np.std(sink_gpu_loss)))
    print("push-relabel-gpu took {}({}) seconds with loss {}({}), iter {}({})".format(np.mean(pl_gpu_time), np.std(pl_gpu_time), np.mean(pl_gpu_loss), np.std(pl_gpu_loss), np.mean(pl_gpu_iter), np.std(pl_gpu_iter)))

    cur_bench_summary_mean = pd.Series([reg, np.mean(pl_gpu_delta_choose), np.mean(emd_time), np.mean(sink_gpu_time), np.mean(pl_gpu_time), np.mean(emd_loss), np.mean(sink_gpu_loss), np.mean(pl_gpu_loss), np.mean(pl_gpu_iter)], index=col)
    cur_bench_suuumary_std = pd.Series([reg, np.std(pl_gpu_delta_choose), np.std(emd_time), np.std(sink_gpu_time), np.std(pl_gpu_time), np.std(emd_loss), np.std(sink_gpu_loss), np.std(pl_gpu_loss), np.std(pl_gpu_iter)], index=col)
    bench_df = bench_df.append(cur_bench_summary_mean, ignore_index=True)
    bench_df = bench_df.append(cur_bench_suuumary_std, ignore_index=True)
    bench_df.to_csv('pl_vs_sink_bench_results_{}_{}_rev.csv'.format(dataset_name, nlp_name), index=False)