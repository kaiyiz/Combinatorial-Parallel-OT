"""
This is a script that conducts the experiments shown in our paper. 
A GPU based implementation of our algorithm is compared with DROT methods (step 2).
"""
import argparse
import numpy as np
import pandas as pd
import os
import time
from load_data import load_DA_SB_cost

from drot_py_ import drot_

def find_maxiters(cost, p, q, m, n, stepsize, target_cost, eps=0.000000, max_iters=100000, filename=None, threshold=0.0000001):
    # find the maxiters that can make the cost close to target_cost
    low = 1
    high = max_iters
    best_iters = 0
    best_cost = float('inf')
    eps = 0.000000
    verbose = False
    log = False
    pre_mid = 0
    while low <= high:
        mid = (low + high) // 2
        if mid % 2 == 0:
            mid += 1
        X = drot_(cost, p, q, m, n, stepsize, mid, eps, verbose, log, filename)
        X_feasible = make_transport_plan_feasible(X, p, q)
        cost_feasible = np.sum(np.multiply(X_feasible, cost))
        print("try iters: ", mid, "mass_X: ", np.sum(X), "target_cost: ", target_cost, "cost_feasible: ", cost_feasible, "best_cost: ", best_cost)

        if mid == pre_mid:
            return mid

        if abs(target_cost -  cost_feasible) < threshold: 
            return mid
        
        if cost_feasible < target_cost:
            high = mid - 1
        else:
            if cost_feasible < best_cost:
                best_cost = cost_feasible
                best_iters = mid
            low = mid + 1
        pre_mid = mid

    return best_iters

def make_transport_plan_feasible(X, p, q):
    # Ensure that the total transport plan is feasible
    # Calculate row and column sums
    row_sums = X.sum(axis=1)
    col_sums = X.sum(axis=0)

    # Calculate the surplus and deficit of supply and demand
    p_surplus = np.maximum(q - row_sums, 0)
    q_deficit = np.maximum(p - col_sums, 0)

    # Calculate the additional transport matrix to balance supply and demand
    additional_transport = np.outer(p_surplus, q_deficit) / p_surplus.sum()

    # Ensure that the total transport plan is feasible
    X_feasible = X + additional_transport

    return X_feasible


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
print("-------------------------------------step2-------------------------------------")
print(args)

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
final_cost_filename = f"./pr_cost_drot_{NUM_EXPERIMENTS}_{n}_{norm_cost}_{dataset_name}_{is_transport}_{scale_factor}_{metric}_{delta_num}_{delta_low}_{delta_high}_{nlp_name}_{nlp_portion_size}.npy"
final_iter_filename = f"./pr_iter_drot_{NUM_EXPERIMENTS}_{n}_{norm_cost}_{dataset_name}_{is_transport}_{scale_factor}_{metric}_{delta_num}_{delta_low}_{delta_high}_{nlp_name}_{nlp_portion_size}.npy"
final_time_filename = f"./pr_time_drot_{NUM_EXPERIMENTS}_{n}_{norm_cost}_{dataset_name}_{is_transport}_{scale_factor}_{metric}_{delta_num}_{delta_low}_{delta_high}_{nlp_name}_{nlp_portion_size}.npy"
bench_res_mean = []
bench_res_std = []
col = ['delta', 'drot_iter', 'pl_gpu_iter', 'drot_gpu_time', 'pl_gpu_time', 'drot_gpu_loss', 'pl_gpu_loss']
bench_df = pd.DataFrame(columns=col)
pl_gpu_loss_all = np.load(final_cost_filename)
pl_gpu_iter_all = np.load(final_iter_filename)
pl_gpu_time_all = np.load(final_time_filename)

ind_ot_pyt = 0

for delta in delta_tryouts:
    drot_gpu_time = []
    drot_gpu_loss = []
    drot_iter_choose = []
    pl_gpu_time = []
    pl_gpu_iter = []
    pl_gpu_loss = []

    for i in range(NUM_EXPERIMENTS):
        DA, SB, cost = load_DA_SB_cost(dataset_name, n=n, norm_cost=norm_cost, metric = metric, scale_factor = scale_factor, seed=i, nlp_i=i+1, nlp_name=nlp_name, nlp_portion_size=nlp_portion_size)
        C = cost.max()
        target_cost = pl_gpu_loss_all[ind_ot_pyt]
        pl_gpu_time.append(pl_gpu_time_all[ind_ot_pyt])
        pl_gpu_iter.append(pl_gpu_iter_all[ind_ot_pyt])
        pl_gpu_loss.append(target_cost)
        ind_ot_pyt += 1

        ######test on drot###########
        m = len(DA)
        n = len(SB)
        filename = f"./output/drot_{dataset_name}_{nlp_name}_{max(m,n)}_{i}.csv"
        stepsize = 2 / (n + m)
        maxiters = find_maxiters(cost, DA, SB, m, n, stepsize, target_cost, filename=filename)
        start = time.time()
        X = drot_(cost, DA, SB, m, n, stepsize, maxiters, 0.0, False, False, '')
        end = time.time()
        X_feasible = make_transport_plan_feasible(X, DA, SB)
        drot_fval = np.sum(X_feasible * cost)
        drot_time = end - start
        # print("drot loss = {}".format(drot_fval))
        # print("drot takes time = {}s".format(drot_time))
        drot_gpu_time.append(drot_time)
        drot_gpu_loss.append(drot_fval)
        drot_iter_choose.append(maxiters)


    print("problem size {} * {}".format(len(DA), len(SB)))
    print("delta in pr={}".format(delta)) 
    print("drot-gpu took {}({}) seconds with loss {}({})".format(np.mean(drot_gpu_time), np.std(drot_gpu_time), np.mean(drot_gpu_loss), np.std(drot_gpu_loss)))
    print("pl-gpu took {}({}) seconds with loss {}({})".format(np.mean(pl_gpu_time), np.std(pl_gpu_time), np.mean(pl_gpu_loss), np.std(pl_gpu_loss)))

    cur_bench_summary_mean = pd.Series([delta, np.mean(drot_iter_choose), np.mean(pl_gpu_iter), np.mean(drot_gpu_time), np.mean(pl_gpu_time), np.mean(drot_gpu_loss), np.mean(pl_gpu_loss)], index=col)
    cur_bench_summary_std = pd.Series([delta, np.std(drot_iter_choose), np.std(pl_gpu_iter), np.std(drot_gpu_time), np.std(pl_gpu_time), np.std(drot_gpu_loss), np.std(pl_gpu_loss)], index=col)
    bench_df = bench_df.append(cur_bench_summary_mean, ignore_index=True)
    bench_df = bench_df.append(cur_bench_summary_std, ignore_index=True)
    bench_df.to_csv('pl_vs_drot_bench_results_{}_{}.csv'.format(dataset_name, nlp_name), index=False)