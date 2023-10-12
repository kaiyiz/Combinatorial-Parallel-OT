import ot
import numpy as np
import time
import torch

from transport import transport_torch as transport_pure_gpu
from matching import matching_torch_v1 as matching_pure_gpu

def run_sink(a, b, W, reg_mid):
    G = ot.sinkhorn(a, b, W, reg_mid, method='sinkhorn', numItermax=100000000)
    return G

def get_sinkorn_reg(a, b, W, target_loss, d = 1e-6, reg_lt = 1e-6, reg_rt = 1, show_sinkh_time = False):
    """
    This function selects regularization parameter for sinkhorn algorithm using binary searching.
    """
    reg_mid = (reg_lt+reg_rt)/2
    while(True):
        torch.cuda.synchronize()
        st = time.perf_counter()
        G = run_sink(a, b, W, reg_mid)
        cur_loss = torch.sum(G*W)
        end = time.perf_counter()
        if show_sinkh_time:
            print("sinkhorn takes {}s when reg={}".format(end - st, reg_mid))
        d_loss = cur_loss - target_loss
        if(d_loss > 0 and d_loss < d):
            break
        if(reg_rt-reg_lt<=d):
            print("choose boundary value")
            return reg_rt
        if(d_loss > 0):
            reg_rt = reg_mid
            reg_mid = (reg_mid + reg_lt)/2
        else:
            reg_lt = reg_mid
            reg_mid = (reg_mid + reg_rt)/2
    
    return reg_mid

def get_pl_delta_transport(DA_tensor, SB_tensor, cost_tensor, target_loss, device, d = 1e-5, delta_lt = 0.0001, delta_rt = 1, show_pl_time = False):
    delta_rt_index = np.log10(delta_rt)
    delta_lt_index = np.log10(delta_lt)
    num_tryouts = 5
    delta_tryouts = np.flip(np.logspace(delta_lt_index, delta_rt_index, num=num_tryouts))
    for ind, delta in enumerate(delta_tryouts):
        delta_tensor = torch.tensor([delta], device=device, requires_grad=False)
        torch.cuda.synchronize()
        st = time.perf_counter()
        Mb, yA, yB, cur_loss, iteration = transport_pure_gpu(DA_tensor, SB_tensor, cost_tensor, delta_tensor, device=device)
        end = time.perf_counter()
        if show_pl_time:
            print("pl takes {}s when delta={}".format(end - st, delta))
        d_loss = cur_loss.cpu().item() - target_loss
        if d_loss < 0:
            break

    delta_lt = delta_tryouts[ind]
    if ind > 0:
        delta_rt = delta_tryouts[ind-1]
    else:
        return delta_tryouts[ind]
    delta_mid = (delta_lt+delta_rt)/2

    itr_count = 0
    count_limit = 1e3

    while(True):
        delta_tensor = torch.tensor([delta_mid], device=device, requires_grad=False)
        torch.cuda.synchronize()
        st = time.perf_counter()
        Mb, yA, yB, cur_loss, iteration = transport_pure_gpu(DA_tensor, SB_tensor, cost_tensor, delta_tensor, device=device)
        end = time.perf_counter()
        if show_pl_time:
            print("pl takes {}s when delta={}".format(end - st, delta_mid))
        d_loss = cur_loss.cpu().item() - target_loss
        if(d_loss < 0 and d_loss > -d):
            break
        if(delta_rt-delta_lt<=d):
            print("choose boundary value")
            return delta_lt
        if(d_loss > 0):
            delta_rt = delta_mid
            delta_mid = (delta_mid + delta_lt)/2
        else:
            delta_lt = delta_mid
            delta_mid = (delta_mid + delta_rt)/2
        if itr_count > count_limit:
            print("iteration limit reached for searching delta, error {}".format(np.absolute(d_loss)))
            break
    
    return delta_mid

def get_pl_delta_matching(cost_tensor, C_tensor, target_loss, n, device, d = 1e-5, delta_lt = 0.0001, delta_rt = 1, show_pl_time = False):
    delta_rt_index = np.log10(delta_rt)
    delta_lt_index = np.log10(delta_lt)
    num_tryouts = 10
    delta_tryouts = np.flip(np.logspace(delta_lt_index, delta_rt_index, num=num_tryouts))
    ind = 0
    for ind, delta in enumerate(delta_tryouts):
        delta_tensor = torch.tensor([delta], device=device, requires_grad=False)
        torch.cuda.synchronize()
        st = time.perf_counter()
        Mb, yA, yB, cur_loss, iteration = matching_pure_gpu(cost_tensor, C_tensor, delta_tensor, device = device)
        end = time.perf_counter()
        if show_pl_time:
            print("pl takes {}s when delta={}".format(end - st, delta))
        d_loss = cur_loss.cpu().item()/n - target_loss
        if d_loss < 0:
            break

    delta_lt = delta_tryouts[ind]
    if ind > 0:
        delta_rt = delta_tryouts[ind-1]
    else:
        return delta_tryouts[ind]
    delta_mid = (delta_lt+delta_rt)/2

    itr_count = 0
    count_limit = 1e3

    while(True):
        delta_tensor = torch.tensor([delta_mid], device=device, requires_grad=False)
        torch.cuda.synchronize()
        st = time.perf_counter()
        Mb, yA, yB, cur_loss, iteration = matching_pure_gpu(cost_tensor, C_tensor, delta_tensor, device=device)
        end = time.perf_counter()
        if show_pl_time:
            print("pl takes {}s when delta={}".format(end - st, delta_mid))
        d_loss = cur_loss.cpu().item()/n - target_loss
        if(d_loss < 0 and d_loss > -d):
            break
        if(delta_rt-delta_lt<=d):
            print("choose boundary value")
            return delta_lt
        if(d_loss > 0):
            delta_rt = delta_mid
            delta_mid = (delta_mid + delta_lt)/2
        else:
            delta_lt = delta_mid
            delta_mid = (delta_mid + delta_rt)/2
        if itr_count > count_limit:
            print("iteration limit reached for searching delta, error {}".format(np.absolute(d_loss)))
            break
    
    return delta_mid