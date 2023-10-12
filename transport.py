import numpy as np
import torch

from scipy.spatial.distance import cdist
import time
from line_profiler import LineProfiler

def flow_validation(F, f, FreeA, FreeA_ori, FreeB, FreeB_ori):
    # check if flow is valid
    if (FreeA < 0).any():
        print("negative FreeA")
    if (FreeB < 0).any():
        print("negative FreeB")
    if (F < 0).any():
        print("negative flow")    
    if (torch.sum(F,0) + FreeA != FreeA_ori).any():
        print("flow not valid for type A")
    if (torch.sum(F,1) + FreeB != FreeB_ori).any():
        print("flow not valid for type B")
    if f != torch.sum(FreeB):
        print("flow not valid for free type B")

def flow_validation_final(F, FreeA, FreeA_ori, FreeB, FreeB_ori, device):
    # check if flow is valid
    if (FreeA < -1e-9).any():
        print("negative FreeA")
    if (FreeB < -1e-9).any():
        print("negative FreeB")
    if (F < 0).any():
        print("negative flow")  
    # torch.testing.assert_close(torch.sum(F,0) + FreeA - FreeA_ori, torch.zeros(FreeA.shape, dtype=torch.float64), msg="Flow not valid for type A!")
    # torch.testing.assert_close(torch.sum(F,1) + FreeB - FreeB_ori, torch.zeros(FreeB.shape, dtype=torch.float64), msg="Flow not valid for type B!")
    torch.testing.assert_close(torch.sum(F,0) - FreeA_ori, torch.zeros(FreeA.shape, dtype=torch.float64, device = device), msg="Flow not valid for type A!")
    torch.testing.assert_close(torch.sum(F,1) - FreeB_ori, torch.zeros(FreeB.shape, dtype=torch.float64, device = device), msg="Flow not valid for type B!")

def feasibilty_validation(yFA, yB, yA, F, C):
    # check feasibility
    zero_f_ind = torch.where(F == 0)
    nonzero_f_ind = torch.where(F > 0)
    if len(zero_f_ind[0]) > 0 and (yA[zero_f_ind[1]] + yB[zero_f_ind[0]] > C[zero_f_ind] + 1).any():
        print("first feasibility condition not valid (F=0)")
    if len(nonzero_f_ind[0]) > 0 and (yFA[nonzero_f_ind] + yB[nonzero_f_ind[0]] > C[nonzero_f_ind] + 1).any():
        print("first feasibility condition not valid (F>0)")
    if len(nonzero_f_ind[0]) > 0 and (yFA[nonzero_f_ind] + yB[nonzero_f_ind[0]] < C[nonzero_f_ind]).any():
        print("second feasibility condition not valid")

def slack_validation(yB, yA, S, C):
    # check slack
    # Only need to check S with yA and yB
    if (S != C + 1 - yB[:,None] - yA[None, :]).any():
        print("slack not valid")

def rand_inputs(n = 100, seed = 0):
    """
    This function creates synthetic experiment data by randomly generating points in a 2d unit square.
    """
    np.random.seed(seed)
    a = np.random.rand(n,2)
    b = np.random.rand(n,2)
    cost = cdist(a, b, 'sqeuclidean')
    DA = np.random.rand(n)
    SB = np.random.rand(n)
    return DA, SB, cost

def unique2(x, input_sorted = False):
    """""
    Returns the unique elements of array x, and the indices of the first occurrences of the unique values in the original array
    Method 2
    """""
    unique, inverse_ind, unique_count = torch.unique(x, return_inverse=True, return_counts=True)
    unique_ind = unique_count.cumsum(0)
    if not unique_ind.size()[0] == 0:
        unique_ind = torch.cat((torch.tensor([0], dtype=x.dtype, device=x.device), unique_ind[:-1]))
    if not input_sorted:
        _, sort2ori_ind = torch.sort(inverse_ind, stable=True)
        unique_ind = sort2ori_ind[unique_ind]
    return unique, unique_ind

def subset_sum_filter(F, sum, dim=0):
    F_cum = torch.cumsum(F, dim)
    F_cum_clamp = torch.clamp(F_cum, max=sum)
    F = F - (F_cum - F_cum_clamp)
    F_mask = F.lt(0)
    F_mask_full = (F_cum.le(sum) & F.gt(0))
    F[F_mask] = 0
    return F, F_mask_full


def transport_torch(DA, SB, C, delta, device):
    """
    This function computes an additive approximation of optimal transport between two discrete distributions.
    This function is a GPU speed-up implementation of the push-relabel algorithm proposed in our paper.

    Parameters
    ----------
    DA : ndarray
        A n by 1 array, each DA(i) represent the mass of demand on ith type a vertex. The sum of DA should equal to 1.
    SB : ndarray
        A n by 1 array, each SB(i) represent the mass of supply on ith type b vertex. The sum of SB should equal to 1.
    C : tensor
        A n by n cost matrix, each i and j represent the cost between ith type b and jth type a vertex.
    delta : tensor
        The scaling factor (scalar) of cost metric. The value of epsilon in paper. 
    
    Returns
    -------
    F, yA, yB, total_cost, iteration
    F : tensor
        A n by n matrix, each i and j represents the flow between ith type b and jth type a vertex.
    yA : tensor
        A 1 by n array, each i represents the final dual value of ith type a vertex.
    yB : tensor
        A 1 by n array, each i represents the final dual value of ith type b vertex.
    total_cost : tensor
        The total cost of the final transport solution.
    iteration : tensor
        The number of iterations ran in while loop when this function finishes.
    """
    torch.manual_seed(0)
    dtyp = torch.int32
    zero = torch.tensor([0], device=device, dtype=dtyp, requires_grad=False)[0]
    one = torch.tensor([1], device=device, dtype=dtyp, requires_grad=False)[0]
    m_one = torch.tensor([-1], device=device, dtype=dtyp, requires_grad=False)[0]

    m = torch.tensor(C.shape[0], device=device, requires_grad=False)
    n = torch.tensor(C.shape[1], device=device, requires_grad=False)
    yA = torch.zeros(n, dtype=dtyp, device=device) # smaller dual weight of type A vertex in absolute value
    yB = torch.ones(m, dtype=dtyp, device=device) # dual weight of type B vertex

    F = torch.zeros(C.shape, device=device, dtype=dtyp, requires_grad=False)
    yFA = torch.full(C.shape, one, device=device, dtype=dtyp, requires_grad=False) # dual weight of type A vertice that matched to certain type B vertice, default value torch.iinfo.min when no matching
    S = torch.div((3*C), delta, rounding_mode='trunc').type(dtyp).to(device)
    # C_scaled = torch.div((3*C), delta, rounding_mode='trunc').type(dtyp).to(device) # scaled cost for feasibility validation

    max_C = torch.max(C)
    alpha = 6 * n * max_C / delta
    FreeA_ = DA * alpha 
    FreeA = torch.ceil(FreeA_).to(dtyp)
    FreeA_ori = FreeA.clone()

    FreeB_ = SB * alpha 
    FreeB = FreeB_.to(dtyp)
    FreeB_ori = FreeB.clone()
    f = torch.sum(FreeB) #flow remaining to push
    ff = 0
    iteration = 0

    # main loop
    while f > n:
        # extract admissiable graph
        ind_b_free = torch.where(FreeB > zero)
        ind_zero_slack_ind = torch.where(S[ind_b_free[0],:]==0)
        # find push-release edges and corresponding flow
        # find push edges
        ind_b_push_tent_ind, ind_a_tent_lt_inclusive = unique2(ind_zero_slack_ind[0], input_sorted=True)
        ind_b_push_tent = ind_b_free[0][ind_b_push_tent_ind] #tentative B (push)
        ind_a_tent_rt_exclusive = torch.cat((ind_a_tent_lt_inclusive[1:], torch.tensor(ind_zero_slack_ind[0].shape, device=device, dtype=dtyp, requires_grad=False)))
        rand_b = torch.rand(ind_b_push_tent.shape[0], device=device)
        ind_a_ind = ind_a_tent_lt_inclusive + ((ind_a_tent_rt_exclusive - ind_a_tent_lt_inclusive)*rand_b).to(dtyp)
        ind_a = ind_zero_slack_ind[1][ind_a_ind] #tentative A push (maybe release)
        ind_a_push, ind_b_push_ind = unique2(ind_a, input_sorted=False) #find exact a to push, and corresponding index
        ind_b_push = ind_b_push_tent[ind_b_push_ind] #final type b vertex to push
        edge_push = (ind_b_push, ind_a_push)
        ind_b_no_push = ind_b_free[0][(ind_b_free[0][:, None] != ind_b_push_tent).all(dim=1)]
        # calculate the flow that push to free copies of A
        push_flow_free, unsatisfied_vertices_ind = torch.min(torch.vstack((FreeB[ind_b_push], FreeA[ind_a_push])),0)
        # find push release edges B->A and corresponding vertices A
        ind_release_ind = torch.where(FreeA[ind_a_push] == 0)
        ind_b_push_release = ind_b_push[ind_release_ind]
        ind_a_push_release = ind_a_push[ind_release_ind]
        edge_push_released = (ind_b_push_release, ind_a_push_release)
        ind_a_release = ind_a_push_release
        # find release edges A->B
        release_edge_tent_ind = torch.where(torch.t(yFA[:,ind_a_release])==yA[ind_a_release][:,None]) # type a sorted, index 0
        ind_a_release, ind_b_release_tent_lt_inclusive = unique2(ind_a_release[release_edge_tent_ind[0]], input_sorted=True)
        ind_b_release_tent_rt_exclusive = torch.cat((ind_b_release_tent_lt_inclusive[1:], torch.tensor(release_edge_tent_ind[1].shape, device=device, dtype=dtyp, requires_grad=False)))
        rand_a = torch.rand(ind_a_release.shape[0], device=device)
        ind_b_ind = ind_b_release_tent_lt_inclusive + ((ind_b_release_tent_rt_exclusive - ind_b_release_tent_lt_inclusive)*rand_a).to(dtyp)
        ind_b_release = release_edge_tent_ind[1][ind_b_ind]
        edge_release = (ind_b_release, ind_a_release)

        FreeB[ind_b_push] -= push_flow_free
        # calculate the release flow and find the full released edges A->B
        push_flow_release, part_release_ind = torch.min(torch.vstack((F[edge_release], FreeB[ind_b_push_release])),0)
        part_release_ind = part_release_ind.to(bool)
        ind_full_release_ind = ~part_release_ind
        ind_a_full_release = ind_a_release[ind_full_release_ind]
        ind_b_full_release = ind_b_release[ind_full_release_ind]
        edge_full_release = (ind_b_full_release, ind_a_full_release)

        # update transport configuration (flow/slack/dual weight)
        # update flow and free vertiecs
        f -= torch.sum(push_flow_free)
        ff += torch.sum(push_flow_free).item()
        F[edge_push] += push_flow_free
        F[edge_push_released] += push_flow_release
        F[edge_release] -= push_flow_release
        FreeA[ind_a_push] -= push_flow_free
        FreeB[ind_b_push_release] -= push_flow_release
        FreeB.index_add_(0, ind_b_release, push_flow_release)
        # dual weight/slack
        # update dual weight and slack of type b vertices that not able be pushed at current iteration
        yB[ind_b_no_push] += one
        S[ind_b_no_push, :] -= one
        b_no_push_edge_w_flow = torch.where(F[ind_b_no_push,:]!=0)
        b_no_push_edge_w_flow = (ind_b_no_push[b_no_push_edge_w_flow[0]], b_no_push_edge_w_flow[1])
        yFA[b_no_push_edge_w_flow] = yA[b_no_push_edge_w_flow[1]] - one
        # update edge-wise dual weight of type A vertices that are pushed/released at current iteration
        yFA[edge_push] = yA[ind_a_push] - one # pushed edge yFA = yA - 1
        yFA[edge_full_release] = one # set dual weight yFA to default value if an edge is fully released
        # update dual weight of type a vertices that is exhausted at current iteration
        ind_a_exhausted_tent = torch.unique(torch.cat((ind_a_push, b_no_push_edge_w_flow[1])))
        ind_a_push_not_free = ind_a_exhausted_tent[FreeA[ind_a_exhausted_tent] == 0]
        yFA_mask = (yA[ind_a_push_not_free] > yFA[:,ind_a_push_not_free])
        yFA_mask[yFA[:,ind_a_push_not_free] == one] = True
        ind_a_exhausted_ind = yFA_mask.all(dim=0)
        ind_a_exhausted = torch.masked_select(ind_a_push_not_free, ind_a_exhausted_ind)
        yA[ind_a_exhausted] -= one
        S[:,ind_a_exhausted] += one

        iteration += 1

    # reverse scaling
    scaling_error_A = FreeA_ori - FreeA_
    scaling_error_B = FreeB_ - FreeB_ori
    F = F/alpha
    ind_a_all_transported_after_scaling = torch.where(FreeA==0)
    FreeA = (FreeA - scaling_error_A)/alpha
    FreeA[ind_a_all_transported_after_scaling] = 0
    FreeB = (FreeB + scaling_error_B)/alpha
    reverse_edges_ = torch.where(torch.t(F[:,ind_a_all_transported_after_scaling[0]])!=0)
    _, reverse_edges_B_ind = unique2(reverse_edges_[0], input_sorted=True)
    reverse_edges = (reverse_edges_[1][reverse_edges_B_ind], ind_a_all_transported_after_scaling[0])
    reverse_flow = scaling_error_A[reverse_edges[1]]/alpha
    F[reverse_edges] -= reverse_flow
    FreeB.index_add_(0, reverse_edges[0], reverse_flow)

    f_left = torch.sum(FreeB)
    while f_left > 1e-12:
        ind_b_left = torch.where(FreeB > 0)[0]
        ind_a_left = torch.where(FreeA > 0)[0]
        rand_b = torch.rand(ind_b_left.shape[0], device=device)
        ind_a_left_push_ind = (rand_b * ind_a_left.shape[0]).to(dtyp).long()
        ind_a_left_push = ind_a_left[ind_a_left_push_ind]
        F_ = torch.zeros(F.shape, dtype=torch.float64, device=device)
        F_[(ind_b_left, ind_a_left_push)] = torch.min(FreeB[:,None], FreeA[None,:])[(ind_b_left, ind_a_left_push)]
        ind_a_left_push_unique = torch.unique(ind_a_left_push)
        F_=F_[ind_b_left,:][:,ind_a_left_push_unique]
        F_, _ = subset_sum_filter(F_, FreeA[ind_a_left_push_unique], dim = 0)
        edge_push = torch.where(F_ > 0)
        push_flow_left = F_[edge_push]
        edge_push = (ind_b_left[edge_push[0]], ind_a_left_push_unique[edge_push[1]])
        F[edge_push] += push_flow_left
        FreeB[edge_push[0]] -= push_flow_left
        FreeA.index_add_(0,edge_push[1],m_one*push_flow_left)
        f_left -= torch.sum(push_flow_left)

    # flow_validation_final(F, FreeA, DA, FreeB, SB, device)

    total_cost = torch.sum(F*C)
    return F, yA, yB, total_cost, iteration