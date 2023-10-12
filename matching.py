import numpy as np
import torch
from scipy.spatial.distance import cdist

def feasibility_check(ind_b, ind_a, yB, yA, Ma, cost):
    # check feasibility
    if yA[ind_a] + yB[ind_b] > cost[ind_b][ind_a] + 1:
                print("assertion failed")
    if Ma[ind_a] == ind_b and yA[ind_a] + yB[ind_b] != cost[ind_b][ind_a]:
                print("assertion failed in matching")
                print(yA[ind_a] + yB[ind_b])
                print(cost[ind_b][ind_a])

def matching_check(Ma, Mb):
    # check matching
    if np.size(np.where(Mb == -1)) != 0:
        print("exist empty matching")
    if np.size(np.where(Ma == -1)) != 0:
        print("exist empty matching")
    for ind_b in range(Mb.shape[0]):
        ind_a = Mb[ind_b]
        if ind_a != -1 and ind_b != Ma[ind_a]:
            print("mismatching")
    for ind_a in range(Ma.shape[0]):
        ind_b = Ma[ind_a]
        if ind_b != -1 and ind_a != Mb[ind_b]:
            print("mismatching")

def matching_check_torch(Ma, Mb):
    # check matching
    if torch.size(torch.where(Mb == -1)) != 0:
        print("exist empty matching")
    if torch.size(torch.where(Ma == -1)) != 0:
        print("exist empty matching")
    for ind_b in range(Mb.shape[0]):
        ind_a = Mb[ind_b]
        if ind_a != -1 and ind_b != Ma[ind_a]:
            print("mismatching")
    for ind_a in range(Ma.shape[0]):
        ind_b = Ma[ind_a]
        if ind_b != -1 and ind_a != Mb[ind_b]:
            print("mismatching")

def matching_torch_v1(W, C, delta, device, seed=1):
    """
    This function computes an additive approximation of the bipartite matching between two discrete distributions.
    This function is a GPU speed-up implementation of the push-relabel algorithm proposed in our paper.

    Parameters
    ----------
    W : tensor
        A n by n cost matrix, each i and j represent the cost between ith type b and jth type a vertex.
    C : tensor
        The scale of cost metric, max value in of W.
    delta : tensor
        The scaling factor of cost metric.
    
    Returns
    -------
    Mb : tensor
        A 1 by n array, each i represents the index of type a vertex matching with ith type b vertex.
    yA : tensor
        A 1 by n array, each i represents the final dual value of ith type a vertex.
    yB : tensor
        A 1 by n array, each i represents the final dual value of ith type b vertex.
    total_cost : tensor
        The total cost of the final matching.
    iteration : tensor
        The number of iterations ran in while loop when this function finishes.
    """
    dtyp = torch.int64
    n = W.shape[1]
    m = W.shape[0]

    S = (3*W/(delta)).type(dtyp).to(device)
    # cost = (3*W/(delta)).type(dtyp).to(device) # scaled cost for feasibility validation
    yB = torch.ones(m, device=device, dtype=dtyp, requires_grad=False)
    yA = torch.zeros(n, device=device, dtype=dtyp, requires_grad=False)
    Mb = torch.ones(m, device=device, dtype=dtyp, requires_grad=False) * -1
    Ma = torch.ones(n, device=device, dtype=dtyp, requires_grad=False) * -1

    f = n
    iteration = 0

    n = W.shape[0]
    zero = torch.tensor([0], device=device, dtype=dtyp, requires_grad=False)[0]
    one = torch.tensor([1], device=device, dtype=dtyp, requires_grad=False)[0]
    m_one = torch.tensor([-1], device=device, dtype=dtyp, requires_grad=False)[0]

    f_threshold = n*delta/C

    torch.manual_seed(1)
    while f > f_threshold:
        ind_b_free = torch.where(Mb == m_one)
        ind_S_zero_ind = torch.where(S[ind_b_free] == zero)

        # find push edges
        ind_b_tent_ind, free_S_edge_B_ind_range_lt_inclusive = unique(ind_S_zero_ind[0], input_sorted=True)
        ind_b_tent = ind_b_free[0][ind_b_tent_ind]
        free_S_edge_B_ind_range_rt_exclusive = torch.cat((free_S_edge_B_ind_range_lt_inclusive[1:], torch.tensor(ind_S_zero_ind[0].shape, device=device, dtype=dtyp, requires_grad=False))) #right index of B range
        rand_n = torch.rand(ind_b_tent.shape[0], device=device)
        free_S_edge_B_ind_rand = free_S_edge_B_ind_range_lt_inclusive + ((free_S_edge_B_ind_range_rt_exclusive - free_S_edge_B_ind_range_lt_inclusive)*rand_n).to(dtyp)
        ind_a_tent = ind_S_zero_ind[1][free_S_edge_B_ind_rand] #get tentative A to push
        ind_a_push, tent_ind = unique(ind_a_tent, input_sorted=False) #find exact a to push, and corresponding index
        ind_b_push = ind_b_tent[tent_ind] #find exact b to push
        # find release edges
        ind_release = torch.nonzero(Ma[ind_a_push] != -1, as_tuple=True)[0]
        edges_released = (Ma[ind_a_push][ind_release], ind_a_push[ind_release])
        # update flow
        f -= len(ind_a_push)-len(ind_release) 
        # release edges
        Mb[Ma[edges_released[1]]] = m_one
        # push edges
        edges_pushed = (ind_b_push, ind_a_push)
        Ma[ind_a_push] = ind_b_push
        Mb[ind_b_push] = ind_a_push
        yA[ind_a_push] -= one
        # find b that not able to be pushed
        min_slack, _ = torch.min(S[ind_b_free[0],:], dim=1)
        min_slack_ind = torch.where(min_slack!=0)[0]
        ind_b_not_pushed = ind_b_free[0][min_slack_ind]
        yB[ind_b_not_pushed] += min_slack[min_slack_ind]
        #update slack
        S[edges_released] += one
        S[edges_pushed] -= one
        S[:,edges_pushed[1]] += one
        S[ind_b_not_pushed, :] -= min_slack[min_slack_ind][:,None]
        iteration += 1
    
    yA = yA.cpu().detach()   
    yB = yB.cpu().detach()
    Ma = Ma.cpu().detach()
    Mb = Mb.cpu().detach()
    
    ind_a = 0
    for ind_b in range(n):
        if Mb[ind_b] == -1:
            while Ma[ind_a] != -1:
                ind_a += 1
            Mb[ind_b] = ind_a
            Ma[ind_a] = ind_b
    
    # matching_check(Ma, Mb) # check the validity of the matching
    matching_cost = torch.sum(W[torch.arange(0,n,dtype=torch.int64),Mb])
    return Mb, yA, yB, matching_cost, iteration

def unique(x, input_sorted = False):
    """""
    Returns the unique elements of array x, and the indices of the first occurrences of the unique values in the original array
    """""
    unique, inverse_ind, unique_count = torch.unique(x, return_inverse=True, return_counts=True)
    unique_ind = unique_count.cumsum(0)
    if not unique_ind.size()[0] == 0:
        unique_ind = torch.cat((torch.tensor([0], dtype=x.dtype, device=x.device), unique_ind[:-1]))
    if not input_sorted:
        _, sort2ori_ind = torch.sort(inverse_ind, stable=True)
        unique_ind = sort2ori_ind[unique_ind]
    return unique, unique_ind

def rand_points(n = 100, seed = 0):
    """
    This function creates synthetic experiment data by randomly generating points in a 2d unit square.
    """
    np.random.seed(seed)
    a = np.random.rand(n,2)
    b = np.random.rand(n,2)
    cost = cdist(a, b, 'sqeuclidean')
    return a, b, cost