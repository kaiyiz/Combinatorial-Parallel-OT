import numpy as np
import torch

def matching_cpu(W, C, delta):
    """
    This function computes an additive approximation of the bipartite matching between two discrete distributions.
    This function is a 100% CPU-based implementation of the push-relabel algorithm proposed in our paper.

    Parameters
    ----------
    W : ndarray
        A n by n cost matrix, each i and j represent the cost between ith type b and jth type a vertex.
    C : scalar
        The scale of cost metric, max value in of W.
    delta : scalar
        The scaling factor of cost metric.
    
    Returns
    -------
    Mb : ndarray
        A 1 by n array, each i represents the index of type a vertex matching with ith type b vertex.
    yA : ndarray
        A 1 by n array, each i represents the final dual value of ith type a vertex.
    yB : ndarray
        A 1 by n array, each i represents the final dual value of ith type b vertex.
    total_cost : scalar
        Total cost of the final matching.
    iteration : scalar
        The number of iterations ran in while loop when this function finishes
    """
    n = W.shape[0]
    S = (4*W//(delta)).astype(int) 
    cost = (4*W//(delta)).astype(int)
    yB = np.ones(n, dtype=int)
    yA = np.zeros(n, dtype=int)
    Mb = np.ones(n, dtype=int) * -1
    Ma = np.ones(n, dtype=int) * -1
    f = n
    iteration = 0

    while f > n*delta/C:
        Mb_gpu = Mb
        ind_b_free = np.where(Mb_gpu==-1) 
        ind_S_zero = np.where(S[ind_b_free]==0)
        ind_b_free_cpu = ind_b_free
        ind_S_zero_cpu = ind_S_zero
        ind_b_not_visited = np.full(n, True, dtype=bool) # boolean array
        ind_a_not_visited = np.full(n, True, dtype=bool)
        edges_released = ([],[])
        edges_pushed = ([],[])
        ind_b_not_pushed = ([])

        cur_S_zero_pt = 0
        for ind_b_tent in ind_b_free_cpu[0]:
            pushed = False
            while(cur_S_zero_pt < len(ind_S_zero_cpu[0]) and ind_b_tent == ind_b_free[0][ind_S_zero_cpu[0][cur_S_zero_pt]]):
                ind_a_tent = ind_S_zero_cpu[1][cur_S_zero_pt]
                cur_S_zero_pt += 1
                if ind_b_not_visited[ind_b_tent] and ind_a_not_visited[ind_a_tent]:
                    pushed = True
                    if(Ma[ind_a_tent] == -1):
                        f -= 1
                    else:
                        Mb[Ma[ind_a_tent]] = -1
                        edges_released[0].append(Ma[ind_a_tent])
                        edges_released[1].append(ind_a_tent)
                    edges_pushed[0].append(ind_b_tent)
                    edges_pushed[1].append(ind_a_tent)
                    ind_b_not_visited[ind_b_tent] = False
                    ind_a_not_visited[ind_a_tent] = False
                    Ma[ind_a_tent] = ind_b_tent
                    Mb[ind_b_tent] = ind_a_tent
                    yA[ind_a_tent] -= 1
            if not pushed:
                yB[ind_b_tent] += 1
                ind_b_not_pushed.append(ind_b_tent)

        edges_released_gpu = edges_released
        edges_pushed_gpu = edges_pushed
        ind_b_not_pushed_gpu = ind_b_not_pushed
        S[edges_released_gpu] += 1
        S[edges_pushed_gpu] -= 1
        S[:,edges_pushed_gpu[1]] += 1
        S[ind_b_not_pushed_gpu, :] -= 1
        iteration += 1

    ind_a = 0
    for ind_b in range(n):
        if Mb[ind_b] == -1:
            while Ma[ind_a] != -1:
                ind_a += 1
            Mb[ind_b] = ind_a
            Ma[ind_a] = ind_b

    total_cost = 0
    for ind_b in range(n):
        total_cost += W[ind_b, Mb[ind_b]]
    return Mb, yA, yB, total_cost, iteration

def matching_gpu(W, W_cpu, C, delta, device):
    """
    This function computes an additive approximation of the bipartite matching between two discrete distributions.
    This function is a GPU speed-up implementation of the push-relabel algorithm proposed in our paper.

    Parameters
    ----------
    W : tensor
        A n by n cost matrix, each i and j represent the cost between ith type b and jth type a vertex.
    W_cpu : ndarray
        A n by n cost matrix stored in memory for CPU use, each i and j represents the cost between ith type b and jth type a vertex.
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
    device2 = torch.device("cpu")
    dtyp = torch.int32

    n = W.shape[0]

    S = (3*W/(delta)).type(dtyp).to(device)
    yB = np.ones(n, dtype=int)
    yA = np.zeros(n, dtype=int)
    Mb = np.ones(n, dtype=int) * -1
    Ma = np.ones(n, dtype=int) * -1

    range_n = np.arange(n)

    f = n
    iteration = 0

    ng = torch.tensor([n], device=device)[0]
    ones_ = torch.ones(ng, device=device)

    n = W.shape[0]
    zero = torch.tensor([0], device=device, dtype=dtyp, requires_grad=False)[0]
    one = torch.tensor([1], device=device, dtype=dtyp, requires_grad=False)[0]
    m_one = torch.tensor([-1], device=device, dtype=dtyp, requires_grad=False)[0]
    m_two = torch.tensor([-2], device=device, dtype=dtyp, requires_grad=False)[0]

    while f > n*delta/C:
        Mb_gpu = torch.tensor(Mb, device=device) 
        ind_b_free = torch.where(Mb_gpu == m_one)
        ind_S_zero = torch.where(S[ind_b_free] == zero)

        ind_b_free_cpu = ind_b_free[0].cpu().numpy()
        ind_S_zero_cpu = (ind_S_zero[0].cpu().numpy(),ind_S_zero[1].cpu().numpy())

        ind_b_not_visited = np.full(n, True, dtype=bool) # boolean array
        ind_a_not_visited = np.full(n, True, dtype=bool)
        edges_released = ([],[])
        edges_pushed = ([],[])
        ind_b_not_pushed = ([])

        cur_S_zero_pt = 0
        for ind_b_tent in ind_b_free_cpu:
            pushed = False
            while(cur_S_zero_pt < len(ind_S_zero_cpu[0]) and ind_b_tent == ind_b_free_cpu[ind_S_zero_cpu[0][cur_S_zero_pt]]):
                ind_a_tent = ind_S_zero_cpu[1][cur_S_zero_pt]
                cur_S_zero_pt += 1
                if ind_b_not_visited[ind_b_tent] and ind_a_not_visited[ind_a_tent]:
                    pushed = True
                    if(Ma[ind_a_tent] == -1):
                        f -= 1
                    else:
                        Mb[Ma[ind_a_tent]] = -1
                        edges_released[0].append(Ma[ind_a_tent])
                        edges_released[1].append(ind_a_tent)
                    edges_pushed[0].append(ind_b_tent)
                    edges_pushed[1].append(ind_a_tent)
                    ind_b_not_visited[ind_b_tent] = False
                    ind_a_not_visited[ind_a_tent] = False
                    Ma[ind_a_tent] = ind_b_tent
                    Mb[ind_b_tent] = ind_a_tent
                    yA[ind_a_tent] -= 1
            if not pushed:
                yB[ind_b_tent] += 1
                ind_b_not_pushed.append(ind_b_tent)

        edges_released_gpu = torch.tensor(edges_released, dtype = torch.long, device=device)
        edges_pushed_gpu = torch.tensor(edges_pushed, dtype = torch.long, device=device)
        ind_b_not_pushed_gpu = torch.tensor(ind_b_not_pushed, dtype = torch.long, device=device)


        S[edges_released_gpu[0], edges_released_gpu[1]] += one
        S[edges_pushed_gpu[0],edges_pushed_gpu[1]] -= one
        S[:,edges_pushed_gpu[1]] += one
        S[ind_b_not_pushed_gpu, :] -= one
        iteration += 1

    ind_a = 0
    for ind_b in range(n):
        if Mb[ind_b] == -1:
            while Ma[ind_a] != -1:
                ind_a += 1
            Mb[ind_b] = ind_a
            Ma[ind_a] = ind_b

    total_cost = 0
    for ind_b in range(n):
        total_cost += W_cpu[ind_b, Mb[ind_b]]
    return Mb, yA, yB, total_cost, iteration