import numpy as np
import torch
from scipy.spatial.distance import cdist

def transport_tensor(DA, SB, C, delta, device):
    """
    This function computes an additive approximation of optimal transport between two discrete distributions.
    This function is a GPU speed-up implementation of the push-relabel algorithm proposed in our paper.

    Parameters
    ----------
    DA : ndarray
        A n by 1 array, each DA(i) represent the mass of demand on ith type a vertex.
    SB : ndarray
        A n by 1 array, each SB(i) represent the mass of supply on ith type b vertex.
    C : tensor
        A n by n cost matrix, each i and j represent the cost between ith type b and jth type a vertex.
    delta : tensor
        The scaling factor (scalar) of cost metric. The value of epsilon in paper. 
    
    Returns
    -------
    F, yA, yB, total_cost, iteration, zero_slack_length
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
    zero_slack_length: ndarray
        A 1 by n array, zero_slack_length(i) represents the the number of zero slack edges fetched in ith iteration.
    """
    dtyp = torch.int32

    n = C.shape[0]
    yA = torch.zeros(n, dtype=dtyp, device=device)
    yB = torch.ones(n, dtype=dtyp, device=device)
    PushableA = np.zeros((n,2), dtype=int)
    F = torch.zeros(C.shape, device=device, dtype=dtyp, requires_grad=False)
    yFA = torch.zeros(C.shape, device=device, dtype=dtyp, requires_grad=False)
    S = torch.div((4*C),delta, rounding_mode='trunc').type(dtyp).to(device)

    max_C = torch.max(C)
    alpha = (6 * n * max_C / delta).cpu().numpy()
    FreeA_wo_scaling = DA * alpha 
    FreeA = np.ceil(FreeA_wo_scaling).astype(int)
    Err_A = (FreeA - FreeA_wo_scaling)/alpha
    PushableA[:,0] = FreeA
    FreeB_wo_scaling = SB * alpha 
    FreeB = (FreeB_wo_scaling).astype(int)
    Err_B = (FreeB_wo_scaling - FreeB)/alpha
    f = np.sum(FreeB) #flow remaining to push
    iteration = 0
    zero_slack_length = ([])

    zero = torch.tensor([0], device=device, dtype=dtyp, requires_grad=False)[0]
    one = torch.tensor([1], device=device, dtype=dtyp, requires_grad=False)[0]

    # main loop
    while f > n:
        FreeB_tensor = torch.tensor(FreeB, device=device)
        ind_b_free_tensor = torch.squeeze(torch.nonzero(FreeB_tensor>0),1)
        ind_zero_slack_tensor = torch.nonzero(S[ind_b_free_tensor,:]==0, as_tuple=True)
        zero_slack_length.append(len(ind_zero_slack_tensor[0]))
        ind_b_free = ind_b_free_tensor.cpu().numpy()
        ind_zero_slack = (ind_zero_slack_tensor[0].cpu().numpy(),ind_zero_slack_tensor[1].cpu().numpy())
        edges_full_released = ([],[])
        edges_part_released = ([],[])
        edges_pushed = ([],[])
        flow_pushed = ([])
        flow_released_ind_a = np.zeros(n, dtype=int)
        flow_partial_released = ([])
        ind_b_not_exhausted = ([])
        ind_a_exhausted = np.zeros(n, dtype=bool)

        cur_S_zero_pt = 0

        # log pushed edges
        if len(ind_zero_slack[0]) > 0:
            ind_zero_slack_b_range = find_ind_range(ind_zero_slack[0])

        for ind_b_free_index in range(len(ind_b_free)):
            ind_b = ind_b_free[ind_b_free_index]
            b_exhausted = False
            try:
                cur_S_zero_pt = ind_zero_slack_b_range[ind_b_free_index][0]
            except:
                ind_b_not_exhausted.append(ind_b)
                continue
            while not b_exhausted and cur_S_zero_pt < len(ind_zero_slack[0]) and ind_b == ind_b_free[ind_zero_slack[0][cur_S_zero_pt]]:
                ind_a = ind_zero_slack[1][cur_S_zero_pt]
                a_exhausted = ind_a_exhausted[ind_a]
                cur_S_zero_pt += 1

                if not a_exhausted:
                    flow_to_push = min(FreeB[ind_b], PushableA[ind_a,0])
                    # push
                    FreeB[ind_b] -= flow_to_push
                    flow_pushed.append(flow_to_push)
                    # relabel
                    if FreeA[ind_a] > 0:
                        f -= flow_to_push
                        FreeA[ind_a] -= flow_to_push
                    else:
                        flow_released_ind_a[ind_a] += flow_to_push
                    # maintain variables
                    if flow_to_push == PushableA[ind_a,0]:
                        ind_a_exhausted[ind_a] = True
                    if FreeB[ind_b] == 0:
                        b_exhausted = True
                    PushableA[ind_a,1] += flow_to_push
                    PushableA[ind_a,0] -= flow_to_push
                    edges_pushed[0].append(ind_b)
                    edges_pushed[1].append(ind_a)
            if not b_exhausted:
                ind_b_not_exhausted.append(ind_b)

        FreeB_release = np.zeros(n, dtype=int)
        flow_released_ind_a_tensor = torch.tensor(flow_released_ind_a, device=device)
        ind_a_release_tensor = torch.nonzero(flow_released_ind_a_tensor, as_tuple=True)[0]
        # log released edges
        all_ind_b_relabel_tensor = torch.nonzero(torch.t(yFA[:,ind_a_release_tensor]) == yA[ind_a_release_tensor][:,None], as_tuple=True)

        all_ind_b_relabel = (all_ind_b_relabel_tensor[1].cpu().numpy(),ind_a_release_tensor[all_ind_b_relabel_tensor[0]].cpu().numpy())
        ind_a_release = ind_a_release_tensor.cpu().numpy()
        if len(all_ind_b_relabel[1]) > 0:
            ind_b_range = find_ind_range(all_ind_b_relabel[1])

        for ind_a in ind_a_release:
            if ind_a_exhausted[ind_a]:
                # List relabel candidates of B, select edges to release
                ind_b_relabel = all_ind_b_relabel[0][ind_b_range[ind_a][0]:ind_b_range[ind_a][1]+1]
                # log full released edges for slack & flow update
                if len(ind_b_relabel) > 0:
                    release_edge_b = ind_b_relabel.tolist()
                    release_edge_a = len(ind_b_relabel)*[ind_a]
                    edges_full_released[0].extend(release_edge_b)
                    edges_full_released[1].extend(release_edge_a)
            else:
                # List relabel candidates of B, select ind_b to full/partial release
                ind_b_relabel = all_ind_b_relabel[0][ind_b_range[ind_a][0]:ind_b_range[ind_a][1]+1]
                ind_b_full_released, ind_b_part_released, part_release_flow = partial_release_a(F, ind_a, ind_b_relabel, flow_released_ind_a[ind_a], device=device)
                # log full/partial released edges for slack & flow update
                if ind_b_full_released is not None:
                    release_edge_b = ind_b_full_released.tolist()
                    release_edge_a = len(ind_b_full_released)*[ind_a]
                    edges_full_released[0].extend(release_edge_b)
                    edges_full_released[1].extend(release_edge_a)
                if ind_b_part_released is not None:
                    edges_part_released[0].append(ind_b_part_released)
                    edges_part_released[1].append(ind_a)
                    flow_partial_released.append(part_release_flow)

        edges_full_released_tensor = torch.tensor(edges_full_released, dtype = torch.long, device=device)
        edges_part_released_tensor = torch.tensor(edges_part_released, dtype = torch.long, device=device)
        edges_pushed_tensor = torch.tensor(edges_pushed, dtype = torch.long, device=device)
        ind_b_not_exhausted_tensor = torch.tensor(ind_b_not_exhausted, dtype = torch.long, device=device)
        ind_a_exhausted_tensor = torch.tensor(ind_a_exhausted, dtype = torch.bool, device=device)
        flow_pushed_tensor = torch.tensor(flow_pushed, dtype = torch.long, device=device)

        PushableA[ind_a_exhausted, 0] = PushableA[ind_a_exhausted, 1]
        PushableA[ind_a_exhausted, 1] = 0

        if len(edges_part_released[0])>0:
            np.add.at(FreeB_release, edges_part_released[0], flow_partial_released)
        if len(edges_full_released[0])>0:
            np.add.at(FreeB_release, edges_full_released[0], F[edges_full_released_tensor[0], edges_full_released_tensor[1]].cpu().numpy())

        # release mass in B and flow (release goes before push)
        FreeB += FreeB_release
        if len(edges_full_released_tensor[0]) > 0:
            F[edges_full_released] = zero
            yFA[edges_full_released_tensor[0], edges_full_released_tensor[1]] = zero # set to default value when no flow
        if len(edges_part_released_tensor[0]) > 0:
            F[edges_part_released] -= torch.tensor(flow_partial_released, dtype = torch.long, device=device)

        # push flow
        F[edges_pushed[0], edges_pushed[1]] += flow_pushed_tensor
        yFA[edges_pushed_tensor[0], edges_pushed_tensor[1]] = yA[edges_pushed_tensor[1]] - one # update dual weight of a on pushed edges 
        yA[ind_a_exhausted_tensor] -= one

        # update slack
        S[:,ind_a_exhausted_tensor] += one
        S[ind_b_not_exhausted_tensor, :] -= one
        yB[ind_b_not_exhausted_tensor] += one

        # self release
        ind_f_tensor = torch.nonzero(F[ind_b_not_exhausted_tensor,:]!=0, as_tuple=True)
        ind_self_release_tensor = (ind_b_not_exhausted_tensor[ind_f_tensor[0]], ind_f_tensor[1])
        ind_f = (ind_f_tensor[0].cpu().numpy(), ind_f_tensor[1].cpu().numpy())
        ind_self_release = (np.array(ind_b_not_exhausted)[ind_f[0]], ind_f[1])
        if len(ind_self_release[0]) > 0:
            flow_release_tensor = F[ind_self_release_tensor]
            flow_release = flow_release_tensor.cpu().numpy()
            np.add.at(PushableA[:,1], ind_self_release[1], flow_release)
            np.subtract.at(PushableA[:,0], ind_self_release[1], flow_release)
            ind_a_exhausted = np.nonzero(PushableA[:,0]==0)[0]
            PushableA[ind_a_exhausted, 0] = PushableA[ind_a_exhausted, 1]
            PushableA[ind_a_exhausted, 1] = 0
            yFA[ind_self_release_tensor[0], ind_self_release_tensor[1]] = yA[ind_self_release_tensor[1]] - one
            ind_a_exhausted_tensor = torch.tensor(ind_a_exhausted, dtype = torch.long, device=device)
            yA[ind_a_exhausted_tensor] -= one
            S[:,ind_a_exhausted_tensor] += one
        iteration += 1

    Err_A_actual = np.zeros(n, dtype=float)
    F = F/torch.tensor(alpha, device=device)
    ind_a = np.nonzero(FreeA==0)[0]
    Err_A_actual[ind_a] = Err_A[ind_a]
    FreeB = FreeB/alpha + Err_B
    FreeA = FreeA/alpha
    ind_f_tensor = torch.nonzero(torch.t(F)!=0, as_tuple=True)
    ind_f = (ind_f_tensor[1].cpu().numpy(), ind_f_tensor[0].cpu().numpy())
    ind_f_range = find_ind_range(ind_f[1])
    for ind_a in range(n):
        try:
            ind_b = ind_f[0][ind_f_range[ind_a][0]]
            correct_flow = Err_A_actual[ind_a]
            F[ind_b, ind_a] -= correct_flow
            FreeB[ind_b] += correct_flow
        except:
            continue

    ind_a_left = np.nonzero(FreeA > 0)[0].tolist()
    ind_a_left_next = ind_a_left.copy()
    ind_b_left = np.nonzero(FreeB > 0)[0].tolist()
    for ind_b in ind_b_left:
        for ind_a in ind_a_left:
            flow_to_push = min(FreeB[ind_b], FreeA[ind_a])
            F[ind_b, ind_a] += flow_to_push
            FreeB[ind_b] -= flow_to_push
            FreeA[ind_a] -= flow_to_push
            if FreeA[ind_a] == 0:
                ind_a_left_next.remove(ind_a)
            if FreeB[ind_b] == 0:
                break
        ind_a_left = ind_a_left_next
        ind_a_left_next = ind_a_left.copy()

    total_cost = torch.sum(F*C)
    return F, yA, yB, total_cost, iteration, zero_slack_length

    
def find_ind_range(ind):
    n = len(ind)
    st = 0
    cur = 0
    val = ind[st]
    ret = dict()
    while(cur < n):
        if ind[cur] != val:
            ret[val] = (st, cur-1)
            st = cur
            val = ind[st]
        cur += 1
    ret[val] = (st, cur-1)
    return ret

def partial_release_a(F, ind_a, ind_b_relabel, flow_to_push, device = "cpu"):
    cumsum_flow = torch.cumsum(F[ind_b_relabel, ind_a], dim=0)
    part_released_ind_tensor = torch.nonzero(cumsum_flow > flow_to_push, as_tuple=False)
    part_released_ind = part_released_ind_tensor[0].cpu().numpy()[0]
    part_release_flow = 0
    if part_released_ind == 0:
        ind_b_full_released = None
        ind_b_part_released = ind_b_relabel[part_released_ind]
        part_release_flow = flow_to_push
    else:
        ind_b_full_released = ind_b_relabel[0:part_released_ind]
        ind_b_part_released = ind_b_relabel[part_released_ind]
        part_release_flow = flow_to_push - cumsum_flow[part_released_ind_tensor-1].cpu().numpy()[0][0]

    return ind_b_full_released, ind_b_part_released, part_release_flow

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