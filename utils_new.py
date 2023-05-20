import numpy as np
import torch as t
from torch import Tensor
import torch_scatter as tsc
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.utils import to_networkx
from torch_geometric.utils import from_networkx
import torch_geometric.utils as tgu
import torch_geometric.data

import torch_sparse as ts

type_graph = torch_geometric.data.Data


def randgraph_nx(N: int, P: float, self_loops=False):
    """randgraph_nx creates a random graph in networkx

    Args:
        N (int): Number of nodes
        P (float): probability of edge formation

    Returns:
        nx graph: graph object in networkx
    """
    graph = nx.fast_gnp_random_graph(n=N, p=P)  # type: ignore

    if self_loops:
        looplist = [(i, i) for i in range(N)]
        graph.add_edges_from(looplist)

    return graph


def randg_torch(N: int, P: float, self_loops=False):
    return from_networkx(randgraph_nx(N, P, self_loops=self_loops))


def get_device(edge_index):
    device = edge_index.get_device()
    if device == -1:
        device = 'cpu'
    return device


def randspin_torch(N: int):
    return t.randint(2, size=(N,))


def visualise_graph(graph: type_graph, spin: Tensor):

    # if graph is torch data type, convert to nx for visualisation
    if type(graph) == torch_geometric.data.data.Data:
        graph = to_networkx(graph, to_undirected=True)

    plt.figure(figsize=(3, 3))
    nx.draw(graph, node_color=spin, cmap=plt.cm.winter)  # type: ignore

def visualise_sp(m_sp:Tensor):
    indices, _ = iv_from_sp(m_sp)
    graph = torch_geometric.data.Data(edge_index=indices)
    g_nx = to_networkx(graph, to_undirected=True)
    plt.figure(figsize=(3, 3))
    nx.draw(g_nx) # type: ignore


# utils
def sprint(indices, values):
    print(t.sparse_coo_tensor(indices, values).to_dense())


def iv_from_sp(m_sp: Tensor):
    m_sp = m_sp.coalesce()

    return m_sp.indices(), m_sp.values()


def sp_from_iv(i, v, N):
    return t.sparse_coo_tensor(indices=i, values=v, size=(N, N)).coalesce()


def Jt_from_Jandh(J_sp: Tensor, h_sp: Tensor):
    # J_sp should be (NxN)
    assert J_sp.shape[0] == J_sp.shape[1]

    N = J_sp.shape[0]

    assert h_sp.shape == (N, 1)

    # need to half h before combining them
    h_sp = h_sp/2

    # (N,1) + (N,N) = (N, N+1)
    Jt_sp = t.cat([h_sp, J_sp], dim=1)

    # added 0 to the first entry of h, shape becomes (N+1, 1)
    h_0 = t.cat([t.zeros((1, 1)).to_sparse_coo(), h_sp], dim=0)

    # (1, N+1) + (N,N+1) = (N+1, N+1)
    Jt_sp = t.cat([h_0.T, Jt_sp], dim=0)

    # extracting sparse entries' index and values
    Jt_ind, Jt_val = iv_from_sp(Jt_sp)

    # remove diagonal
    Jt_ind, Jt_val = tgu.remove_self_loops(Jt_ind, Jt_val)

    return sp_from_iv(i=Jt_ind, v=Jt_val, N=(N+1))


def degree(ind, val, N):
    # calculate the degree matrix of abs(Jtsl)
    row, col = ind
    values = val

    deg = tsc.scatter_add(src=values, index=row, dim_size=N)

    return deg


def sym_norm_abs(i, v, size):

    row, col = i

    abs_v = v.abs()

    deg = degree(i, abs_v, N=size)

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    normed_values = deg_inv_sqrt[row] * deg_inv_sqrt[col] * v

    return normed_values


def random_J_sp(N, P):
    edge_index = tgu.random.erdos_renyi_graph(N, P)
    num_edges = edge_index.shape[1]
    J_val = t.rand(num_edges)*2-1
    J_sp = sp_from_iv(edge_index, J_val, N=N)
    J_sp = J_sp + J_sp.T
    return J_sp


def random_h_sp(N):
    select = t.randint(high=2, size=(N,))
    indices = t.arange(N) * select
    indices = t.cat([t.zeros((1, N)), indices.unsqueeze(0)], dim=0)

    h_val = t.rand(size=(N,))*2-1

    h = t.sparse_coo_tensor(indices, values=h_val, size=(1, N))
    h = h.T
    return h.coalesce()


def norm_attention_from_raw(raw_val, Adj_ind):
    i, _ = Adj_ind
    return tgu.softmax(src=raw_val, index=i)

def graph_info(g):

    print(f"{g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

def Loss_ising(Jt_sp: Tensor, s: Tensor):

    Js = t.sparse.mm(Jt_sp, s)
    return t.sparse.mm(Js.T, s).squeeze()


def hard_project(ss):
    s_hard = t.ones_like(ss)
    s_hard[ss < 0] = -1
    return s_hard

def cut_count(Jt_sp, s):
    sum = Jt_sp.sum()
    L = Loss_ising(Jt_sp, s)

    return (sum-L)/4

def add_custom_diag(indices:Tensor, values: Tensor, custom_diag:Tensor, size:int):
    assert custom_diag.shape[0] == size
    loop_index = t.arange(0, size, dtype=t.long, device=indices.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    indices = t.cat([indices, loop_index], dim=1)
    values = t.cat([values, custom_diag], dim=0)
    return indices, values

# Generate random graph of specified size and type,
# with specified degree (d) or edge probability (p)
def generate_graph(n, d=None, p=None, graph_type='reg', random_seed=0):
    """
    Helper function to generate a NetworkX random graph of specified type,
    given specified parameters (e.g. d-regular, d=3). Must provide one of
    d or p, d with graph_type='reg', and p with graph_type in ['prob', 'erdos'].

    Input:
        n: Problem size
        d: [Optional] Degree of each node in graph
        p: [Optional] Probability of edge between two nodes
        graph_type: Specifies graph type to generate
        random_seed: Seed value for random generator
    Output:
        nx_graph: NetworkX OrderedGraph of specified type and parameters
    """
    if graph_type == 'reg':
        print(f'Generating d-regular graph with n={n}, d={d}, seed={random_seed}')
        nx_temp = nx.random_regular_graph(d=d, n=n, seed=random_seed)
    elif graph_type == 'prob':
        print(f'Generating p-probabilistic graph with n={n}, p={p}, seed={random_seed}')
        nx_temp = nx.fast_gnp_random_graph(n, p, seed=random_seed)
    elif graph_type == 'erdos':
        print(f'Generating erdos-renyi graph with n={n}, p={p}, seed={random_seed}')
        nx_temp = nx.erdos_renyi_graph(n, p, seed=random_seed)
    else:
        raise NotImplementedError(f'!! Graph type {graph_type} not handled !!')

    graph_info(nx_temp)
    # Need to pull nx graph into OrderedGraph so training will work properly
    return from_networkx(nx_temp)

# the theoretical upperbound of the number of cuts on a d-regular n-node graph
def theo_mc_reg(n,d):
    return int((d/4 + 0.7632*((d/4)**0.5))*n)

def Jt_from_Q(Q_sp):

    J_sp = Q_sp

    h_sp = 2*Q_sp.sum(1).unsqueeze(1)

    Jt = Jt_from_Jandh(J_sp, h_sp)
    return Jt

def Jt_MIS(A_sp):
    N = A_sp.shape[0]
    Q_sp = A_sp - t.eye(n=N).to_sparse_coo()
    return Jt_from_Q(Q_sp)

def Jt_MC(A_sp):
    N = A_sp.shape[0]
    h_sp = t.zeros(size=(N, 1)).to_sparse_coo()
    Jt_MC = Jt_from_Jandh(A_sp, h_sp)
    return Jt_MC

from torch_geometric.utils import to_torch_coo_tensor
def A_sp_from_txt(path):
    g_nx = nx.read_weighted_edgelist(path)  # type: ignore
    g = from_networkx(G=g_nx)

    N = g.num_nodes
    A_sp = to_torch_coo_tensor(edge_index=g.edge_index, size=N, edge_attr=g.weight)
    return A_sp

def evaluate_ss(Jt_sp, Jt_sp_sum, ss, theo):
    loss = Loss_ising(Jt_sp, ss)
    loss_ = loss.detach().item()

    hs_ = hard_project(ss).detach()
    hard_loss_ = Loss_ising(Jt_sp, hs_).item()
    hard_cut_ = (Jt_sp_sum - hard_loss_)/4  # needs to be written
    ratio_ = hard_cut_/theo

    return loss, loss_, hs_, hard_loss_, hard_cut_, ratio_

def print_status(epoch, loss_, ratio):
    output = ''
    output += f'Epoch: {epoch:.4g}, Loss: {loss_}, Ratio: {ratio:.2%} '
    print(output)
    print('')