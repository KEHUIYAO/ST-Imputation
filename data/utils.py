from typing import Optional

import numpy as np
from numpy.random import choice
from torch_geometric.utils import k_hop_subgraph
from tsl.data import Batch
from tsl.ops.connectivity import weighted_degree



def k_hop_subgraph_sampler(batch: Batch, k: int, num_nodes: int,
                           max_edges: Optional[int] = None,
                           cut_edges_uniformly: bool = False):
    N = batch.x.size(-2)
    roots = choice(np.arange(N), num_nodes, replace=False).tolist()
    subgraph = k_hop_subgraph(roots, k, batch.edge_index, relabel_nodes=True,
                              num_nodes=N, flow='target_to_source')
    node_idx, edge_index, node_map, edge_mask = subgraph

    col = edge_index[1]
    if max_edges is not None and max_edges < edge_index.size(1):
        if not cut_edges_uniformly:
            in_degree = weighted_degree(col, num_nodes=len(node_idx))
            deg = (1 / in_degree)[col].cpu().numpy()
            p = deg / deg.sum()
        else:
            p = None
        keep_edges = sorted(choice(len(col), max_edges, replace=False, p=p))
    else:
        keep_edges = slice(None)
    for key, pattern in batch.pattern.items():
        if key in batch.target or key == 'eval_mask':
            batch[key] = batch[key][..., roots, :]
        elif 'n' in pattern:
            batch[key] = batch[key][..., node_idx, :]
        elif 'e' in pattern and key != 'edge_index':
            batch[key] = batch[key][edge_mask][keep_edges]
    batch.input.node_index = node_idx  # index of nodes in subgraph
    batch.input.target_nodes = node_map  # index of roots in subgraph
    batch.edge_index = edge_index[:, keep_edges]
    return batch


def positional_encoding(seq_len, num_nodes, d_model):
    # Create a position array for seq_len
    angle_rads = get_angles(np.arange(seq_len)[:, np.newaxis, np.newaxis],
                            np.arange(d_model)[np.newaxis, np.newaxis, :],
                            d_model)

    # Apply the sin to even indices in the array; 2i
    angle_rads[:, :, 0::2] = np.sin(angle_rads[:, :, 0::2])

    # Apply the cos to odd indices in the array; 2i+1
    angle_rads[:, :, 1::2] = np.cos(angle_rads[:, :, 1::2])

    pos_encoding = np.tile(angle_rads, (1, num_nodes, 1))

    return pos_encoding


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates
