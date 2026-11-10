

import torch
from typing import Dict


def normalize_features(dataset: Dict) -> Dict:
    pass
    x = dataset["x"]
    if x is None:
        return dataset
    row_sum = x.sum(dim=1, keepdim=True)
    row_sum = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
    dataset["x"] = x / row_sum
    return dataset


def add_self_loops(dataset: Dict) -> Dict:
    pass
    from torch_geometric.utils import add_remaining_self_loops

    edge_index = dataset["edge_index"]
    num_nodes = dataset["num_nodes"]
    new_edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
    dataset["edge_index"] = new_edge_index
    return dataset


def symmetric_normalize_adj(dataset: Dict) -> Dict:
    pass
    from torch_geometric.utils import degree

    edge_index = dataset["edge_index"]
    num_nodes = dataset["num_nodes"]
    row, col = edge_index
    deg = degree(edge_index[0], num_nodes, dtype=torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    dataset["edge_weight"] = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return dataset


def preprocess_dataset(
    dataset: Dict,
    enable_self_loops: bool = False,
    normalize: bool = True,
    symmetric_norm: bool = False,
) -> Dict:
    pass
    processed = dataset.copy()

    if enable_self_loops:
        processed["edge_index"] = processed["edge_index"].clone()
        processed = add_self_loops(processed)

    if normalize:
        processed["x"] = processed["x"].clone()
        processed = normalize_features(processed)

    if symmetric_norm:
        processed["edge_index"] = processed["edge_index"].clone()
        processed = symmetric_normalize_adj(processed)

    return processed
