

import torch
from typing import Tuple


def create_simple_graph_dataset(
    num_nodes: int = 10,
    num_classes: int = 3,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    pass
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    x = torch.eye(num_nodes, dtype=torch.float32)

    edges = []
    for i in range(num_nodes - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    labels = torch.tensor([i % num_classes for i in range(num_nodes)], dtype=torch.long)

    num_train = int(num_nodes * train_ratio)
    num_val = int(num_nodes * val_ratio)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:num_train] = True
    val_mask[num_train : num_train + num_val] = True
    test_mask[num_train + num_val :] = True

    return x, edge_index, labels, train_mask, val_mask, test_mask


def get_simple_dataset(**kwargs):
    pass
    x, edge_index, labels, train_mask, val_mask, test_mask = (
        create_simple_graph_dataset(num_nodes=10, num_classes=3)
    )
    return {
        "x": x,
        "edge_index": edge_index,
        "labels": labels,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "num_features": x.size(1),
        "num_classes": 3,
        "num_nodes": 10,
    }
