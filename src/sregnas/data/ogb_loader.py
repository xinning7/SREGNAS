

import logging
from pathlib import Path
from sregnas.paths import DATA_DIR
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_PROJECT_DATA_DIR = DATA_DIR


def _resolve_data_dir(data_dir: Optional[str] = None) -> str:
    if data_dir is not None:
        return data_dir
    _PROJECT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return str(_PROJECT_DATA_DIR)


def has_ogb_node_data(name: str, data_dir: Optional[str] = None) -> bool:
    pass
    root = Path(_resolve_data_dir(data_dir))
    dataset_dir = root / name.replace("-", "_")
    edge_file = dataset_dir / "raw" / "edge.csv.gz"

    has_release = (
        any(dataset_dir.glob("RELEASE_v*.txt")) if dataset_dir.exists() else False
    )
    return edge_file.exists() and has_release


def has_ogbn_arxiv_data(data_dir: Optional[str] = None) -> bool:
    return has_ogb_node_data("ogbn-arxiv", data_dir)


def has_ogbn_products_data(data_dir: Optional[str] = None) -> bool:
    return has_ogb_node_data("ogbn-products", data_dir)


def load_ogb_node_dataset(name: str, data_dir: Optional[str] = None) -> Dict:
    pass
    try:
        from ogb.nodeproppred import NodePropPredDataset
    except ImportError:
        raise ImportError(
            "加载 ogbn-arxiv 数据集需要 ogb 包（ogb>=1.3.6）。\n"
            "当前环境应已安装，请检查依赖。"
        )

    import torch

    resolved_dir = _resolve_data_dir(data_dir)

    try:
        dataset = NodePropPredDataset(name=name, root=resolved_dir)
    except Exception as e:
        raise RuntimeError(
            f"加载 {name} 失败（路径: {resolved_dir}/{name.replace('-', '_')}/）。\n"
            f"请将数据集文件上传到正确位置后重试。\n"
            f"原始错误: {e}"
        ) from e

    graph, labels = dataset[0]




    x = torch.tensor(graph["node_feat"], dtype=torch.float32)
    node_labels = torch.tensor(labels.flatten(), dtype=torch.long)
    num_nodes: int = int(graph["num_nodes"])
    num_features: int = x.shape[1]
    num_classes: int = int(node_labels.max().item()) + 1


    edge_index_np = graph["edge_index"]
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    edge_index_rev = edge_index.flip(0)
    edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    logger.info(
        "[%s] 原始边 %d 条 → 无向边 %d 条（转换后）",
        name,
        edge_index_np.shape[1],
        edge_index.shape[1],
    )


    split_idx = dataset.get_idx_split()
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[torch.tensor(split_idx["train"], dtype=torch.long)] = True
    val_mask[torch.tensor(split_idx["valid"], dtype=torch.long)] = True
    test_mask[torch.tensor(split_idx["test"], dtype=torch.long)] = True

    logger.info(
        "[%s] 加载完成: %d 节点, %d 维特征, %d 类, %d 边 (无向), "
        "train=%d val=%d test=%d",
        name,
        num_nodes,
        num_features,
        num_classes,
        edge_index.shape[1],
        train_mask.sum().item(),
        val_mask.sum().item(),
        test_mask.sum().item(),
    )

    return {
        "x": x,
        "edge_index": edge_index,
        "labels": node_labels,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "num_features": num_features,
        "num_classes": num_classes,
        "num_nodes": num_nodes,
    }


def load_ogbn_arxiv_dataset(data_dir: Optional[str] = None) -> Dict:
    return load_ogb_node_dataset("ogbn-arxiv", data_dir)


def load_ogbn_products_dataset(data_dir: Optional[str] = None) -> Dict:
    return load_ogb_node_dataset("ogbn-products", data_dir)
