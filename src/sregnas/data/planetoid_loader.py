

import logging
from pathlib import Path
from sregnas.paths import DATA_DIR
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _resolve_data_dir(dataset_name: str, data_dir: Optional[str] = None) -> str:
    if data_dir is not None:
        return data_dir
    project_data_dir = DATA_DIR
    project_data_dir.mkdir(parents=True, exist_ok=True)
    return str(project_data_dir)


def load_planetoid_dataset(name: str, data_dir: Optional[str] = None) -> Dict:
    pass
    try:
        from torch_geometric.datasets import Planetoid
    except ImportError:
        raise ImportError(
            f"加载 {name} 数据集需要 torch_geometric。\n"
            "请运行: pip install torch_geometric"
        )

    resolved_dir = _resolve_data_dir(name, data_dir)
    try:
        dataset = Planetoid(root=resolved_dir, name=name)
    except Exception as e:
        raise RuntimeError(f"加载数据集 {name} 失败 (路径: {resolved_dir}): {e}") from e

    data = dataset[0]


    from torch_geometric.utils import to_undirected

    edge_index = to_undirected(data.edge_index)

    result = {
        "x": data.x,
        "edge_index": edge_index,
        "labels": data.y,
        "train_mask": data.train_mask,
        "val_mask": data.val_mask,
        "test_mask": data.test_mask,
        "num_features": dataset.num_features,
        "num_classes": dataset.num_classes,
        "num_nodes": data.num_nodes,
    }

    logger.info(
        "[%s] 加载完成: %d 节点, %d 维特征, %d 类, %d 边",
        name,
        result["num_nodes"],
        result["num_features"],
        result["num_classes"],
        result["edge_index"].size(1),
    )
    return result


def load_cora_dataset(data_dir: Optional[str] = None) -> Dict:
    return load_planetoid_dataset("Cora", data_dir)


def load_pubmed_dataset(data_dir: Optional[str] = None) -> Dict:
    return load_planetoid_dataset("PubMed", data_dir)


def load_citeseer_dataset(data_dir: Optional[str] = None) -> Dict:
    return load_planetoid_dataset("CiteSeer", data_dir)
