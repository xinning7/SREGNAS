

from __future__ import annotations

import logging
from pathlib import Path
from sregnas.paths import DATA_DIR
from typing import Dict, Optional

logger = logging.getLogger(__name__)
_PROJECT_DATA_DIR = DATA_DIR


def has_flickr_data(data_dir: Optional[str] = None) -> bool:
    root = Path(data_dir) if data_dir is not None else _PROJECT_DATA_DIR
    raw = root / "Flickr" / "raw"
    return all(
        (raw / filename).exists()
        for filename in ("adj_full.npz", "feats.npy", "class_map.json", "role.json")
    )


def load_flickr_dataset(data_dir: Optional[str] = None) -> Dict:
    pass
    try:
        from torch_geometric.datasets import Flickr
        from torch_geometric.utils import to_undirected
    except ImportError as exc:
        raise ImportError("加载 Flickr 需要 torch_geometric。") from exc

    root = Path(data_dir) if data_dir is not None else _PROJECT_DATA_DIR
    root.mkdir(parents=True, exist_ok=True)
    dataset_root = root / "Flickr"
    try:
        dataset = Flickr(root=str(dataset_root))
    except Exception as exc:
        raise RuntimeError(f"加载 Flickr 失败（路径: {dataset_root}）: {exc}") from exc

    data = dataset[0]
    edge_index = to_undirected(data.edge_index)
    result = {
        "x": data.x.float(),
        "edge_index": edge_index.long().contiguous(),
        "labels": data.y.long().reshape(-1),
        "train_mask": data.train_mask.bool(),
        "val_mask": data.val_mask.bool(),
        "test_mask": data.test_mask.bool(),
        "num_features": int(dataset.num_features),
        "num_classes": int(dataset.num_classes),
        "num_nodes": int(data.num_nodes),
    }
    logger.info(
        "[Flickr] 加载完成: %d 节点, %d 维特征, %d 类, %d 边",
        result["num_nodes"],
        result["num_features"],
        result["num_classes"],
        result["edge_index"].size(1),
    )
    return result
