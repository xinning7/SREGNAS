

import logging
from typing import Dict, Optional, Callable, Any
from pathlib import Path

from sregnas.data.simple_dataset import get_simple_dataset
from sregnas.data.planetoid_loader import (
    load_cora_dataset,
    load_pubmed_dataset,
    load_citeseer_dataset,
)
from sregnas.data.flickr_loader import load_flickr_dataset
from sregnas.data.reddit_loader import load_reddit_dataset
from sregnas.data.ogb_loader import (
    load_ogbn_arxiv_dataset,
    load_ogbn_products_dataset,
)
from sregnas.data.validate import validate_dataset_dict

logger = logging.getLogger(__name__)

SUPPORTED_DATASETS: Dict[str, Callable] = {
    "simple": get_simple_dataset,
    "cora": load_cora_dataset,
    "pubmed": load_pubmed_dataset,
    "citeseer": load_citeseer_dataset,
    "flickr": load_flickr_dataset,
    "reddit": load_reddit_dataset,
    "ogbn-arxiv": load_ogbn_arxiv_dataset,
    "ogbn-products": load_ogbn_products_dataset,
}


DATASET_ARTIFACT_NAMES: Dict[str, str] = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
    "flickr": "Flickr",
    "reddit": "Reddit",
    "ogbn-arxiv": "ogbn-arxiv",
    "ogbn-products": "ogbn-products",
}

DATASET_INFO: Dict[str, Dict[str, Any]] = {
    "reddit": {
        "num_nodes": 232965,
        "num_features": 602,
        "num_classes": 41,
        "description": "Reddit 帖子社区分类网络（41 类）",
        "format": "Reddit (PyTorch Geometric)",
    },
    "simple": {
        "num_nodes": 10,
        "num_features": 10,
        "num_classes": 3,
        "description": "简单测试数据集（链式图）",
        "format": "synthetic",
    },
    "cora": {
        "num_nodes": 2708,
        "num_features": 1433,
        "num_classes": 7,
        "description": "Cora 论文引用网络（7 类）",
        "format": "Planetoid (PyTorch Geometric)",
    },
    "pubmed": {
        "num_nodes": 19717,
        "num_features": 500,
        "num_classes": 3,
        "description": "PubMed 糖尿病文献网络（3 类）",
        "format": "Planetoid (PyTorch Geometric)",
    },
    "citeseer": {
        "num_nodes": 3327,
        "num_features": 3703,
        "num_classes": 6,
        "description": "CiteSeer 论文引用网络（6 类）",
        "format": "Planetoid (PyTorch Geometric)",
    },
    "flickr": {
        "num_nodes": 89250,
        "num_features": 500,
        "num_classes": 7,
        "description": "Flickr 图片关系网络（7 类）",
        "format": "Flickr (PyTorch Geometric)",
    },
    "ogbn-arxiv": {
        "num_nodes": 169343,
        "num_features": 128,
        "num_classes": 40,
        "description": "ogbn-arxiv CS 论文引用网络（OGB，40 类，转无向）",
        "format": "OGB NodePropPredDataset",
    },
    "ogbn-products": {
        "num_nodes": 2449029,
        "num_features": 100,
        "num_classes": 47,
        "description": "ogbn-products 商品共购网络（OGB，47 类）",
        "format": "OGB NodePropPredDataset",
    },
}



_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def has_pubmed_data() -> bool:
    pass
    raw_dir = _PROJECT_ROOT / "data" / "PubMed" / "raw"
    return (raw_dir / "ind.pubmed.graph").exists()


def has_ogbn_arxiv_data() -> bool:
    pass
    from sregnas.data.ogb_loader import has_ogbn_arxiv_data as _check

    return _check()


def get_dataset_artifact_name(dataset_name: str) -> str:
    pass
    normalized = dataset_name.lower()
    try:
        return DATASET_ARTIFACT_NAMES[normalized]
    except KeyError as exc:
        raise ValueError(f"数据集没有 v2 artifact 映射: {dataset_name}") from exc


def load_dataset(
    dataset_name: str = "cora",
    normalize: bool = False,
    preprocess: bool = False,
    **kwargs,
) -> Dict:
    pass
    dataset_name = dataset_name.lower()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"不支持的数据集: {dataset_name}\n"
            f"支持的数据集: {list(SUPPORTED_DATASETS.keys())}"
        )

    dataset = SUPPORTED_DATASETS[dataset_name](**kwargs)
    validate_dataset_dict(dataset, dataset_name)

    if normalize or preprocess:
        from sregnas.data.preprocess import preprocess_dataset

        dataset = preprocess_dataset(
            dataset,
            enable_self_loops=preprocess,
            normalize=normalize,
        )

    return dataset


def get_dataset_info(dataset_name: str = "cora") -> Dict[str, Any]:
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_INFO:
        raise ValueError(f"未知数据集: {dataset_name}")
    return DATASET_INFO[dataset_name].copy()


def list_datasets() -> list:
    return list(SUPPORTED_DATASETS.keys())


def get_dataset_metadata(dataset_name: str = "cora") -> Dict:
    info = get_dataset_info(dataset_name)
    info["available_splits"] = ["train", "val", "test"]
    return info


def register_dataset(
    name: str,
    loader_func: Callable,
    info: Optional[Dict] = None,
) -> None:
    pass
    SUPPORTED_DATASETS[name.lower()] = loader_func
    if info:
        DATASET_INFO[name.lower()] = info
    logger.info("注册数据集: %s", name)
