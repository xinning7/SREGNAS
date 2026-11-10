

from typing import Dict, Set

REQUIRED_KEYS: Set[str] = {
    "x",
    "edge_index",
    "labels",
    "train_mask",
    "val_mask",
    "test_mask",
    "num_features",
    "num_classes",
    "num_nodes",
}


def validate_dataset_dict(dataset: Dict, name: str) -> None:
    pass
    missing = REQUIRED_KEYS - set(dataset.keys())
    if missing:
        raise ValueError(f"数据集 '{name}' 缺少必需键: {missing}")

    n = dataset["num_nodes"]

    x = dataset["x"]
    assert x.shape[0] == n, f"数据集 '{name}': x.shape[0]={x.shape[0]} != num_nodes={n}"
    assert (
        x.shape[1] == dataset["num_features"]
    ), f"数据集 '{name}': x.shape[1]={x.shape[1]} != num_features={dataset['num_features']}"

    assert dataset["labels"].shape[0] == n, f"数据集 '{name}': labels 长度 != num_nodes"

    ei = dataset["edge_index"]
    assert (
        ei.ndim == 2 and ei.shape[0] == 2
    ), f"数据集 '{name}': edge_index 应为 [2, E]，实际 {ei.shape}"

    if ei.numel() > 0:
        assert (
            ei.max().item() < n
        ), f"数据集 '{name}': edge_index 最大值 {ei.max().item()} >= num_nodes={n}"
        assert ei.min().item() >= 0, f"数据集 '{name}': edge_index 包含负值索引"

    for mask_name in ("train_mask", "val_mask", "test_mask"):
        assert (
            dataset[mask_name].shape[0] == n
        ), f"数据集 '{name}': {mask_name} 长度 != num_nodes"
