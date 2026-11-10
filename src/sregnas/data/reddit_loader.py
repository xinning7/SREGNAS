

from pathlib import Path
from sregnas.paths import DATA_DIR


def has_reddit_data(data_dir=None):
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    return all(
        (root / "Reddit" / "raw" / name).is_file()
        for name in ("reddit_data.npz", "reddit_graph.npz")
    )


def load_reddit_dataset(data_dir=None):
    from torch_geometric.datasets import Reddit

    root = Path(data_dir) if data_dir is not None else DATA_DIR
    if not has_reddit_data(root):
        raise FileNotFoundError(
            f"Upload reddit_data.npz and reddit_graph.npz to {root / 'Reddit' / 'raw'}"
        )
    dataset = Reddit(root=str(root / "Reddit"))
    data = dataset[0]
    return {
        "x": data.x.float(),
        "edge_index": data.edge_index.long().contiguous(),
        "labels": data.y.long().reshape(-1),
        "train_mask": data.train_mask.bool(),
        "val_mask": data.val_mask.bool(),
        "test_mask": data.test_mask.bool(),
        "num_nodes": int(data.num_nodes),
        "num_features": int(dataset.num_features),
        "num_classes": int(dataset.num_classes),
    }
