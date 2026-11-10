

import hashlib
import numpy as np


def balanced_random_membership(num_nodes, parts, seed=42):
    if not 1 <= parts <= num_nodes:
        raise ValueError("partitions must be between one and the number of nodes")
    nodes = np.random.RandomState(seed).permutation(num_nodes)
    membership = np.empty(num_nodes, dtype=np.int64)
    for part, ids in enumerate(np.array_split(nodes, parts)):
        membership[ids] = part
    return membership


def select_partition(config, rowptr, col, labels, train_mask, val_mask):
    from .proxy_partition_mode import partition_mode
    if partition_mode() == "proxy_aware" and config.partition_strategy == "fi":
        from .proxy_partition import masked_labels, select_from_config
        y, split = masked_labels(labels, train_mask, val_mask)
        candidates, detail = select_from_config(config, rowptr, col, y, split)
        return candidates[detail["selected"]], detail
    from .sampling import metis_partition

    n = len(rowptr) - 1
    parts = config.num_gpus
    from sregnas.icde27.s0 import ordered_map, span
    def construct(method):
        with span("partition_build", method=method):
            return (metis_partition(rowptr, col, parts) if method == "metis"
                    else balanced_random_membership(n, parts, config.partition_seed))
    methods = ["metis", "random"] if config.partition_strategy == "fi" else ["random"]
    candidates = dict(zip(methods, ordered_map(construct, methods)))
    scores = {}
    if config.partition_strategy == "fi":
        train_mask = np.asarray(train_mask, dtype=bool)
        val_mask = np.asarray(val_mask, dtype=bool)
        labeled = (
            train_mask | val_mask
            if config.partition_label_scope == "train_val"
            else train_mask
        )
        def score(method):
            with span("FI_statistics", method=method):
                return partition_index(
                    rowptr, col, np.asarray(labels), labeled, candidates[method], parts
                )
        scores = dict(zip(methods, ordered_map(score, methods)))
        selected = max(candidates, key=lambda method: scores[method]["FI"])
    elif config.partition_strategy == "random":
        selected = "random"
    else:
        raise ValueError("select_partition handles only random and fi")
    return candidates[selected], {
        "schema": "partition_fi_v1",
        "requested_strategy": config.partition_strategy,
        "selected": selected,
        "random_seed": config.partition_seed,
        "random_rng": "numpy.RandomState",
        "label_scope": config.partition_label_scope if scores else "unused",
        "criterion": (
            "SNR*LA; cap=5; exact ties prefer METIS"
            if scores
            else "fixed balanced random"
        ),
        "metis_backend": "torch_sparse.partition" if "metis" in candidates else None,
        "scores": scores,
        "candidate_membership_sha256": {
            method: hashlib.sha256(membership.astype(np.int64).tobytes()).hexdigest()
            for method, membership in candidates.items()
        },
        "test_labels_used": False,
    }


def partition_index(rowptr, col, labels, labeled, membership, n=8):
    pass
    nodes = np.flatnonzero(labeled)
    assert len(nodes) and len(membership) == len(labels)
    assert set(np.unique(membership)) == set(range(n))
    classes = int(np.max(labels[nodes])) + 1
    global_counts = np.bincount(labels[nodes], minlength=classes)
    prior = global_counts / global_counts.sum()
    ratios = np.empty(len(nodes), dtype=np.float64)

    for k, v in enumerate(nodes):
        neighbors = col[rowptr[v] : rowptr[v + 1]]
        degree = len(neighbors)
        ratios[k] = (
            (
                min(int(np.count_nonzero(membership[neighbors] == membership[v])), 5)
                / min(degree, 5)
            )
            if degree
            else 1.0
        )
    tv, counts, shard_snr = [], [], []
    for s in range(n):
        mask = membership[nodes] == s
        count = np.bincount(labels[nodes[mask]], minlength=classes)
        assert count.sum(), "Undefined label distribution in empty labeled shard"
        tv.append(float(np.abs(count / count.sum() - prior).sum() / 2))
        counts.append(int(count.sum()))
        shard_snr.append(float(ratios[mask].mean()))
    snr, la = float(ratios.mean()), 1 - float(np.mean(tv))
    return dict(
        SNR=snr,
        LA=la,
        FI=snr * la,
        per_shard_TV=tv,
        per_shard_SNR=shard_snr,
        per_shard_labeled_count=counts,
        labeled_count=len(nodes),
        isolated_labeled=int(np.sum(np.diff(rowptr)[nodes] == 0)),
    )
