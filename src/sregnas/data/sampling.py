

import numpy as np

FULL = None


def build_csr(edge_index_np, num_nodes):
    pass
    src = edge_index_np[0].astype(np.int64, copy=False)
    dst = edge_index_np[1].astype(np.int64, copy=False)

    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])

    keep = u != v
    u, v = u[keep], v[keep]

    key = u * np.int64(num_nodes) + v
    key = np.unique(key)
    u = key // np.int64(num_nodes)
    v = key % np.int64(num_nodes)

    deg = np.bincount(u, minlength=num_nodes).astype(np.int64)
    rowptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(deg, out=rowptr[1:])
    col = v.astype(np.int32)
    return rowptr, col, deg


def shuffle_csr_segments(rowptr, col, seed):
    pass
    rng = np.random.default_rng(seed)
    E = len(col)

    row = np.repeat(
        np.arange(len(rowptr) - 1, dtype=np.int64), np.diff(rowptr).astype(np.int64)
    )

    keys = rng.random(E, dtype=np.float32)

    idx = np.lexsort((keys, row))
    return col[idx]


def build_layer_perms(rowptr, col, base_seed, num_layers=2):
    pass
    import os
    if os.environ.get('SREGNAS_X18_PARALLEL_PERMUTATIONS') == '1':
        from sregnas.icde27.x18.prep import layer_permutations
        return layer_permutations(rowptr, col, base_seed, num_layers)
    return [
        shuffle_csr_segments(rowptr, col, base_seed + layer)
        for layer in range(num_layers)
    ]


def _ragged_positions(offsets, caps):
    pass
    caps = caps.astype(np.int64, copy=False)
    total = int(caps.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    cs = np.zeros(len(caps) + 1, dtype=np.int64)
    np.cumsum(caps, out=cs[1:])
    node_of = np.repeat(np.arange(len(caps), dtype=np.int64), caps)
    within = np.arange(total, dtype=np.int64) - cs[node_of]
    return offsets.astype(np.int64)[node_of] + within


def compute_2hop_cache(rowptr, col_L1, col_L2, core_mask, f1, f2):
    pass
    N = len(rowptr) - 1
    deg = (rowptr[1:] - rowptr[:-1]).astype(np.int64)
    core_ids = np.flatnonzero(core_mask).astype(np.int64)


    if f2 is FULL:
        cap2 = deg[core_ids]
    else:
        cap2 = np.minimum(deg[core_ids], np.int64(f2))
    pos2 = _ragged_positions(rowptr[core_ids], cap2)
    frontier1 = np.unique(col_L2[pos2]) if len(pos2) else np.empty(0, dtype=np.int32)
    msg2 = int(len(pos2))


    block1_mask = core_mask.copy()
    block1_mask[frontier1.astype(np.int64)] = True
    block1_ids = np.flatnonzero(block1_mask).astype(np.int64)
    if f1 is FULL:
        cap1 = deg[block1_ids]
    else:
        cap1 = np.minimum(deg[block1_ids], np.int64(f1))
    pos1 = _ragged_positions(rowptr[block1_ids], cap1)
    frontier2 = np.unique(col_L1[pos1]) if len(pos1) else np.empty(0, dtype=np.int32)
    msg1 = int(len(pos1))


    touched = np.unique(
        np.concatenate(
            [
                core_ids.astype(np.int32),
                frontier1.astype(np.int32),
                frontier2.astype(np.int32),
            ]
        )
    )
    E_msg_sub = msg2 + msg1
    return touched, E_msg_sub


def _segmented_exclusive_count(node_of, mask):
    pass
    n = len(node_of)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    m = mask.astype(np.int64)
    excl = np.cumsum(m) - m
    seg_start = np.empty(n, dtype=bool)
    seg_start[0] = True
    seg_start[1:] = node_of[1:] != node_of[:-1]
    seg_id = np.cumsum(seg_start) - 1
    base = excl[np.flatnonzero(seg_start)][seg_id]
    return excl - base


def _locality_select(rowptr, col_perm, source_ids, membership, part, f_int, f_ext):
    pass
    deg = (rowptr[1:] - rowptr[:-1]).astype(np.int64)
    caps = deg[source_ids]
    pos = _ragged_positions(rowptr[source_ids], caps)
    if len(pos) == 0:
        return np.empty(0, dtype=np.int32), 0
    node_of = np.repeat(np.arange(len(source_ids), dtype=np.int64), caps)
    nbr = col_perm[pos].astype(np.int64)
    is_inside = membership[nbr] == part
    inside_rank = _segmented_exclusive_count(node_of, is_inside)
    outside_rank = _segmented_exclusive_count(node_of, ~is_inside)
    keep = (is_inside & (inside_rank < np.int64(f_int))) | (
        (~is_inside) & (outside_rank < np.int64(f_ext))
    )
    kept_nbr = np.unique(nbr[keep]).astype(np.int32)
    return kept_nbr, int(keep.sum())


def compute_2hop_cache_locality(rowptr, col_L1, col_L2, membership, part, f_int, f_ext):
    pass
    core_mask = membership == part
    core_ids = np.flatnonzero(core_mask).astype(np.int64)


    frontier1, msg2 = _locality_select(
        rowptr, col_L2, core_ids, membership, part, f_int, f_ext
    )


    block1_mask = core_mask.copy()
    block1_mask[frontier1.astype(np.int64)] = True
    block1_ids = np.flatnonzero(block1_mask).astype(np.int64)
    frontier2, msg1 = _locality_select(
        rowptr, col_L1, block1_ids, membership, part, f_int, f_ext
    )

    touched = np.unique(
        np.concatenate(
            [
                core_ids.astype(np.int32),
                frontier1.astype(np.int32),
                frontier2.astype(np.int32),
            ]
        )
    )
    E_msg_sub = msg2 + msg1
    return touched, E_msg_sub


def metis_partition(rowptr, col, num_parts):
    pass
    import torch
    import torch_sparse

    rowptr_t = torch.from_numpy(rowptr.astype(np.int64))
    col_t = torch.from_numpy(col.astype(np.int64))
    membership = torch.ops.torch_sparse.partition(
        rowptr_t, col_t, None, int(num_parts), False
    )
    return membership.numpy().astype(np.int64)


def partition_stats(rowptr, col, membership, num_parts):
    pass
    N = len(rowptr) - 1
    row = np.repeat(np.arange(N, dtype=np.int64), np.diff(rowptr).astype(np.int64))
    cut = membership[row] != membership[col.astype(np.int64)]
    E_msg = len(col)
    n_cut = int(cut.sum())

    boundary = np.zeros(N, dtype=bool)
    boundary[row[cut]] = True
    part_sizes = np.bincount(membership, minlength=num_parts).astype(int).tolist()
    return {
        "edge_cut_pct": 100.0 * n_cut / E_msg if E_msg else 0.0,
        "n_cut_edges": n_cut,
        "E_msg": E_msg,
        "part_sizes": part_sizes,
        "boundary_frac": float(boundary.sum()) / N,
        "n_boundary": int(boundary.sum()),
    }






H = 256


def flops_2layer(M_total, V, feat_dim, h=H):
    pass
    agg = 2 * M_total * h
    xf = 2 * V * feat_dim * h + 2 * V * h * h
    return agg + xf


def full_graph_sampled_messages(deg, f1, f2):
    pass
    m2 = np.minimum(deg, np.int64(f2)).sum() if f2 is not FULL else deg.sum()
    m1 = np.minimum(deg, np.int64(f1)).sum() if f1 is not FULL else deg.sum()
    return int(m1 + m2)


def full_graph_sampled_messages_locality(deg, f_int, f_ext):
    pass
    cap = np.int64(f_int + f_ext)
    m = np.minimum(deg, cap).sum()
    return int(2 * m)


def cost_ratios(
    M_full_all,
    V_all,
    M_full_sampled_all,
    M_part_sampled_max,
    V_part_touched_max,
    M_part_full_max,
    V_part_touched_full_max,
    feat_dim,
):
    pass
    f_full = flops_2layer(M_full_all, V_all, feat_dim)
    f_full_samp = flops_2layer(M_full_sampled_all, V_all, feat_dim)
    f_part_samp = flops_2layer(M_part_sampled_max, V_part_touched_max, feat_dim)
    f_part_full = flops_2layer(M_part_full_max, V_part_touched_full_max, feat_dim)
    return {
        "R1": f_full / f_part_samp,
        "R2": f_full_samp / f_part_samp,
        "R3": f_part_full / f_part_samp,
        "flops_full": f_full,
        "flops_full_sampled": f_full_samp,
        "flops_part_sampled": f_part_samp,
        "flops_part_full": f_part_full,
    }





BYTES_F32 = 4
GB = 1024**3
USABLE_GB = 44.0
GAT_HEADS = 8


def mem_spmm_gb(V, h=H, L=2):
    return V * h * (L + 1) * BYTES_F32 * 3 / GB


def mem_edgemsg_gb(V, E_msg, h=H, L=2, heads=GAT_HEADS):
    base = mem_spmm_gb(V, h, L)
    extra = (E_msg * heads * BYTES_F32 + E_msg * h * BYTES_F32) / GB
    return base + extra



def exact_2hop_closure(rowptr, col, core_mask):
    pass
    N = len(rowptr) - 1
    deg = np.diff(rowptr).astype(np.int64)

    def neighbors_of(mask):
        ids = np.flatnonzero(mask).astype(np.int64)
        pos = _ragged_positions(rowptr[ids], deg[ids])
        return np.unique(col[pos]) if len(pos) else np.empty(0, dtype=np.int32)


    f1 = neighbors_of(core_mask)
    m1 = core_mask.copy()
    m1[f1.astype(np.int64)] = True

    f2 = neighbors_of(m1)
    core_ids = np.flatnonzero(core_mask).astype(np.int32)
    return np.unique(
        np.concatenate([core_ids, f1.astype(np.int32), f2.astype(np.int32)])
    )
