

import numpy as np
from . import sampling as L


def induced_metis(rowptr, col, node_ids, k):
    pass
    nsub = len(node_ids)
    if k == 1:
        return np.zeros(nsub, dtype=np.int64)
    if nsub <= k:
        return np.arange(nsub, dtype=np.int64) % k
    N = len(rowptr) - 1
    old2new = -np.ones(N, dtype=np.int64)
    old2new[node_ids] = np.arange(nsub, dtype=np.int64)
    row = np.repeat(np.arange(N, dtype=np.int64), np.diff(rowptr).astype(np.int64))
    c = col.astype(np.int64)
    keep = (old2new[row] >= 0) & (old2new[c] >= 0)
    su, sv = old2new[row[keep]], old2new[c[keep]]
    sd = np.bincount(su, minlength=nsub).astype(np.int64)
    srp = np.zeros(nsub + 1, dtype=np.int64)
    np.cumsum(sd, out=srp[1:])
    order = np.lexsort((sv, su))
    sc = sv[order].astype(np.int32)
    return L.metis_partition(srp, sc, k)


def two_level_membership(rowptr, col, R, N_fine):
    pass
    Nn = len(rowptr) - 1
    if R == 1:
        membership = L.metis_partition(rowptr, col, N_fine)
        macro = np.zeros(Nn, dtype=np.int64)
        return membership, macro
    macro = L.metis_partition(rowptr, col, R)
    membership = -np.ones(Nn, dtype=np.int64)
    for r in range(R):
        node_ids = np.flatnonzero(macro == r).astype(np.int64)
        fine = induced_metis(rowptr, col, node_ids, N_fine)
        membership[node_ids] = fine
    assert (membership >= 0).all(), "unassigned node in two-level membership"
    return membership, macro
