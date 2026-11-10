
import hashlib
import time
from math import comb

import numpy as np
from numba import njit
from .proxy_partition_mode import partition_mode


def masked_labels(labels, train_mask, val_mask):
    pass
    train = np.asarray(train_mask, dtype=bool)
    val = np.asarray(val_mask, dtype=bool)
    split = np.full(len(labels), 2, dtype=np.int64)
    split[val] = 1
    split[train] = 0
    y = np.full(len(labels), -1, dtype=np.int64)
    labeled = split < 2
    y[labeled] = np.asarray(labels)[labeled]
    return y, split


def strata(y, split):
    assert np.all(y[split >= 2] == -1), "test labels must be masked to -1 before entry"
    assert np.all(y[split < 2] >= 0), "train/val labels must be available"
    return np.where(split < 2, y * 2 + np.minimum(split, 1), -1)


def d_star(N, f=5, eps=0.1):

    p, d = 1.0 / N, 1
    while f - sum((f-x)*comb(d, x)*p**x*(1-p)**(d-x)
                  for x in range(min(f, d+1))) < (1-eps)*f:
        d += 1
    return d


@njit(cache=True)
def _weights(rowptr, col):
    deg = np.diff(rowptr)
    w = np.empty(len(col), dtype=np.float64)
    wi = np.empty(len(col), dtype=np.int64)
    for v in range(len(deg)):
        for j in range(rowptr[v], rowptr[v+1]):
            value = 1.0 / deg[v] + 1.0 / deg[col[j]]
            w[j] = value
            wi[j] = max(1, int(np.rint(100 * value)))
    return w, wi


def neighborhood_loss_weights(A):
    return _weights(A.indptr, A.indices)


def metis_parts(A, nparts, eweights=None, seed=0):
    import torch
    import torch_sparse
    result = torch.ops.torch_sparse.partition(
        torch.from_numpy(A.indptr.astype(np.int64)),
        torch.from_numpy(A.indices.astype(np.int64)),
        torch.from_numpy(eweights) if eweights is not None else None,
        int(nparts), False,
    )
    return result.numpy().astype(np.int64)


@njit(cache=True)
def _edge_statistics(rowptr, col, y, split, part, f):
    n = len(part)
    retained = np.zeros(n, dtype=np.int64)
    cut = 0
    ll = 0
    ll_same = 0
    kept_ll = 0
    kept_same = 0
    omega = 0.0
    ratios = np.ones(n, dtype=np.float64)
    for v in range(n):
        degree = rowptr[v+1] - rowptr[v]
        for j in range(rowptr[v], rowptr[v+1]):
            u = col[j]
            same = part[v] == part[u]
            if same:
                retained[v] += 1
            else:
                cut += 1
            if split[v] < 2 and split[u] < 2:
                ll += 1
                ll_same += int(y[v] == y[u])
                if same:
                    kept_ll += 1
                    kept_same += int(y[v] == y[u])
        if degree:
            omega += (degree-retained[v]) / degree
            ratios[v] = min(retained[v], f) / min(degree, f)
    return ratios, omega, cut, ll, ll_same, kept_ll, kept_same


def fi_metrics(A, y, split, part, N, f=5):
    st = strata(y, split)
    lab = split < 2
    ratios, omega, cut, ll, same_ll, keep_ll, keep_same = _edge_statistics(
        A.indptr, A.indices, y, split, part, f)
    snr = float(ratios[lab].mean())
    C = int(y[lab].max()) + 1
    prior = np.bincount(y[lab], minlength=C) / lab.sum()
    tv, counts, val_counts = [], [], []
    for s in range(N):
        mask = lab & (part == s)
        hist = np.bincount(y[mask], minlength=C)
        tv.append(float(np.abs(hist / max(mask.sum(), 1) - prior).sum() / 2))
        counts.append(int(mask.sum()))
        val_counts.append(int(np.sum((split == 1) & (part == s))))
    sizes = np.bincount(part, minlength=N)
    la = 1.0 - float(np.mean(tv))
    K = int(st.max()) + 1
    hist = np.bincount(part[lab]*K + st[lab], minlength=N*K).reshape(N, K)
    target = hist.sum(0) / N
    la_st = 1-float(np.abs(hist-target).sum()/(2*lab.sum()))
    quota = np.ceil(1.05*hist.sum(0)/N)
    hbias = keep_same / keep_ll - same_ll / ll if keep_ll and ll else float("nan")
    return dict(SNR=snr, LA=la, FI=snr*la, LA_st=la_st,
        cut=cut/len(A.indices) if len(A.indices) else float("nan"), Omega=omega,
        omega_per_vertex=omega/len(part), omega_ok=bool(lab.sum()*(1-snr) <= omega+1e-6),
        hbias=hbias, eta_V=float(sizes.max()/(len(part)/N)),
        val_min=min(val_counts), val_max=max(val_counts), per_shard_val_count=val_counts,
        per_shard_labeled_count=counts, per_shard_size=sizes.tolist(), per_shard_TV=tv,
        cap=1.05*len(part)/N, capacity_ok=bool(sizes.max() <= 1.05*len(part)/N),
        max_quota_excess=int(max(0, (hist-quota).max())), quota_ok=bool((hist <= quota).all()))


@njit(cache=True)
def _bfs(rowptr, col, anchor):
    n = len(anchor)
    level = np.full(n, -1, dtype=np.int64)
    queue = np.empty(n, dtype=np.int64)
    tail = 0
    for v in range(n):
        if anchor[v]:
            queue[tail] = v
            tail += 1
            level[v] = 0
    head = 0
    while head < tail:
        v = queue[head]
        head += 1
        for j in range(rowptr[v], rowptr[v+1]):
            u = col[j]
            if level[u] < 0:
                level[u] = level[v]+1
                queue[tail] = u
                tail += 1
    return level


@njit(cache=True)
def _followers(rowptr, col, st, followers, part, sizes, qcnt, qcap, cap, rng):
    N = len(sizes)
    quota_fallbacks = 0
    for v in followers:
        ok = sizes < cap
        k = st[v]
        if k >= 0:
            okq = ok & (qcnt[:, k] < qcap[k])
            if okq.any():
                ok = okq
            else:
                quota_fallbacks += 1
        cnt = np.zeros(N, dtype=np.float64)
        for j in range(rowptr[v], rowptr[v+1]):
            s = part[col[j]]
            if s >= 0:
                cnt[s] += 1
        sc = np.where(ok, cnt * (1.0-sizes/cap), -1.0)
        best = sc.max()
        if best > 0:
            ties = np.flatnonzero(sc == best)
            s = ties[rng.integers(0, len(ties))]
        else:
            rem = np.where(ok, cap-sizes, 0.0)


            cdf = np.cumsum(rem/rem.sum())
            cdf /= cdf[-1]
            s = np.searchsorted(cdf, rng.random(), side="right")
        part[v] = s
        sizes[s] += 1
        if k >= 0:
            qcnt[s, k] += 1
    return quota_fallbacks


def da_random(A, y, split, N, f=5, eps=0.1, delta=0.05,
              quota_slack=0.05, seed=0, return_info=False):
    st = strata(y, split)
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    threshold = d_star(N, f, eps)
    anchor = np.diff(A.indptr) >= threshold
    K = int(st.max())+1
    part = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(N)
    cap = (1+delta)*n/N
    qcap = np.ceil((1+quota_slack)*np.bincount(st[st >= 0], minlength=K)/N)
    qcnt = np.zeros((N, K))
    order = rng.permutation(N)
    ctr = int(rng.integers(N))
    for k in rng.permutation(np.unique(st[anchor])):
        ids = rng.permutation(np.flatnonzero(anchor & (st == k)))
        shards = order[(ctr + np.arange(len(ids))) % N]
        part[ids] = shards
        counts = np.bincount(shards, minlength=N)
        sizes += counts
        if k >= 0:
            qcnt[:, k] += counts
        ctr += len(ids)
    level = _bfs(A.indptr, A.indices, anchor)
    fol = np.flatnonzero(~anchor)
    key = np.where(level[fol] < 0, 1e9, level[fol]) + rng.random(fol.size)
    fallbacks = _followers(A.indptr, A.indices, st, fol[np.argsort(key)],
        part, sizes, qcnt, qcap, cap, rng)
    info = dict(d_star=threshold, anchors=int(anchor.sum()), quota_fallbacks=int(fallbacks),
        follower_backend="numba_serial_numpy_generator", cap=cap)
    return (part, info) if return_info else part


@njit(cache=True)
def _quotient(rowptr, col, w, micro, M):
    W = np.zeros((M, M))
    for v in range(len(micro)):
        a = micro[v]
        for j in range(rowptr[v], rowptr[v+1]):
            b = micro[col[j]]
            if a != b:
                W[a, b] += w[j]/2
    return W + W.T


@njit(cache=True)
def _recombine(H, size, W, n, N, delta, max_sweeps, rng):
    M, K = H.shape
    T = H.sum(0)/N
    norm = 2.0*H.sum()
    cap = (1+delta)*n/N
    assign = np.full(M, -1, dtype=np.int64)
    Hs = np.zeros((N, K))
    S = np.zeros(N)
    Wps = np.zeros((M, N))
    for j in np.argsort(-(size+1e-3*rng.random(M))):
        best, bc = -1, np.inf
        for s in rng.permutation(N):
            if S[s]+size[j] > cap and S[s] > 0:
                continue
            c = (np.abs(Hs[s]+H[j]-T).sum()-np.abs(Hs[s]-T).sum())/norm - Wps[j,s]/n
            if c < bc-1e-12:
                best, bc = s, c
        if best < 0:
            best = np.argmin(S)
        assign[j] = best
        Hs[best] += H[j]
        S[best] += size[j]
        Wps[:,best] += W[:,j]
    sweeps = 0
    for _ in range(max_sweeps):
        sweeps += 1
        improved = False
        for j in rng.permutation(M):
            s = assign[j]
            for t in rng.permutation(N):
                if t == s:
                    continue
                if S[t]+size[j] <= cap:
                    dtv = (np.abs(Hs[t]+H[j]-T).sum()-np.abs(Hs[t]-T).sum()
                           + np.abs(Hs[s]-H[j]-T).sum()-np.abs(Hs[s]-T).sum())/norm
                    if dtv+(Wps[j,s]-Wps[j,t])/n < -1e-9:
                        assign[j] = t
                        Hs[s] -= H[j]
                        Hs[t] += H[j]
                        S[s] -= size[j]
                        S[t] += size[j]
                        Wps[:,s] -= W[:,j]
                        Wps[:,t] += W[:,j]
                        improved = True
                        break
                moved = False
                for k in range(M):
                    if assign[k] != t:
                        continue
                    if S[s]-size[j]+size[k] > cap or S[t]-size[k]+size[j] > cap:
                        continue
                    hs_new = Hs[s]-H[j]+H[k]
                    ht_new = Hs[t]-H[k]+H[j]
                    dtv = (np.abs(hs_new-T).sum()-np.abs(Hs[s]-T).sum()
                           + np.abs(ht_new-T).sum()-np.abs(Hs[t]-T).sum())/norm
                    dw = (Wps[j,s]-(Wps[j,t]-W[j,k]))+(Wps[k,t]-(Wps[k,s]-W[k,j]))
                    if dtv+dw/n < -1e-9:
                        assign[j], assign[k] = t, s
                        Hs[s], Hs[t] = hs_new, ht_new
                        S[s] += size[k]-size[j]
                        S[t] += size[j]-size[k]
                        Wps[:,s] += W[:,k]-W[:,j]
                        Wps[:,t] += W[:,j]-W[:,k]
                        improved, moved = True, True
                        break
                if moved:
                    break
        if not improved:
            break
    return assign, sweeps


def la_metis(A, y, split, N, m=32, delta=0.05, seed=0, max_sweeps=50, return_info=False):
    st = strata(y, split)
    started = time.perf_counter()
    w, wi = neighborhood_loss_weights(A)
    weights_s = time.perf_counter()-started
    M = m*N
    t = time.perf_counter()
    micro = metis_parts(A, M, wi, seed)
    metis_s = time.perf_counter()-t
    del wi
    t = time.perf_counter()
    K = int(st.max())+1
    lab = st >= 0
    H = np.bincount(micro[lab]*K+st[lab], minlength=M*K).reshape(M, K).astype(float)
    size = np.bincount(micro, minlength=M).astype(float)
    W = _quotient(A.indptr, A.indices, w, micro, M)
    assign, sweeps = _recombine(H, size, W, A.shape[0], N, delta, max_sweeps, np.random.default_rng(seed))
    part = assign[micro]
    info = dict(weights_s=weights_s, metis_micro_s=metis_s, regroup_s=time.perf_counter()-t,
        total_s=time.perf_counter()-started, micro_parts=M, sweeps=sweeps,
        metis_backend="torch_sparse.partition", metis_seed="not_exposed_by_existing_binding",
        regroup_seed=seed)
    return (part, info) if return_info else part


def select_partition(A, y, split, N, f=5, eps=0.1, seed=0):
    strata(y, split)
    t = time.perf_counter()
    da, detail = da_random(A, y, split, N, f=f, eps=eps, seed=seed, return_info=True)
    md = fi_metrics(A, y, split, da, N, f)
    candidates = {"da_random": da}
    info = dict(scores={"da_random": md}, t_da=time.perf_counter()-t, DA=detail)
    selected = "da_random"
    if md["SNR"] < 1-eps:
        t = time.perf_counter()
        la, detail = la_metis(A, y, split, N, seed=seed, return_info=True)
        ml = fi_metrics(A, y, split, la, N, f)
        candidates["la_metis"] = la
        info.update(t_la=time.perf_counter()-t, LA=detail)
        info["scores"]["la_metis"] = ml
        if ml["FI"] > md["FI"]:
            selected = "la_metis"
    info["selected"] = selected
    return candidates, info


def select_from_config(config, rowptr, col, y, split):
    from scipy.sparse import csr_matrix
    strata(y, split)
    n = len(rowptr)-1
    A = csr_matrix((np.ones(len(col), dtype=np.int8), col, rowptr), shape=(n,n))
    candidates, detail = select_partition(A, y, split, config.num_gpus,
        f=config.fanout_a, seed=config.partition_seed)
    detail.update(schema="partition_proxy_aware_v1", partition_mode="proxy_aware",
        requested_strategy=config.partition_strategy, random_seed=config.partition_seed,
        random_rng="numpy.default_rng", label_scope="train_val", test_labels_used=False,
        criterion="DA SNR>=0.9; else max(SNR*LA), ties prefer DA-Random",
        candidate_membership_sha256={k:hashlib.sha256(v.astype(np.int64).tobytes()).hexdigest()
                                    for k,v in candidates.items()})
    return candidates, detail
