
import numpy as np
import torch


def gpu_csr(edge_index, num_nodes, device='cuda:0'):
    edge = torch.as_tensor(np.asarray(edge_index), dtype=torch.long, device=device)
    if edge.ndim != 2 or edge.shape[0] != 2:
        raise ValueError('edge_index must have shape [2, E]')
    src, dst = edge
    valid = (src != dst) & (src >= 0) & (dst >= 0) & (src < num_nodes) & (dst < num_nodes)
    src, dst = src[valid], dst[valid]
    packed = torch.cat((src * num_nodes + dst, dst * num_nodes + src)).sort().values
    packed = torch.unique_consecutive(packed)
    counts = torch.bincount(packed // num_nodes, minlength=num_nodes)
    rowptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    col = (packed % num_nodes).to(torch.int32)
    return rowptr.cpu().numpy(), col.cpu().numpy()


def gpu_fixed_edges(rowptr, permutations, membership, caps, num_parts, device='cuda:0'):
    pass
    n = len(rowptr) - 1
    ptr = torch.as_tensor(rowptr, dtype=torch.long, device=device)
    member = torch.as_tensor(membership, dtype=torch.long, device=device)
    receptive = torch.repeat_interleave(torch.arange(n, device=device), ptr[1:] - ptr[:-1],
                                       output_size=int(rowptr[-1]))
    local = torch.empty(n, dtype=torch.long, device=device)
    for part in range(num_parts):
        nodes = (member == part).nonzero().flatten()
        local[nodes] = torch.arange(len(nodes), device=device)
    output = {part: [] for part in range(num_parts)}
    internal = None
    for permutation, cap in zip(permutations, caps):
        neighbor = torch.as_tensor(permutation, dtype=torch.long, device=device)
        inside = member[neighbor] == member[receptive]
        prefix = torch.cat((ptr.new_zeros(1), inside.long().cumsum(0)))
        rank = prefix[1:] - 1 - prefix[ptr[:-1]][receptive]
        keep = inside & (rank < cap)
        if internal is None:
            internal = (prefix[ptr[1:]] - prefix[ptr[:-1]]).cpu().numpy()
        kept_rows, kept_neighbors = receptive[keep], neighbor[keep]
        parts = member[kept_rows]
        for part in range(num_parts):
            chosen = parts == part
            edge = torch.stack((local[kept_neighbors[chosen]], local[kept_rows[chosen]]))
            output[part].append(edge.cpu().contiguous())
    return output, internal
