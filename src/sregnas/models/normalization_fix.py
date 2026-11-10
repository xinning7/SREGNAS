
import os
import torch
import torch.nn.functional as F
from torch_geometric.utils import softmax

FLAG = 'SREGNAS_CANONICAL_NORMALIZATION_FIX'
SCHEMA = 'canonical_full_degree_ht_v1'
SOFTMAX_MAX_FLAG = 'SREGNAS_PI_SOFTMAX_MAX_FIX'


def enabled():
    return os.environ.get(FLAG, '1') == '1'


def softmax_max_enabled():
    pass
    return os.environ.get(SOFTMAX_MAX_FLAG, '0') == '1'


class DegreeEdges(list):
    def __init__(self, edges, full_degrees):
        super().__init__(edges)
        self.nf_full_degrees = full_degrees


def attach_degrees(graph, sampler, batch, edges, device):
    pass
    if not enabled():return edges
    if not hasattr(sampler, '_nf_full_degrees'):
        from sregnas.training.executor import GPUProductsNeighborSampler
        if isinstance(sampler,GPUProductsNeighborSampler):
            sampler._nf_full_degrees = sampler.rowptr[1:] - sampler.rowptr[:-1]
        else:
            import numpy as np
            sampler._nf_full_degrees = torch.as_tensor(np.diff(graph.rowptr),device=device)
    ids=torch.as_tensor(batch.node_ids,device=device,dtype=torch.int64)
    return DegreeEdges(edges,sampler._nf_full_degrees.index_select(0,ids))


def forward(conv, x, edge_index, full_degrees, *, projected=None):
    pass
    n=x.size(0);h=conv.lin(x) if projected is None else projected
    degrees=full_degrees.to(device=h.device,dtype=h.dtype)
    if degrees.shape!=(n,):raise ValueError('full degree context shape mismatch')

    source,target=edge_index[:,edge_index[0]!=edge_index[1]]
    count=torch.bincount(target,minlength=n).to(h.dtype)
    if bool((count>degrees).any()):raise ValueError('sample exceeds full graph degree')
    ratio=degrees/count.clamp_min(1)
    weight=torch.cat((ratio[target],torch.ones(n,device=h.device,dtype=h.dtype)))
    ids=torch.arange(n,device=h.device)
    source=torch.cat((source,ids));target=torch.cat((target,ids))
    op=conv.encoding_op
    effective='sum' if op=='gcn' else conv.aggregation_op
    if op=='const':
        message=h[source]
        if effective!='max':message=message*weight[:,None]
    elif op=='gcn':
        norm=(degrees+1).pow(-.5)
        message=h[source]*(norm[source]*norm[target]*weight)[:,None]
    else:
        if op in ('gat','sym-gat'):
            hh=h.view(n,conv.num_heads,conv._head_dim)
            logits=((hh[source]*conv.att_src).sum(-1)+(hh[target]*conv.att_dst).sum(-1)
                    if op=='gat' else ((hh[source]+hh[target])*conv.att).sum(-1))
            logits=F.leaky_relu(logits,.2)
            values=hh[source]
        else:
            values=h[source]
            if op=='cos':
                hn=F.normalize(h,dim=-1);logits=(hn[source]*hn[target]).sum(-1,keepdim=True)
            elif op=='linear':logits=(h[source]*h[target]).sum(-1,keepdim=True)/(conv.out_channels**.5)
            elif op=='gene-linear':logits=(h[source]*h[target]*conv.att_vec).sum(-1,keepdim=True)
            else:raise ValueError(op)


        alpha=softmax(logits+weight.log()[:,None],target,num_nodes=n)
        if effective=='max' and softmax_max_enabled():
            alpha=alpha/weight[:,None]
        alpha=F.dropout(alpha,p=conv.dropout_p,training=conv.training)
        message=values*(alpha.unsqueeze(-1) if values.dim()==3 else alpha)
        message=message.reshape(-1,conv.out_channels)
    if effective=='max':
        output=torch.full((n,conv.out_channels),-torch.inf,device=h.device,dtype=h.dtype)
        output.scatter_reduce_(0,target[:,None].expand_as(message),message,reduce='amax',include_self=True)
    else:
        output=torch.zeros((n,conv.out_channels),device=h.device,dtype=h.dtype)


        output.scatter_add_(0,target[:,None].expand_as(message),message)
        if effective=='mean':output=output/(degrees+1)[:,None]
    if conv.aggregation_op=='mlp':output=conv.mlp_agg(output)
    return output+conv.bias
