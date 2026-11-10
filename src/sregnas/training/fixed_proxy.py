
from __future__ import annotations

from dataclasses import dataclass
import os
import time

import torch
import torch.nn.functional as F

STEPS_ENV = "SREGNAS_PROXY_STEPS"
BATCH_ENV = "SREGNAS_PROXY_BATCH_TARGETS"
AP_TARGETS_ENV = "SREGNAS_PROXY_AP_TARGETS"
SCHEMA = "fixed_layer_target_batch_v1"
AP_TARGET_SCHEMA = "ap_fresh_loader_first_batch_v1"


def options(spec=None):
    spec = spec or {}
    values = {}
    for key, env in (("proxy_steps", STEPS_ENV), ("proxy_batch_targets", BATCH_ENV)):
        value = spec.get(key, os.environ.get(env))
        if value is not None:
            value = int(value)
            if value <= 0:
                raise ValueError(f"{key} must be positive")
            values[key] = value
    if spec.get("proxy_ap_targets", os.environ.get(AP_TARGETS_ENV) == "1"):
        values["proxy_ap_targets"] = True
    return values


def apply_options(spec):
    if spec.get("mode") != "proxy":
        return spec
    opts = options(spec)
    if not opts:
        return spec
    spec = dict(spec, **opts)
    if "proxy_steps" in opts:
        spec["epochs"] = opts["proxy_steps"]
    return spec


def index_state(edges, num_nodes):
    pass
    layers = []
    for edge in edges:
        order = torch.argsort(edge[1], stable=True)
        counts = torch.bincount(edge[1], minlength=num_nodes)
        ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        nonself = edge[0] != edge[1]
        degree = torch.bincount(edge[0, nonself], minlength=num_nodes) + 1
        layers.append(dict(order=order, ptr=ptr, degree=degree))
    return layers


def add_prepared_index(payload, *, shard=None):
    pass
    if BATCH_ENV not in os.environ:
        return
    from sregnas.icde27.s0 import span
    with span("proxy_fixed_index", shard=shard):
        payload["proxy_fixed_index"] = index_state(
            [payload["edge_index_layer1"], payload["edge_index_layer2"]],
            len(payload["x"]),
        )


@dataclass
class FixedBatch:
    nodes: torch.Tensor
    targets: torch.Tensor
    edges: list[torch.Tensor]
    original_edge_ids: list[torch.Tensor]
    degrees: list[torch.Tensor]


class FixedProxyIndex:
    def __init__(self, edges, num_nodes, state=None):
        if len(edges) != 2:
            raise ValueError("fixed proxy requires exactly two layers")
        self.edges = edges
        self.num_nodes = num_nodes
        device = edges[0].device
        state = index_state(edges, num_nodes) if state is None else state
        self.layers = [{k: v.to(device) for k, v in layer.items()} for layer in state]

    def incoming(self, layer, targets):
        index = self.layers[layer]
        starts = index["ptr"][targets]
        counts = index["ptr"][targets + 1] - starts
        offsets = torch.repeat_interleave(starts - counts.cumsum(0) + counts, counts)
        positions = offsets + torch.arange(offsets.numel(), device=targets.device)

        ids = index["order"][positions].sort().values
        return ids, self.edges[layer][:, ids]

    def slice(self, targets):
        targets = targets.sort().values
        ids2, e2 = self.incoming(1, targets)
        middle = torch.unique(torch.cat((targets, e2[0])), sorted=True)
        ids1, e1 = self.incoming(0, middle)
        nodes = torch.unique(torch.cat((middle, e1[0])), sorted=True)
        return FixedBatch(
            nodes=nodes, targets=torch.searchsorted(nodes, targets),
            edges=[torch.searchsorted(nodes, edge) for edge in (e1, e2)],
            original_edge_ids=[ids1, ids2],
            degrees=[layer["degree"][nodes] for layer in self.layers],
        )


def forward_batch(model, x, batch):
    pass
    from torch_geometric.utils import add_remaining_self_loops
    canonical = model.canonical_model
    if canonical.dropout != 0:
        raise ValueError("fixed proxy is scoped to the frozen dropout=0 protocol")
    h = x[batch.nodes]
    for layer, (conv, act) in enumerate(zip(canonical.convs, canonical._act_names)):
        edge = batch.edges[layer]
        if conv.encoding_op == "gcn":
            projected = conv.lin(h)
            ei, _ = add_remaining_self_loops(edge, num_nodes=len(h))
            norm = batch.degrees[layer].to(projected.dtype).clamp(min=1).pow(-0.5)
            conv._cur_edge_weight = norm[ei[0]] * norm[ei[1]]
            try:
                h = conv.propagate(ei, x=projected)
            finally:
                conv._cur_edge_weight = None
            if conv.aggregation_op == "mlp":
                h = conv.mlp_agg(h)
            h = h + conv.bias
        else:
            h = conv(h, edge)
        h = canonical._act(h, act)
    return canonical.classifier(h[batch.targets])


class APTargetDraws:
    pass

    def __init__(self, train_nodes, batch_targets, seed):
        self.nodes = train_nodes
        self.batch_targets = int(batch_targets)
        self.rng_state = torch.Generator(device="cpu").manual_seed(seed).get_state()

    def draw(self):
        from torch.utils.data import DataLoader
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.rng_state)
            loader = DataLoader(range(len(self.nodes)), batch_size=self.batch_targets,
                                shuffle=True, num_workers=0, pin_memory=False)
            positions = next(iter(loader))
            self.rng_state = torch.get_rng_state()
        return self.nodes[positions.to(self.nodes.device)]


def train_proxy_batches(model, optimizer, x, edges, labels, train_mask, val_mask,
                        *, epochs, batch_targets, seed, index, ap_targets=False):
    from .fullbatch import ProxyTrainResult, _accuracy
    train_nodes = train_mask.nonzero(as_tuple=False).flatten()
    if not train_nodes.numel():
        raise ValueError("proxy training mask is empty")
    generator = torch.Generator(device=x.device).manual_seed(seed)
    size = min(int(batch_targets), len(train_nodes))
    draws = APTargetDraws(train_nodes, batch_targets, seed) if ap_targets else None
    prepared = None
    if os.environ.get('SREGNAS_X19_PREBUILD_BATCHES') == '1':
        if not ap_targets:
            raise ValueError('X19 prebuilding requires the frozen AP target stream')
        from sregnas.icde27.x19.proxy import prebuild
        prepared = prebuild(index, train_nodes, epochs, batch_targets, seed)
    for step in range(epochs):
        if prepared is None:
            targets = draws.draw() if draws is not None else (train_nodes if size == len(train_nodes) else
                       train_nodes[torch.randperm(len(train_nodes), generator=generator,
                                                 device=x.device)[:size]])
            targets = targets.sort().values
            batch = index.slice(targets)
        else:
            from sregnas.icde27.x19.proxy import prepared_step
            targets, batch = prepared_step(prepared, step)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(forward_batch(model, x, batch), labels[targets])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite proxy loss: {loss.item()}")
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        value = _accuracy(model(x, edges), labels, val_mask)
    return ProxyTrainResult(sub_acc=value, epochs=epochs)


def payload_index(payload, edges, device):
    cached = payload.get("_proxy_fixed_runtime_index")
    if cached is None:
        started = time.perf_counter()
        cached = FixedProxyIndex(edges, len(payload["x"]), payload.get("proxy_fixed_index"))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        payload["_proxy_fixed_runtime_index"] = cached
        payload["_proxy_fixed_index_load_s"] = time.perf_counter() - started
    return cached
