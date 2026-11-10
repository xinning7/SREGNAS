

from __future__ import annotations
import time
import json
import os
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from sregnas.data.products import (
    ReusableProductsNeighborSampler,
    ProductsNeighborBatch,
)

OPTIONS = {
    "sampler_impl": ("python", "gpu"),
    "feature_residency": ("host", "gpu"),
    "inference_residency": ("host", "gpu"),
}
DEFAULTS = dict(
    sampler_impl="python", feature_residency="host", inference_residency="host"
)


def add_executor_arguments(parser):
    for key, choices in OPTIONS.items():
        parser.add_argument("--" + key.replace("_", "-"), choices=choices, default=None)


def apply_executor_arguments(args, spec):


    inherited = json.loads(os.environ.get("SREGNAS_EXECUTOR_OPTIONS", "{}"))
    if not isinstance(inherited, dict) or set(inherited) - set(DEFAULTS):
        raise ValueError("invalid SREGNAS_EXECUTOR_OPTIONS")
    for key, default in DEFAULTS.items():
        value = getattr(args, key, None)
        spec[key] = (
            value if value is not None else spec.get(key, inherited.get(key, default))
        )
        if spec[key] not in OPTIONS[key]:
            raise ValueError(f"invalid {key}: {spec[key]}")
    return spec


def ordered_unique(values):
    if values.numel() == 0:
        return values
    unique, inverse = torch.unique(values, sorted=True, return_inverse=True)
    first = torch.full(
        (unique.numel(),), values.numel(), device=values.device, dtype=torch.int64
    )
    first.scatter_reduce_(
        0, inverse, torch.arange(values.numel(), device=values.device), reduce="amin"
    )
    return unique[first.argsort()]


def local_indices(nodes, values):
    sorted_nodes, order = torch.sort(nodes)
    return order[torch.searchsorted(sorted_nodes, values)]


@dataclass
class DeviceBatch:
    node_ids: torch.Tensor
    seed_nodes: torch.Tensor
    edge_index_layer1: torch.Tensor
    edge_index_layer2: torch.Tensor

    @property
    def batch_size(self):
        return self.seed_nodes.numel()


class GPUProductsNeighborSampler(ReusableProductsNeighborSampler):
    pass

    def __init__(self, graph, *, device):
        super().__init__(graph)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("GPU sampler requires CUDA")
        self.rowptr = torch.as_tensor(
            np.array(graph.rowptr), device=self.device, dtype=torch.int64
        )
        self.col = torch.as_tensor(
            np.array(graph.col), device=self.device, dtype=torch.int64
        )
        self.pass_generator = None

    def begin_pass(self, seed):
        self.pass_generator = torch.Generator(device=self.device).manual_seed(int(seed))

    def neighbors_device(self, targets, fanout, generator=None):
        targets = torch.as_tensor(targets, device=self.device, dtype=torch.int64)
        starts = self.rowptr[targets]
        degrees = self.rowptr[targets + 1] - starts
        if fanout is None:
            counts = degrees
            dest = torch.repeat_interleave(targets, counts)
            base = torch.repeat_interleave(starts, counts)
            offsets = torch.arange(
                dest.numel(), device=self.device
            ) - torch.repeat_interleave(counts.cumsum(0) - counts, counts)
            return self.col[base + offsets], dest
        k = int(fanout)
        offsets = torch.arange(k, device=self.device).expand(targets.numel(), k).clone()
        random_rows = torch.nonzero(degrees > k).flatten()
        d = degrees[random_rows]
        chosen = torch.empty(
            (random_rows.numel(), k), device=self.device, dtype=torch.int64
        )
        limit = 1 << 62
        cutoff = limit - limit % d
        for j in range(k):
            pending = torch.arange(d.numel(), device=self.device)
            while pending.numel():
                r = torch.randint(
                    limit,
                    (pending.numel(),),
                    generator=generator,
                    device=self.device,
                    dtype=torch.int64,
                )
                v = r % d[pending]
                bad = r >= cutoff[pending]
                if j:
                    bad |= (chosen[pending, :j] == v[:, None]).any(1)
                good = ~bad
                chosen[pending[good], j] = v[good]
                pending = pending[bad]
        offsets[random_rows] = chosen
        keep = torch.arange(k, device=self.device)[None, :] < degrees[:, None]
        source = self.col[(starts[:, None] + offsets)[keep]]
        dest = targets[:, None].expand(-1, k)[keep]
        return source, dest

    def sample_device(self, seed_nodes, *, seed, use_pass_generator=False):
        seeds = ordered_unique(
            torch.as_tensor(seed_nodes, device=self.device, dtype=torch.int64)
        )
        generator = self.pass_generator if use_pass_generator else None
        if generator is None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
        targets = seeds
        output_edges = []
        for k in reversed(self.fanout):
            sources, destinations = self.neighbors_device(targets, k, generator)
            output_edges.append((sources, destinations))
            targets = ordered_unique(torch.cat((targets, sources)))
        nodes = targets
        edges = [
            torch.stack((local_indices(nodes, s), local_indices(nodes, t)))
            for s, t in reversed(output_edges)
        ]
        return DeviceBatch(nodes, seeds, edges[0], edges[1])

    def sample(self, seed_nodes, *, seed):
        batch = self.sample_device(seed_nodes, seed=seed)
        return ProductsNeighborBatch(
            **{
                k: getattr(batch, k).cpu().numpy()
                for k in (
                    "node_ids",
                    "seed_nodes",
                    "edge_index_layer1",
                    "edge_index_layer2",
                )
            }
        )

    def full_neighbor_edges_device(self, targets):
        targets = torch.as_tensor(targets, device=self.device, dtype=torch.int64)
        src, dst = self.neighbors_device(targets, None)
        nodes = ordered_unique(torch.cat((targets, src)))
        return nodes, torch.stack(
            (local_indices(nodes, src), local_indices(nodes, dst))
        )


def configure_executor(graph, sampler, *, device, **options):
    opts = {k: options.get(k, v) for k, v in DEFAULTS.items()}
    for k, v in opts.items():
        if v not in OPTIONS[k]:
            raise ValueError(f"invalid {k}: {v}")
    if opts == DEFAULTS:
        return sampler
    if torch.device(device).type != "cuda":
        raise ValueError("resident executor requires CUDA")
    if opts["sampler_impl"] == "gpu":
        sampler = GPUProductsNeighborSampler(graph, device=device)
    sampler.executor_options = opts
    sampler.executor_device = torch.device(device)
    return sampler


def device_sampler(sampler):
    if isinstance(sampler, GPUProductsNeighborSampler):
        return sampler
    if not hasattr(sampler, "_gpu_structure"):
        sampler._gpu_structure = GPUProductsNeighborSampler(
            sampler.graph, device=sampler.executor_device
        )
    return sampler._gpu_structure


def resident_features(sampler, graph, device):
    if not hasattr(sampler, "_resident_x"):
        sampler._resident_x = torch.as_tensor(np.array(graph.x), device=device)
        sampler._resident_labels = torch.as_tensor(
            np.array(graph.labels), device=device, dtype=torch.int64
        )
    return sampler._resident_x, sampler._resident_labels


def training_sample(sampler, seeds, *, root_seed, pass_index, batch_index):
    from sregnas.training.minibatch import _subseed

    seed = _subseed(root_seed, "neighbor_sample", pass_index, batch_index)
    if isinstance(sampler, GPUProductsNeighborSampler):
        return sampler.sample_device(seeds, seed=seed)
    return sampler.sample(seeds, seed=seed)


def training_inputs(graph, sampler, batch, *, device):
    pass
    from sregnas.training.minibatch import _copy_rows

    if getattr(sampler, "executor_options", DEFAULTS)["feature_residency"] == "gpu":
        xall, yall = resident_features(sampler, graph, device)
        ids = torch.as_tensor(batch.node_ids, device=device, dtype=torch.int64)
        seeds = torch.as_tensor(batch.seed_nodes, device=device, dtype=torch.int64)
        x, labels = xall.index_select(0, ids), yall.index_select(0, seeds)
    else:
        ids = (
            batch.node_ids.cpu().numpy()
            if isinstance(batch.node_ids, torch.Tensor)
            else batch.node_ids
        )
        seeds = (
            batch.seed_nodes.cpu().numpy()
            if isinstance(batch.seed_nodes, torch.Tensor)
            else batch.seed_nodes
        )
        x = _copy_rows(graph.x, ids).to(device)
        labels = _copy_rows(graph.labels, seeds).long().to(device)
    edges = [
        torch.as_tensor(e, device=device, dtype=torch.int64)
        for e in (batch.edge_index_layer1, batch.edge_index_layer2)
    ]
    from sregnas.models.normalization_fix import enabled, attach_degrees
    if enabled():
        edges = attach_degrees(graph, sampler, batch, edges, device)
    return x, edges, labels


def train_resident(
    model, graph, sampler, optimizer, *, device, root_seed, passes, pass_callback=None
):
    from sregnas.training.minibatch import (
        ProductsTrainResult,
        _subseed,
        _copy_rows,
        _torch_edge,
    )

    if passes <= 0:
        raise ValueError("products passes must be positive")
    started = time.perf_counter()
    opts = sampler.executor_options
    init_started = time.perf_counter()
    xall = yall = None
    if opts["feature_residency"] == "gpu":
        xall, yall = resident_features(sampler, graph, device)
    torch.cuda.synchronize(device)
    init_s = time.perf_counter() - init_started
    sampling_s = h2d_s = compute_s = 0.0
    steps = seen = 0
    final_loss = float("nan")
    final_batch_size = 0
    expected = int(graph.train_mask.sum())
    for p in range(passes):
        pass_start = time.perf_counter()
        before = (sampling_s, h2d_s, compute_s)
        count = 0
        for b, seeds in enumerate(
            sampler.seed_batches(
                "train", seed=_subseed(root_seed, "train_order", p), shuffle=True
            )
        ):
            t = time.perf_counter()
            batch = training_sample(
                sampler, seeds, root_seed=root_seed, pass_index=p, batch_index=b
            )
            torch.cuda.synchronize(device)
            sampling_s += time.perf_counter() - t
            t = time.perf_counter()
            x, edges, labels = training_inputs(graph, sampler, batch, device=device)
            torch.cuda.synchronize(device)
            h2d_s += time.perf_counter() - t
            t = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, edges)[: batch.batch_size]
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite products loss: {loss.item()}")
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)
            compute_s += time.perf_counter() - t
            final_loss = float(loss.detach().item())
            steps += 1
            count += batch.batch_size
            final_batch_size = batch.batch_size
        if count != expected:
            raise AssertionError("pass did not cover all training seeds")
        seen += count
        if pass_callback:
            pass_callback(
                dict(
                    pass_index=p,
                    wall_s=time.perf_counter() - pass_start,
                    sampling_s=sampling_s - before[0],
                    h2d_s=h2d_s - before[1],
                    train_compute_s=compute_s - before[2],
                )
            )
    total = time.perf_counter() - started
    return ProductsTrainResult(
        passes,
        steps,
        seen,
        final_batch_size,
        final_loss,
        max(0.0, total - sampling_s - h2d_s - compute_s),
        sampling_s,
        h2d_s,
        compute_s,
        total,
    )


def exact_resident(
    canonical_model, graph, sampler, *, device, splits, batch_size, collect_logits=False
):
    from sregnas.training.minibatch import ProductsInferenceResult, _exact_layer_batch

    started = time.perf_counter()
    canonical_model.eval()
    features, labels = resident_features(sampler, graph, device)
    gpu = device_sampler(sampler)
    degrees = (gpu.rowptr[1:] - gpu.rowptr[:-1] + 1).float()
    n = graph.num_nodes
    batches = 0
    with torch.no_grad():
        for conv, activation in zip(canonical_model.convs, canonical_model._act_names):
            output = torch.empty(
                (n, conv.out_channels), device=device, dtype=torch.float32
            )
            for start in range(0, n, batch_size):
                targets = torch.arange(start, min(n, start + batch_size), device=device)
                output[start : start + targets.numel()] = _exact_layer_batch(
                    conv,
                    activation,
                    features,
                    graph,
                    gpu,
                    targets,
                    device=device,
                    global_degrees_with_loops=degrees,
                )
                batches += 1
            features = output
        masks = {
            s: torch.as_tensor(
                np.array(getattr(graph, s + "_mask")), device=device, dtype=torch.bool
            )
            for s in splits
        }
        counts = {s: int(m.sum()) for s, m in masks.items()}
        if any(v == 0 for v in counts.values()):
            raise AssertionError("empty inference split")
        correct = {s: torch.zeros((), device=device, dtype=torch.int64) for s in splits}
        logits_all = (
            torch.empty((n, graph.num_classes), dtype=torch.float32)
            if collect_logits
            else None
        )
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            logits = canonical_model.classifier(features[start:stop])
            pred = logits.argmax(1)
            for s, m in masks.items():
                correct[s] += ((pred == labels[start:stop]) & m[start:stop]).sum()
            if logits_all is not None:
                logits_all[start:stop] = logits.cpu()
    torch.cuda.synchronize(device)
    acc = {s: int(correct[s]) / counts[s] for s in splits}
    return ProductsInferenceResult(
        acc.get("val"),
        acc.get("test"),
        batches,
        time.perf_counter() - started,
        logits_all,
    )
