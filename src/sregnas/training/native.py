

import time
import numpy as np
import torch
from torch_geometric.nn import GCNConv, GATConv
from sregnas.training.minibatch import ProductsInferenceResult


class FastPerLayerModel(torch.nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native

    def forward(self, x, edges):
        assert len(edges) == len(self.native.layers)
        for i, (layer, act, edge) in enumerate(
            zip(self.native.layers, self.native.activations, edges)
        ):
            if i < len(edges) - 1:
                x = torch.nn.functional.dropout(
                    x, p=self.native.dropout, training=self.training
                )
            x = act(layer(x, edge))
        return x


def exact_fastgnas_inference(
    model, graph, sampler, *, device, splits, batch_size=128, collect_logits=False
):
    if (
        not splits
        or len(set(splits)) != len(splits)
        or not set(splits) <= {"val", "test"}
    ):
        raise ValueError("Explicit unique validation/test splits required")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.eval()
    started = time.perf_counter()
    resident = (
        getattr(sampler, "executor_options", {}).get("inference_residency") == "gpu"
    )
    if resident:
        from sregnas.training.executor import device_sampler, resident_features

        gpu_sampler = device_sampler(sampler)
        features, _ = resident_features(sampler, graph, device)
    else:
        features = graph.x
    degree = torch.from_numpy(np.array(np.diff(graph.rowptr) + 1)).float()
    if resident:
        degree = degree.to(device)
    batches = 0
    with torch.no_grad():
        for layer, act in zip(model.layers, model.activations):
            if not isinstance(layer, (GCNConv, GATConv)):
                raise TypeError(
                    f"Exact Reddit inference not validated for {type(layer)}"
                )
            width = layer.out_channels * (
                layer.heads if isinstance(layer, GATConv) and layer.concat else 1
            )
            output = torch.empty(
                (graph.num_nodes, width),
                dtype=torch.float32,
                device=device if resident else "cpu",
            )
            for targets in sampler.inference_batches(batch_size):
                if resident:
                    nodes, edge = gpu_sampler.full_neighbor_edges_device(targets)
                    ids = nodes
                else:
                    nodes, edge = sampler.full_neighbor_edges(targets)
                    ids = torch.from_numpy(nodes).long()
                x = (
                    features.index_select(0, ids)
                    if isinstance(features, torch.Tensor)
                    else torch.from_numpy(np.array(features[nodes]))
                ).to(device)
                edge = torch.as_tensor(edge, dtype=torch.int64, device=device)
                if isinstance(layer, GCNConv):
                    assert (
                        layer.normalize and layer.add_self_loops and not layer.improved
                    )
                    h = layer.lin(x)
                    loops = torch.arange(len(targets), device=device)
                    src = torch.cat((edge[0], loops))
                    dst = torch.cat((edge[1], loops))
                    norm = degree.index_select(0, ids).to(device).pow(-0.5)
                    msg = h[src] * (norm[src] * norm[dst]).unsqueeze(1)
                    out = torch.zeros((len(targets), width), device=device)
                    out.index_add_(0, dst, msg)
                    if layer.bias is not None:
                        out += layer.bias
                else:
                    out = layer(x, edge)[: len(targets)]
                output[torch.as_tensor(targets, device=output.device).long()] = act(
                    out
                ).to(output.device)
                batches += 1
            features = output
        accuracies = {}
        labels = torch.as_tensor(np.array(graph.labels), device=features.device).long()
        predictions = features.argmax(1)
        for split in splits:
            mask = torch.as_tensor(
                np.array(getattr(graph, split + "_mask")), device=features.device
            ).bool()
            if not mask.any():
                raise ValueError("empty inference split")
            accuracies[split] = float(
                (predictions[mask] == labels[mask]).float().mean()
            )
    return ProductsInferenceResult(
        val_acc=accuracies.get("val"),
        test_acc=accuracies.get("test"),
        layer_batches=batches,
        inference_s=time.perf_counter() - started,
        logits=features.cpu() if collect_logits else None,
    )
