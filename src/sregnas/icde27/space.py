

import hashlib
import itertools
import json

NAME = "icde27_token_intersection_v1"
FIELDS = {
    "encoding_op": ["const", "gcn", "gat", "sym-gat", "cos", "linear", "gene-linear"],
    "aggregation_op": ["sum", "mean", "max", "mlp"],
    "activation": ["sigmoid", "tanh", "relu", "none", "softplus", "leaky_relu", "relu6", "elu"],
    "num_heads": [1, 2, 4, 8],
    "hidden_dim": [8, 16, 32, 64, 128, 256],
}


def canonical_layers(architecture):
    if isinstance(architecture, list):
        return architecture
    return [dict(encoding_op=l["encoding"], aggregation_op=l["aggregation"],
                 activation=l["activation"], num_heads=l["num_heads"],
                 hidden_dim=l["hidden_dim"]) for l in architecture["layers"]]


def native_spec(layers):
    return dict(num_layers=2, layers=[dict(encoding=l["encoding_op"],
        aggregation=l["aggregation_op"], activation=l["activation"],
        num_heads=l["num_heads"], hidden_dim=l["hidden_dim"]) for l in layers])


def canonical_fingerprint(architecture):
    from sregnas.core.architecture import Architecture
    return Architecture(canonical_layers(architecture)).fingerprint()


def legal_layer(layer):
    return (set(layer) == set(FIELDS)
            and all(layer[k] in values for k, values in FIELDS.items())
            and (layer["num_heads"] == 1 if layer["encoding_op"] not in {"gat", "sym-gat"}
                 else layer["hidden_dim"] % layer["num_heads"] == 0))


def layer_space():
    return [l for values in itertools.product(*FIELDS.values())
            if legal_layer(l := dict(zip(FIELDS, values)))]


def definition():

    return dict(schema=NAME, num_layers=2, fields=FIELDS,
                layer_constraint="nonattention heads=1; attention hidden_dim divisible by heads",
                pair_constraint="cartesian product; both backend static gates pass")


def space_hash():
    return hashlib.sha256(json.dumps(definition(), sort_keys=True,
        separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def validate_pair(layers, dataset):
    from sregnas.models.gnn import _canonicalize_layer
    from sregnas.data.policy import system_static_gate_enabled
    from sregnas.search.memory_gate import arxiv_static_memory_gate_rejects
    from sregnas.baselines.fastgnas.core.scheduling.base import SchedulerInterface
    if len(layers) != 2 or not all(legal_layer(l) for l in layers):
        raise ValueError("architecture outside ICDE27 token intersection")
    assert [_canonicalize_layer(l) for l in layers] == layers
    if system_static_gate_enabled(dataset) and arxiv_static_memory_gate_rejects(layers):
        raise ValueError("canonical static gate rejected architecture")

    SchedulerInterface.validate_architectures(None, [dict(arch_id="static", arch_spec=native_spec(layers))])
    return True
