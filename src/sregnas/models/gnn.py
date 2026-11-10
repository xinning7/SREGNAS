

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, add_remaining_self_loops, degree
from sregnas.core import Architecture
from .codegen import configure_project_codegen

configure_project_codegen()

SEARCH_SPACE = {
    "encoding_op": ["const", "gcn", "gat", "sym-gat", "cos", "linear", "gene-linear"],
    "aggregation_op": ["sum", "mean", "max", "mlp"],
    "activation": [
        "sigmoid",
        "tanh",
        "relu",
        "none",
        "softplus",
        "leaky_relu",
        "relu6",
        "elu",
    ],
    "num_heads": [1, 2, 4, 6, 8, 16],
    "hidden_dim": [8, 16, 32, 64, 128, 256, 512],
}

ACTIVATION_FNS = {
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "relu": torch.relu,
    "none": lambda x: x,
    "softplus": F.softplus,
    "leaky_relu": F.leaky_relu,
    "relu6": F.relu6,
    "elu": F.elu,
}



_MULTIHEAD_OPS = {"gat", "sym-gat"}


def _canonicalize_layer(layer: dict) -> dict:
    pass
    layer = dict(layer)
    enc = layer["encoding_op"]
    if enc not in _MULTIHEAD_OPS:
        layer["num_heads"] = 1
    else:

        layer["num_heads"] = _valid_heads(layer["hidden_dim"], layer["num_heads"])
    return layer


def _make_sample_fn(num_layers: int):
    pass

    def _sample() -> list:
        return [
            _canonicalize_layer({k: random.choice(v) for k, v in SEARCH_SPACE.items()})
            for _ in range(num_layers)
        ]

    return _sample


def _valid_heads(out_channels: int, requested: int) -> int:
    pass
    for h in range(min(requested, out_channels), 0, -1):
        if out_channels % h == 0:
            return h
    return 1



class GNASConv(MessagePassing):
    pass

    _VALID_ENCODING_OPS = frozenset(
        ["const", "gcn", "gat", "sym-gat", "cos", "linear", "gene-linear"]
    )
    _VALID_AGGREGATION_OPS = frozenset(["sum", "mean", "max", "mlp"])

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoding_op: str,
        aggregation_op: str,
        num_heads: int = 1,
        dropout: float = 0.0,
    ):

        if encoding_op not in self._VALID_ENCODING_OPS:
            raise ValueError(
                f"encoding_op={encoding_op!r} 不合法，"
                f"合法值：{sorted(self._VALID_ENCODING_OPS)}"
            )
        if aggregation_op not in self._VALID_AGGREGATION_OPS:
            raise ValueError(
                f"aggregation_op={aggregation_op!r} 不合法，"
                f"合法值：{sorted(self._VALID_AGGREGATION_OPS)}"
            )
        if out_channels <= 0:
            raise ValueError(f"out_channels 必须为正整数，得到 {out_channels}")
        if num_heads <= 0:
            raise ValueError(f"num_heads 必须为正整数，得到 {num_heads}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout 必须在 [0, 1) 范围内，得到 {dropout}")





        if encoding_op == "gcn":
            aggr = "add"
        else:
            aggr = {"sum": "add", "mean": "mean", "max": "max", "mlp": "add"}[
                aggregation_op
            ]
        super().__init__(aggr=aggr, node_dim=0)

        self.encoding_op = encoding_op
        self.aggregation_op = aggregation_op
        self.dropout_p = dropout
        self.out_channels = out_channels


        if encoding_op in ("gat", "sym-gat"):
            self.num_heads = _valid_heads(out_channels, num_heads)
            self._head_dim = out_channels // self.num_heads
        else:
            self.num_heads = 1
            self._head_dim = out_channels

        self.lin = nn.Linear(in_channels, out_channels, bias=False)

        if encoding_op == "gat":
            self.att_src = nn.Parameter(torch.empty(1, self.num_heads, self._head_dim))
            self.att_dst = nn.Parameter(torch.empty(1, self.num_heads, self._head_dim))
            nn.init.xavier_uniform_(self.att_src)
            nn.init.xavier_uniform_(self.att_dst)
        elif encoding_op == "sym-gat":
            self.att = nn.Parameter(torch.empty(1, self.num_heads, self._head_dim))
            nn.init.xavier_uniform_(self.att)
        elif encoding_op == "gene-linear":
            self.att_vec = nn.Parameter(torch.ones(out_channels))

        if aggregation_op == "mlp":
            self.mlp_agg = nn.Sequential(
                nn.Linear(out_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )

        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.lin.weight)

        self._cur_alpha: torch.Tensor = None
        self._cur_edge_weight: torch.Tensor = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        h = self.lin(x)




        ei_sl, _ = add_remaining_self_loops(edge_index, num_nodes=N)
        r_sl, c_sl = ei_sl

        if self.encoding_op == "const":
            out = self.propagate(ei_sl, x=h)

        elif self.encoding_op == "gcn":
            deg = degree(r_sl, N, dtype=h.dtype).clamp(min=1)
            norm = deg.pow(-0.5)
            self._cur_edge_weight = norm[r_sl] * norm[c_sl]
            out = self.propagate(ei_sl, x=h)
            self._cur_edge_weight = None

        elif self.encoding_op == "gat":
            hh = h.view(N, self.num_heads, self._head_dim)
            alpha = (hh[r_sl] * self.att_src).sum(-1) + (hh[c_sl] * self.att_dst).sum(
                -1
            )
            alpha = F.leaky_relu(alpha, 0.2)
            alpha = softmax(alpha, c_sl, num_nodes=N)
            self._cur_alpha = F.dropout(alpha, p=self.dropout_p, training=self.training)
            out = self.propagate(ei_sl, x=hh)
            out = out.reshape(-1, self.out_channels)
            self._cur_alpha = None

        elif self.encoding_op == "sym-gat":
            hh = h.view(N, self.num_heads, self._head_dim)
            alpha = ((hh[r_sl] + hh[c_sl]) * self.att).sum(-1)
            alpha = F.leaky_relu(alpha, 0.2)
            alpha = softmax(alpha, c_sl, num_nodes=N)
            self._cur_alpha = F.dropout(alpha, p=self.dropout_p, training=self.training)
            out = self.propagate(ei_sl, x=hh)
            out = out.reshape(-1, self.out_channels)
            self._cur_alpha = None

        elif self.encoding_op == "cos":
            h_n = F.normalize(h, dim=-1)
            alpha = (h_n[r_sl] * h_n[c_sl]).sum(-1, keepdim=True)
            alpha = softmax(alpha, c_sl, num_nodes=N)
            self._cur_alpha = F.dropout(alpha, p=self.dropout_p, training=self.training)
            out = self.propagate(ei_sl, x=h)
            self._cur_alpha = None

        elif self.encoding_op == "linear":
            alpha = (h[r_sl] * h[c_sl]).sum(-1, keepdim=True) / (self.out_channels**0.5)
            alpha = softmax(alpha, c_sl, num_nodes=N)
            self._cur_alpha = F.dropout(alpha, p=self.dropout_p, training=self.training)
            out = self.propagate(ei_sl, x=h)
            self._cur_alpha = None

        elif self.encoding_op == "gene-linear":
            alpha = (h[r_sl] * h[c_sl] * self.att_vec).sum(-1, keepdim=True)
            alpha = softmax(alpha, c_sl, num_nodes=N)
            self._cur_alpha = F.dropout(alpha, p=self.dropout_p, training=self.training)
            out = self.propagate(ei_sl, x=h)
            self._cur_alpha = None

        else:
            out = self.propagate(ei_sl, x=h)

        if self.aggregation_op == "mlp":
            out = self.mlp_agg(out)

        return out + self.bias

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        if self._cur_alpha is not None:
            alpha = self._cur_alpha

            if alpha.dim() == 2 and x_j.dim() == 3:
                return x_j * alpha.unsqueeze(-1)
            return alpha * x_j
        if self._cur_edge_weight is not None:
            return self._cur_edge_weight.view(-1, 1) * x_j
        return x_j



class GNASModel(nn.Module):
    pass

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        layer_configs: list,
        dropout: float = 0.0,
    ):
        super().__init__()


        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout 必须在 [0, 1) 范围内，得到 {dropout}")
        if not layer_configs:
            raise ValueError("layer_configs 不能为空")
        for i, cfg in enumerate(layer_configs):
            if cfg.get("hidden_dim", 0) <= 0:
                raise ValueError(
                    f"layer_configs[{i}].hidden_dim 必须为正整数，"
                    f"得到 {cfg.get('hidden_dim')}"
                )
            act = cfg.get("activation")
            if act not in ACTIVATION_FNS:
                raise ValueError(
                    f"layer_configs[{i}].activation={act!r} 不合法，"
                    f"合法值：{sorted(ACTIVATION_FNS)}"
                )


        self.dropout = dropout
        self.convs = nn.ModuleList()
        self._act_names: list = []
        current_dim = in_features

        for cfg in layer_configs:
            out_dim = cfg["hidden_dim"]
            self.convs.append(
                GNASConv(
                    current_dim,
                    out_dim,
                    encoding_op=cfg["encoding_op"],
                    aggregation_op=cfg["aggregation_op"],
                    num_heads=cfg["num_heads"],
                    dropout=dropout,
                )
            )
            self._act_names.append(cfg["activation"])
            current_dim = out_dim

        self.classifier = nn.Linear(current_dim, num_classes)

    def _act(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return ACTIVATION_FNS[name](x)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        halo_layer_embeddings=None,
        num_local_nodes=None,
    ) -> torch.Tensor:
        for i, (conv, act_name) in enumerate(zip(self.convs, self._act_names)):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self._act(conv(x, edge_index), act_name)

            if (
                halo_layer_embeddings is not None
                and num_local_nodes is not None
                and i < len(self.convs) - 1
                and i < len(halo_layer_embeddings)
                and halo_layer_embeddings[i] is not None
            ):
                num_halo = x.size(0) - num_local_nodes
                if num_halo > 0 and halo_layer_embeddings[i].size(0) == num_halo:
                    x = x.clone()
                    x[num_local_nodes:] = halo_layer_embeddings[i]

        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

    def get_all_intermediate_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        halo_layer_embeddings=None,
        num_local_nodes=None,
    ):
        pass
        layer_outputs = []
        for i, (conv, act_name) in enumerate(
            zip(self.convs[:-1], self._act_names[:-1])
        ):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self._act(conv(x, edge_index), act_name)
            layer_outputs.append(x)

            if (
                halo_layer_embeddings is not None
                and num_local_nodes is not None
                and i < len(halo_layer_embeddings)
                and halo_layer_embeddings[i] is not None
            ):
                num_halo = x.size(0) - num_local_nodes
                if num_halo > 0 and halo_layer_embeddings[i].size(0) == num_halo:
                    x = x.clone()
                    x[num_local_nodes:] = halo_layer_embeddings[i]
        return layer_outputs


def build_model(
    arch: Architecture,
    num_features: int,
    num_classes: int,
    dropout: float = 0.0,
) -> nn.Module:
    pass
    return GNASModel(num_features, num_classes, arch.arch_description, dropout)
