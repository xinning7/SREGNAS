

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml

from sregnas.data.policy import (
    CANONICAL_TWO_LAYER_ARCHITECTURES,
    DATASET_POLICIES,
    dataset_policy,
)


class ConfigError(ValueError):
    pass


@dataclass
class GraphConfig:
    dataset: str = "cora"
    partition_strategy: str = "metis"
    partition_seed: int = 42
    partition_label_scope: str = "train_val"
    hierarchical_r: int = 1
    num_gpus: int = 2
    permutation_seed: int = 42
    fanout_a: int = 5
    fanout_b: int = 5
    f_int: int = 5
    f_ext: int = 0


@dataclass
class SearchConfig:
    token_subset: str | None = None
    generator: str = "random"
    pool_mode: str = "independent"
    q: int = 16
    m: int = 16
    num_layers: int = 2
    base_seed: int = 42
    max_resamples_per_slot: int = 10
    floor_reward: float = 0.0
    entropy_weight: float = 1e-4
    baseline_decay: float = 0.9
    controller_lr: float = 3e-4
    hidden_size: int = 128
    temperature: float = 5.0
    tanh_constant: float = 2.5


@dataclass
class TrainConfig:
    proxy_epochs: int = 25
    retrain_epochs: int = 200
    lr: float = 0.01
    weight_decay: float = 5e-4
    dropout: float = 0.0
    retrain_seed: int | None = None
    retrain_seed_policy: str = "legacy_fingerprint"
    retrain_replica: int = 0


@dataclass
class SelectionConfig:
    plan: str = "plan1"
    k: int = 1


@dataclass
class RuntimeConfig:
    native_retrain_dropout: float | None = None
    comparison_method: str | None = None
    require_fi_gpu: bool = False
    sampler_impl: str | None = None
    feature_residency: str | None = None
    inference_residency: str | None = None
    execution_scope: str = "both"
    side_timer: str = "phase_sum"
    system_gpus: list[int] = field(default_factory=lambda: [4, 5])
    baseline_gpus: list[int] = field(default_factory=lambda: [6, 7])
    parallel_sides: bool = True
    monitor_interval_s: float = 2.0
    foreign_pid_policy: str = "flag"
    execution_mode: str = "persistent_worker"
    shared_search_scheduler: str = "fixed_per_gpu_streams"
    shared_pool_reference: str | None = None
    balance_reference_run: str | None = None
    balance_reference_profile: str | None = None
    legacy_barrier_reference_run: str | None = None
    legacy_barrier_reference_profile: str | None = None
    projection_reference_run: str | None = None
    oversubscribed_reference_run: str | None = None
    oversubscribed_reference_profile: str | None = None
    scheduler_family: str | None = None
    topology_mode: str | None = None
    require_native_topology: bool = False
    require_idle_preflight: bool = False
    sequential_side_order: str | None = None
    baseline_static_gate: bool = False
    baseline_search_graph: str = "full_neighbour_conventional"
    baseline_sampled_graph_reference: str | None = None
    graph_cache_reference: str | None = None
    products_verification_reference: str | None = None


@dataclass
class ExperimentConfig:
    name: str = "m1_cora_n2"
    graph: GraphConfig = field(default_factory=GraphConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    full_budget: dict[str, Any] | None = None
    fixed_x: int | None = None

    def validate(self) -> None:
        g, s, t, sel, rt = (
            self.graph,
            self.search,
            self.train,
            self.selection,
            self.runtime,
        )
        g.dataset = g.dataset.lower()
        if rt.comparison_method not in {
            None,
            "BaselineFair",
            "BaselineFull",
            "FastGNAS",
            "GraphNAP",
            "AP",
            "PAS-PL",
        }:
            raise ConfigError("unknown standalone comparison method")
        if rt.comparison_method is not None and (
            s.generator != "rl" or s.pool_mode != "independent" or rt.require_fi_gpu
        ):
            raise ConfigError(
                "standalone comparisons require independent RL and no FI requirement"
            )
        if rt.execution_scope not in {"both", "system"}:
            raise ConfigError("runtime.execution_scope must be both|system")
        if rt.execution_scope == "system" and (
            s.pool_mode != "independent"
            or rt.parallel_sides
            or rt.shared_search_scheduler != "independent_bound_system_dynamic_baseline"
            or rt.side_timer != "continuous"
        ):
            raise ConfigError(
                "system-only execution requires independent controller-bound "
                "search, parallel_sides=false and side_timer=continuous"
            )
        if rt.side_timer not in {"phase_sum", "continuous"}:
            raise ConfigError("runtime.side_timer must be phase_sum|continuous")
        if rt.side_timer == "continuous" and (
            rt.parallel_sides or rt.sequential_side_order != "system_then_baseline"
        ):
            raise ConfigError(
                "continuous timing requires sequential system_then_baseline"
            )
        if g.dataset not in DATASET_POLICIES:
            raise ConfigError(f"v2 dataset not supported: {g.dataset}")
        policy = dataset_policy(g.dataset)
        if g.partition_strategy not in {"metis", "hierarchical", "random", "fi"}:
            raise ConfigError("partition_strategy must be metis|hierarchical|random|fi")
        if not isinstance(rt.require_fi_gpu, bool):
            raise ConfigError("require_fi_gpu must be boolean")
        if rt.require_fi_gpu and g.partition_strategy != "fi":
            raise ConfigError("require_fi_gpu requires partition_strategy=fi")
        if rt.require_fi_gpu and any(
            getattr(rt, key) != "gpu"
            for key in ("sampler_impl", "feature_residency", "inference_residency")
        ):
            raise ConfigError("require_fi_gpu requires gpu/gpu/gpu")
        if g.partition_label_scope not in {"train", "train_val"}:
            raise ConfigError("partition_label_scope must be train|train_val")
        if not isinstance(g.partition_seed, int) or not 0 <= g.partition_seed < 2**32:
            raise ConfigError("partition_seed must be an integer in [0,2**32)")
        for key, choices in {
            "sampler_impl": {"python", "gpu"},
            "feature_residency": {"host", "gpu"},
            "inference_residency": {"host", "gpu"},
        }.items():
            if getattr(rt, key) is not None and getattr(rt, key) not in choices:
                raise ConfigError(f"invalid runtime.{key}: {getattr(rt, key)}")
        if g.hierarchical_r not in policy.permitted_r:
            allowed = ",".join(str(value) for value in sorted(policy.permitted_r))
            raise ConfigError(
                f"dataset {g.dataset} permits hierarchical_r in {{{allowed}}}"
            )
        if (g.hierarchical_r > 1) != (g.partition_strategy == "hierarchical"):
            raise ConfigError(
                "partition contract is R>1 for hierarchical and R=1 otherwise"
            )
        if g.permutation_seed != 42 or (g.f_int, g.f_ext) != (5, 0):
            raise ConfigError("authoritative sampling is seed=42 locality(5,0)")
        if g.fanout_a != 5 or g.fanout_b != 5:
            raise ConfigError("M1/M2 authoritative per-layer caps are a=b=5")
        if s.generator not in {"random", "rl"}:
            raise ConfigError("generator must be random|rl")
        if s.pool_mode not in {"independent", "shared_random"}:
            raise ConfigError("search.pool_mode must be independent|shared_random")
        if s.pool_mode == "shared_random" and s.generator != "random":
            raise ConfigError(
                "shared_random is a KPI-2 hygiene mode for the random generator"
            )
        if s.num_layers != 2:
            raise ConfigError("canonical v2 search is exactly two layers")
        if s.token_subset not in {None, "icde27_token_intersection_v1"}:
            raise ConfigError("unknown opt-in search.token_subset")
        if not (0 < s.q <= CANONICAL_TWO_LAYER_ARCHITECTURES):
            raise ConfigError(
                "Q must be within the 10,863,616-architecture canonical space"
            )
        if not (10 <= s.m <= 20):
            raise ConfigError("M must be within [10,20]")
        if s.m % g.num_gpus or s.q % s.m:
            raise ConfigError("M must divide Q and be divisible by N")
        if s.max_resamples_per_slot != 10:
            raise ConfigError("amendment B locks max resamples/slot to 10")
        if t.proxy_epochs not in {25, 100}:
            raise ConfigError("proxy_epochs must be 25 (fast) or 100")
        expected_retrain_epochs = (
            25 if g.dataset in {"ogbn-products", "reddit"} else 200
        )
        if (
            t.retrain_epochs != expected_retrain_epochs
            or t.lr != 0.01
            or t.weight_decay != 5e-4
        ):
            protocol = (
                "large-graph mini-batch protocol is Adam .01/5e-4 for 25 passes"
                if g.dataset in {"ogbn-products", "reddit"}
                else "full-batch protocol is Adam .01/5e-4 for 200 epochs"
            )
            raise ConfigError(protocol)
        if t.retrain_seed is not None and not (0 <= int(t.retrain_seed) < 2**31):
            raise ConfigError("train.retrain_seed must be in [0,2**31)")
        if g.dataset == "ogbn-products" and t.retrain_seed is not None:
            raise ConfigError(
                "products retrain seed is architecture-derived; explicit side/"
                "controller seeds are forbidden"
            )
        if t.retrain_seed_policy not in {
            "legacy_fingerprint",
            "fingerprint_retrain_replica",
        }:
            raise ConfigError(
                "train.retrain_seed_policy must be legacy_fingerprint|"
                "fingerprint_retrain_replica"
            )
        if not (0 <= int(t.retrain_replica) < 2**31):
            raise ConfigError("train.retrain_replica must be in [0,2**31)")
        if t.retrain_seed is not None and t.retrain_seed_policy != "legacy_fingerprint":
            raise ConfigError(
                "explicit train.retrain_seed cannot be combined with a "
                "derived retrain_seed_policy"
            )
        if sel.plan not in {"plan1", "plan2"}:
            raise ConfigError("selection plan must be plan1|plan2")
        if self.fixed_x is not None and (
            int(self.fixed_x) <= 0 or g.num_gpus * sel.k != int(self.fixed_x)
        ):
            raise ConfigError("fixed_x must equal graph.num_gpus * selection.k")
        if self.full_budget is not None:
            reddit_four = (
                self.full_budget.get("protocol") == "reddit_comparison_4gpu_v1"
            )
            reddit_staged = (
                self.full_budget.get("protocol") == "reddit_comparison_staged_v1"
            )
            reddit = (
                self.full_budget.get("protocol") == "reddit_distnas_2gpu_v1"
                or reddit_four
                or reddit_staged
            )
            stage0 = self.full_budget.get("protocol") == "five_dataset_stage0_v1"
            cost_aligned = (
                self.full_budget.get("protocol") == "arxiv_cost_aligned_sampled_v1"
            )
            products_cost_aligned = (
                self.full_budget.get("protocol") == "products_cost_aligned_sampled_v1"
            )
            stage1 = self.full_budget.get("protocol") == "five_dataset_stage1_v1"
            expected_points = (
                {"N8": {"m": 16, "t": 2}}
                if stage0
                else (
                    {
                        "N2": {"m": 16, "t": 25},
                        "N4": {"m": 16, "t": 25},
                        "N8": {"m": 16, "t": 25},
                    }
                    if cost_aligned or stage1 or reddit
                    else (
                        {"N8": {"m": 16, "t": 25}}
                        if products_cost_aligned
                        else {
                            "N2": {"m": 20, "t": 20},
                            "N4": {"m": 20, "t": 20},
                            "N8": {"m": 16, "t": 25},
                        }
                    )
                )
            )
            declared_points = {
                key: value
                for key, value in self.full_budget.items()
                if key not in {"q", "protocol"}
            }
            if (
                self.full_budget.get("q") != (32 if stage0 else 400)
                or not declared_points
                or (
                    reddit
                    and (
                        g.dataset != "reddit"
                        or s.generator != "rl"
                        or s.pool_mode != "independent"
                        or self.fixed_x != 8
                        or rt.system_gpus
                        != (
                            [4, 5]
                            if reddit_staged and g.num_gpus == 2
                            else (
                                [4, 5, 6, 7] if reddit_four or reddit_staged else [5, 6]
                            )
                        )
                        or rt.baseline_gpus != rt.system_gpus
                        or t.proxy_epochs != 100
                    )
                )
                or (
                    stage0
                    and (
                        g.dataset != "citeseer"
                        or g.num_gpus != 8
                        or s.generator != "rl"
                    )
                )
                or (cost_aligned and g.dataset != "ogbn-arxiv")
                or (
                    products_cost_aligned
                    and (
                        g.dataset != "ogbn-products"
                        or g.num_gpus != 8
                        or s.generator != "rl"
                        or s.pool_mode != "independent"
                    )
                )
                or (
                    stage1
                    and (
                        g.dataset
                        not in {
                            "cora",
                            "citeseer",
                            "pubmed",
                            "ogbn-arxiv",
                            "flickr",
                        }
                        or s.generator != "rl"
                        or s.pool_mode != "independent"
                    )
                )
                or (
                    not cost_aligned
                    and not products_cost_aligned
                    and not stage1
                    and not stage0
                    and not reddit
                    and "protocol" in self.full_budget
                )
                or any(
                    key not in expected_points or value != expected_points[key]
                    for key, value in declared_points.items()
                )
                or self.full_budget.get(f"N{g.num_gpus}")
                != expected_points.get(f"N{g.num_gpus}")
            ):
                raise ConfigError(
                    "campaign full_budget must declare the current registered "
                    "Q400 point, the exact arxiv cost-aligned sampled "
                    "N2/N4/N8(M16,T25) protocol, the exact five-dataset "
                    "Stage-1 Q400/N2,N4,N8(M16,T25) protocol, or the exact five-dataset "
                    "CiteSeer Stage-0 Q32/M16/T2 protocol"
                )
            point = self.full_budget.get(f"N{g.num_gpus}")
            if (
                point is None
                or s.q != self.full_budget["q"]
                or s.m != point["m"]
                or self.rounds != point["t"]
            ):
                raise ConfigError(
                    "profile q/m/t must exactly match its declared " "full_budget point"
                )
        if not rt.system_gpus or not rt.baseline_gpus:
            raise ConfigError("system/baseline physical GPU lists cannot be empty")
        if rt.monitor_interval_s != 2.0:
            raise ConfigError("campaign GPU monitoring interval is fixed at 2 seconds")
        if rt.foreign_pid_policy != "flag":
            raise ConfigError("foreign_pid_policy must be flag")
        if rt.scheduler_family not in {None, "barrier_free_symmetric"}:
            raise ConfigError("runtime.scheduler_family must be barrier_free_symmetric")
        if (
            rt.scheduler_family == "barrier_free_symmetric"
            and rt.shared_search_scheduler
            not in {
                "cost_balanced_system_dynamic_baseline",
                "independent_bound_system_dynamic_baseline",
            }
        ):
            raise ConfigError(
                "barrier_free_symmetric requires either cost-balanced "
                "shared-pool system scheduling or independent controller-"
                "bound system scheduling, with a dynamic baseline"
            )
        if rt.require_native_topology and (
            len(rt.system_gpus) != g.num_gpus or len(rt.baseline_gpus) != g.num_gpus
        ):
            raise ConfigError(
                "native topology requires one physical GPU per logical shard "
                "on both sides"
            )
        if rt.topology_mode not in {
            None,
            "native",
            "oversubscribed_2x",
            "oversubscribed_4x",
            "replicated_2x",
        }:
            raise ConfigError(
                "runtime.topology_mode must be native|oversubscribed_2x|oversubscribed_4x|replicated_2x"
            )
        if rt.topology_mode == "native" and not rt.require_native_topology:
            raise ConfigError(
                "topology_mode=native requires require_native_topology=true"
            )
        if rt.topology_mode == "replicated_2x" and (
            g.dataset != "reddit"
            or rt.require_native_topology
            or rt.parallel_sides
            or len(rt.system_gpus) != 2 * g.num_gpus
            or rt.system_gpus != rt.baseline_gpus
            or rt.shared_search_scheduler != "independent_bound_system_dynamic_baseline"
        ):
            raise ConfigError(
                "replicated_2x requires two physical workers per logical controller"
            )
        if rt.topology_mode == "oversubscribed_2x" and (
            rt.require_native_topology
            or g.num_gpus != 2 * len(rt.system_gpus)
            or g.num_gpus != 2 * len(rt.baseline_gpus)
        ):
            raise ConfigError(
                "topology_mode=oversubscribed_2x requires exactly two logical "
                "shards per physical GPU on both sides"
            )
        if rt.topology_mode == "oversubscribed_4x" and (
            rt.require_native_topology
            or g.num_gpus != 4 * len(rt.system_gpus)
            or g.num_gpus != 4 * len(rt.baseline_gpus)
        ):
            raise ConfigError(
                "topology_mode=oversubscribed_4x requires exactly four logical "
                "shards per physical GPU on both sides"
            )
        if rt.sequential_side_order is not None and (
            rt.sequential_side_order != "system_then_baseline" or rt.parallel_sides
        ):
            raise ConfigError(
                "sequential_side_order must be system_then_baseline with "
                "parallel_sides=false"
            )
        if rt.baseline_static_gate and (
            g.dataset != "ogbn-arxiv"
            or s.pool_mode != "independent"
            or rt.shared_search_scheduler != "independent_bound_system_dynamic_baseline"
        ):
            raise ConfigError(
                "baseline_static_gate is restricted to the arxiv "
                "independent-controller dynamic-baseline ablation"
            )
        if rt.baseline_search_graph not in {
            "full_neighbour_conventional",
            "cost_aligned_uniform_5_5",
        }:
            raise ConfigError(
                "runtime.baseline_search_graph must be "
                "full_neighbour_conventional|cost_aligned_uniform_5_5"
            )
        old_cost_aligned_contract = (
            g.dataset == "ogbn-arxiv"
            and s.pool_mode == "independent"
            and s.generator == "rl"
            and rt.baseline_static_gate
            and rt.baseline_sampled_graph_reference
            and self.full_budget is not None
            and self.full_budget.get("protocol") == "arxiv_cost_aligned_sampled_v1"
        )
        stage1_cost_aligned_contract = (
            g.dataset in {"cora", "citeseer", "pubmed", "ogbn-arxiv", "flickr"}
            and s.pool_mode == "independent"
            and s.generator == "rl"
            and rt.baseline_static_gate == (g.dataset == "ogbn-arxiv")
            and rt.baseline_sampled_graph_reference
            and self.full_budget is not None
            and self.full_budget.get("protocol") == "five_dataset_stage1_v1"
        )
        products_cost_aligned_contract = (
            g.dataset == "ogbn-products"
            and g.num_gpus == 8
            and s.pool_mode == "independent"
            and s.generator == "rl"
            and not rt.baseline_static_gate
            and rt.baseline_sampled_graph_reference
            and self.full_budget is not None
            and self.full_budget.get("protocol") == "products_cost_aligned_sampled_v1"
        )
        if (
            rt.baseline_search_graph == "cost_aligned_uniform_5_5"
            and not old_cost_aligned_contract
            and not stage1_cost_aligned_contract
            and not products_cost_aligned_contract
            and not (
                rt.comparison_method == "BaselineFair"
                and g.dataset
                in {"reddit", "cora", "citeseer", "pubmed", "ogbn-arxiv", "flickr"}
                and rt.baseline_sampled_graph_reference
                and not rt.baseline_static_gate
                and rt.shared_search_scheduler
                == "independent_bound_system_dynamic_baseline"
            )
            and not (
                g.dataset == "reddit"
                and self.full_budget
                and self.full_budget.get("protocol")
                in {"reddit_comparison_4gpu_v1", "reddit_comparison_staged_v1"}
                and rt.baseline_sampled_graph_reference
                and not rt.baseline_static_gate
            )
        ):
            raise ConfigError(
                "cost-aligned sampled baseline requires an independent-RL "
                "registered campaign, the dataset-specific arxiv-only gate "
                "policy, an explicit sampled graph reference, and an exact "
                "campaign budget declaration"
            )
        if (
            rt.baseline_search_graph == "full_neighbour_conventional"
            and rt.baseline_sampled_graph_reference is not None
        ):
            raise ConfigError(
                "a conventional baseline cannot declare a sampled graph reference"
            )
        if rt.execution_mode not in {"subprocess_per_cell", "persistent_worker"}:
            raise ConfigError(
                "runtime.execution_mode must be "
                "subprocess_per_cell|persistent_worker"
            )
        if rt.shared_search_scheduler not in {
            "fixed_per_gpu_streams",
            "cost_balanced_system_dynamic_baseline",
            "independent_bound_system_dynamic_baseline",
        }:
            raise ConfigError(
                "runtime.shared_search_scheduler must be "
                "fixed_per_gpu_streams|"
                "cost_balanced_system_dynamic_baseline|"
                "independent_bound_system_dynamic_baseline"
            )
        if rt.shared_search_scheduler == "cost_balanced_system_dynamic_baseline" and (
            s.pool_mode != "shared_random"
            or s.generator != "random"
            or rt.parallel_sides
            or g.num_gpus != len(rt.system_gpus)
            or g.num_gpus != len(rt.baseline_gpus)
            or rt.system_gpus != rt.baseline_gpus
            or not rt.shared_pool_reference
        ):
            raise ConfigError(
                "balanced shared search requires random shared-pool, "
                "sequential equal native GPU lists, and an ordered-pool "
                "reference"
            )
        if (
            rt.shared_search_scheduler == "independent_bound_system_dynamic_baseline"
            and (
                s.pool_mode != "independent"
                or rt.parallel_sides
                or rt.system_gpus != rt.baseline_gpus
                or not (
                    (
                        rt.require_native_topology
                        and g.num_gpus == len(rt.system_gpus)
                        and g.num_gpus == len(rt.baseline_gpus)
                    )
                    or (
                        rt.topology_mode == "replicated_2x"
                        and len(rt.system_gpus) == 2 * g.num_gpus
                        and rt.system_gpus == rt.baseline_gpus
                    )
                    or (
                        rt.topology_mode == "oversubscribed_2x"
                        and g.num_gpus == 2 * len(rt.system_gpus)
                        and g.num_gpus == 2 * len(rt.baseline_gpus)
                    )
                    or (
                        rt.topology_mode == "oversubscribed_4x"
                        and g.num_gpus == 4 * len(rt.system_gpus)
                        and g.num_gpus == 4 * len(rt.baseline_gpus)
                    )
                )
                or rt.sequential_side_order != "system_then_baseline"
            )
        ):
            raise ConfigError(
                "independent controller-bound search requires independent-"
                "pool, sequential equal native, replicated_2x or explicit 2x/4x-oversubscribed "
                "GPU lists, and system_then_baseline ordering"
            )
        if len(rt.system_gpus) != len(rt.baseline_gpus):
            raise ConfigError(
                "physical_gpus_system == physical_gpus_baseline is required"
            )
        if len(set(rt.system_gpus)) != len(rt.system_gpus) or len(
            set(rt.baseline_gpus)
        ) != len(rt.baseline_gpus):
            raise ConfigError("physical GPU lists cannot contain duplicates")
        if rt.topology_mode != "replicated_2x" and (
            len(rt.system_gpus) > g.num_gpus
            or len(rt.baseline_gpus) > g.num_gpus
            or g.num_gpus % len(rt.system_gpus)
            or g.num_gpus % len(rt.baseline_gpus)
        ):
            raise ConfigError(
                "N logical shards must map evenly onto each physical GPU list"
            )
        if rt.parallel_sides and set(rt.system_gpus) & set(rt.baseline_gpus):
            raise ConfigError(
                "parallel system/baseline sides must use disjoint physical GPUs"
            )

    @property
    def allocated_gpus(self) -> list[int]:
        gpus = set(self.runtime.system_gpus)
        if self.runtime.execution_scope == "both":
            gpus.update(self.runtime.baseline_gpus)
        return sorted(gpus)

    @property
    def rounds(self) -> int:
        return self.search.q // self.search.m

    @property
    def launches_per_shard_round(self) -> int:
        return self.search.m // self.graph.num_gpus

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _strict_dataclass(cls: type[T], raw: dict[str, Any], path: str) -> T:
    valid = {f.name for f in fields(cls)}
    unknown = set(raw) - valid
    if unknown:
        raise ConfigError(f"unknown keys at {path}: {sorted(unknown)}")
    hints = get_type_hints(cls)
    values: dict[str, Any] = {}
    for f in fields(cls):
        value = raw.get(f.name)
        typ = hints[f.name]
        if value is not None and hasattr(typ, "__dataclass_fields__"):
            if not isinstance(value, dict):
                raise ConfigError(f"{path}.{f.name} must be a mapping")
            value = _strict_dataclass(typ, value, f"{path}.{f.name}")
        if f.name in raw:
            values[f.name] = value
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping")
    cfg = _strict_dataclass(ExperimentConfig, raw, "config")
    cfg.validate()
    return cfg
