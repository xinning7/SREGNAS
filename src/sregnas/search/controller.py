

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from sregnas.core import Architecture

from sregnas.models.canonical import load_model_module
from sregnas.config import SearchConfig
from sregnas.search.memory_gate import (
    STATIC_GATE_PREDICATE,
    arxiv_static_memory_gate_rejects,
)


@dataclass
class Proposal:
    architecture: Architecture
    trace: tuple[list[torch.Tensor], list[torch.Tensor]] | None


class PerSlotPolicy(nn.Module):
    pass

    def __init__(
        self,
        space: dict[str, list[Any]],
        num_layers: int,
        hidden_size: int,
        temperature: float,
        tanh_constant: float,
    ):
        super().__init__()
        self.space = space
        self.keys = list(space)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.tanh_constant = tanh_constant
        self.start = nn.Parameter(torch.zeros(1, hidden_size))
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.heads = nn.ModuleDict(
            {key: nn.Linear(hidden_size, len(values)) for key, values in space.items()}
        )
        self.embeddings = nn.ModuleDict(
            {
                key: nn.Embedding(len(values), hidden_size)
                for key, values in space.items()
            }
        )

    def sample(
        self, generator: torch.Generator
    ) -> tuple[list[dict[str, Any]], list[torch.Tensor], list[torch.Tensor]]:
        device = self.start.device
        h = torch.zeros(1, self.hidden_size, device=device)
        c = torch.zeros(1, self.hidden_size, device=device)
        inp = self.start
        layers: list[dict[str, Any]] = []
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for _ in range(self.num_layers):
            layer: dict[str, Any] = {}
            for key in self.keys:
                h, c = self.lstm(inp, (h, c))
                logits = self.heads[key](h).squeeze(0)
                if self.temperature > 0:
                    logits = logits / self.temperature
                    logits = self.tanh_constant * torch.tanh(logits)
                probs = torch.softmax(logits, dim=0)
                index = torch.multinomial(probs, 1, generator=generator).squeeze(0)
                log_probs.append(torch.log(probs[index].clamp_min(1e-12)))
                entropies.append(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
                layer[key] = self.space[key][int(index)]
                inp = self.embeddings[key](index).unsqueeze(0)
            layers.append(layer)
        return layers, log_probs, entropies


class ShardController:
    def __init__(self, config: SearchConfig, seed: int):
        self.config = config
        self.seed = seed
        self.generator_name = config.generator
        self.model_module = load_model_module()
        self.space = self.model_module.SEARCH_SPACE
        if config.token_subset is not None:
            from sregnas.icde27.space import FIELDS, NAME
            if config.token_subset != NAME:
                raise ValueError("unknown token subset")
            self.space = {key: list(values) for key, values in FIELDS.items()}
        self.seen: set[str] = set()
        self.baseline = 0.0
        self._grad_observations = 0
        self.random_rng = random.Random(seed)
        self.torch_rng = torch.Generator(device="cpu")
        self.torch_rng.manual_seed(seed)
        self.policy: PerSlotPolicy | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        if config.generator == "rl":

            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.policy = PerSlotPolicy(
                    self.space,
                    config.num_layers,
                    config.hidden_size,
                    config.temperature,
                    config.tanh_constant,
                )
            self.optimizer = torch.optim.Adam(
                self.policy.parameters(), lr=config.controller_lr
            )
            self.optimizer.zero_grad(set_to_none=True)

    def propose(self) -> Proposal:
        if self.generator_name == "random":
            layers = [
                self.model_module._canonicalize_layer(
                    {
                        key: self.random_rng.choice(values)
                        for key, values in self.space.items()
                    }
                )
                for _ in range(self.config.num_layers)
            ]
            return Proposal(Architecture(layers), None)
        assert self.policy is not None
        raw, log_probs, entropies = self.policy.sample(self.torch_rng)
        canonical = [self.model_module._canonicalize_layer(layer) for layer in raw]
        return Proposal(Architecture(canonical), (log_probs, entropies))

    def classify(self, proposal: Proposal, *, static_gate: bool) -> str:
        fp = proposal.architecture.fingerprint()
        if fp in self.seen:
            return "DEDUP_REJECT"
        self.seen.add(fp)
        if static_gate and arxiv_static_memory_gate_rejects(
            proposal.architecture.arch_description
        ):
            return "STATIC_REJECT"
        return "ACCEPT"

    def observe(
        self,
        observations: list[tuple[Proposal, float]],
        *,
        launches_per_shard_round: int,
        flush: bool = False,
    ) -> dict[str, Any]:
        if self.generator_name != "rl" or not observations:
            return {"updated": False, "observations": len(observations)}
        assert self.policy is not None and self.optimizer is not None
        raw = np.asarray([reward for _, reward in observations], dtype=np.float64)
        proposal_entropies = [
            float(torch.stack(proposal.trace[1]).sum().detach())
            for proposal, _reward in observations
            if proposal.trace is not None and proposal.trace[1]
        ]
        max_entropy = float(
            self.config.num_layers
            * sum(np.log(len(values)) for values in self.space.values())
        )
        if len(raw) < 2 or float(raw.std()) == 0.0:
            normalized = np.zeros_like(raw)
        else:
            normalized = (raw - raw.mean()) / raw.std()
        losses = []
        for (proposal, _), reward in zip(observations, normalized.tolist()):
            if proposal.trace is None:
                continue
            log_probs, entropies = proposal.trace
            self.baseline = (
                self.config.baseline_decay * self.baseline
                + (1.0 - self.config.baseline_decay) * reward
            )
            advantage = reward - self.baseline
            lp = torch.stack(log_probs).sum()
            entropy = torch.stack(entropies).sum()
            losses.append(-(lp * advantage + self.config.entropy_weight * entropy))
        if not losses:
            return {"updated": False, "observations": len(observations)}
        torch.stack(losses).mean().backward()


        self._grad_observations += launches_per_shard_round
        threshold = 4 if launches_per_shard_round < 4 else 1
        updated = self._grad_observations >= threshold or flush
        if updated:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._grad_observations = 0
        return {
            "updated": updated,
            "observations": len(observations),
            "zscore_mean": float(normalized.mean()),
            "zscore_std": float(normalized.std()),
            "baseline": self.baseline,
            "sample_policy_entropy_mean": (
                float(np.mean(proposal_entropies)) if proposal_entropies else None
            ),
            "sample_policy_entropy_min": (
                float(np.min(proposal_entropies)) if proposal_entropies else None
            ),
            "sample_policy_entropy_max": (
                float(np.max(proposal_entropies)) if proposal_entropies else None
            ),
            "sample_policy_entropy_normalized_mean": (
                float(np.mean(proposal_entropies) / max_entropy)
                if proposal_entropies and max_entropy > 0
                else None
            ),
            "sample_policy_entropy_uniform_max": max_entropy,
        }

    def save(self, path: Path) -> None:
        payload: dict[str, Any] = {
            "generator_name": self.generator_name,
            "seed": self.seed,
            "seen": sorted(self.seen),
            "baseline": self.baseline,
            "random_state": self.random_rng.getstate(),
            "torch_rng_state": self.torch_rng.get_state(),
            "grad_observations": self._grad_observations,
        }
        if self.policy is not None and self.optimizer is not None:
            payload.update(
                {
                    "policy": self.policy.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "grads": [
                        None if p.grad is None else p.grad.detach().clone()
                        for p in self.policy.parameters()
                    ],
                }
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.seen = set(payload["seen"])
        self.baseline = float(payload["baseline"])
        self.random_rng.setstate(payload["random_state"])
        self.torch_rng.set_state(payload["torch_rng_state"])
        self._grad_observations = int(payload["grad_observations"])
        if self.policy is not None and self.optimizer is not None:
            self.policy.load_state_dict(payload["policy"])
            self.optimizer.load_state_dict(payload["optimizer"])
            for param, grad in zip(self.policy.parameters(), payload["grads"]):
                param.grad = grad
