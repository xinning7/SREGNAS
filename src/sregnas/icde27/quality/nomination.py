
import random
from sregnas.selection.nomination import (
    plan1_nominate, rank_per_shard, _unique_global_top,
)

RULES = ("Plan1", "Global-Raw", "Quota-Random")


def nominate(rows, shards, k, rule="Plan1"):
    if rule == "Plan1":
        return plan1_nominate(rows, shards, k)
    ranked = rank_per_shard(rows, shards)
    if rule == "Global-Raw":
        return _unique_global_top(rows, x=shards*k, score_key="value"), ranked
    if rule != "Quota-Random":
        raise ValueError(rule)
    from sregnas.training.minibatch import _hash_seed
    selected, seen = [], set()
    for shard in range(shards):

        choices = list(ranked[shard])
        random.Random(_hash_seed(42, "Quota-Random", shard)).shuffle(choices)
        count = 0
        for row in choices:
            if row["fingerprint"] in seen:
                continue
            selected.append(row); seen.add(row["fingerprint"]); count += 1
            if count == k:
                break
        if count != k:
            raise AssertionError(f"shard {shard} cannot provide k={k} unique nominees")

    return selected, ranked
