from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def scaffold_group_split(groups: Iterable[str], seed: int = 0, ratios=(0.8, 0.1, 0.1)):
    """Assign group keys to train/val/test without group leakage."""
    group_to_indices = defaultdict(list)
    for i, group in enumerate(groups):
        group_to_indices[str(group)].append(i)

    rng = np.random.default_rng(seed)
    items = list(group_to_indices.items())
    rng.shuffle(items)
    items.sort(key=lambda kv: len(kv[1]), reverse=True)

    names = ["train", "val", "test"]
    total = sum(len(v) for _, v in items)
    targets = {name: ratios[i] * total for i, name in enumerate(names)}
    counts = {name: 0 for name in names}
    assignment = {}

    for group, idxs in items:
        deficits = {name: targets[name] - counts[name] for name in names}
        split = max(names, key=lambda name: deficits[name])
        assignment[group] = split
        counts[split] += len(idxs)

    return assignment
