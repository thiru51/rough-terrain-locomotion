"""Experiment matrix: policies x conditions.

One harness covers every comparison worth running — the deployable policy against
the baselines, ablations of it against itself, and either of those swept over
terrain type or randomisation level. They are all "run these policies under these
conditions and tabulate", so they share code rather than each getting a script
that drifts out of sync with the others.

Policies and environments are supplied as factories, not instances, so every cell
gets a freshly seeded environment and no state leaks between runs.

This module measures; it does not interpret. It will happily tabulate a policy
that was never trained, and the numbers will be meaningless — which is the point,
since the alternative is a script that quietly refuses to run until everything is
perfect.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from tert.eval.rollout import evaluate


@dataclass
class Condition:
    """A named environment setting: a terrain type, a randomisation level, a seed."""

    name: str
    make_env: Callable[[], object]


@dataclass
class Policy:
    """A named policy, as a factory taking the environment it will run in."""

    name: str
    make_runner: Callable[[object], object]


def run_suite(policies: list[Policy], conditions: list[Condition], num_steps: int, seed: int = 0):
    """Evaluate every policy under every condition. Returns {policy: {condition: summary}}."""
    results: dict[str, dict[str, dict[str, float]]] = {}
    for policy in policies:
        results[policy.name] = {}
        for condition in conditions:
            torch.manual_seed(seed)
            env = condition.make_env()
            metrics = evaluate(env, policy.make_runner(env), num_steps)
            results[policy.name][condition.name] = metrics.summary()
    return results


def format_table(results, metric: str = "episode_return", precision: int = 2) -> str:
    """Markdown table of one metric across the matrix."""
    policies = list(results)
    conditions = list(next(iter(results.values()))) if policies else []

    width = max((len(p) for p in policies), default=6)
    header = f"| {'policy'.ljust(width)} | " + " | ".join(conditions) + " |"
    rule = f"|{'-' * (width + 2)}|" + "|".join("-" * (len(c) + 2) for c in conditions) + "|"

    rows = []
    for policy in policies:
        cells = [f"{results[policy][c][metric]:.{precision}f}".rjust(len(c)) for c in conditions]
        rows.append(f"| {policy.ljust(width)} | " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *rows])


def save_results(results, path: str | Path, metadata: dict | None = None) -> Path:
    """Write raw numbers to JSON. Raw, not summarised — re-tabulating beats re-running."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"metadata": metadata or {}, "results": results}, indent=2), encoding="utf-8"
    )
    return path
