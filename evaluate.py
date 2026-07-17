"""Evaluate predictions and run the paper's paired bootstrap comparison."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

import numpy as np

from config import NO_RELATION
from utils import read_jsonl


def metrics(gold: list[str], predicted: list[str]) -> dict[str, float]:
    tp = fp = fn = 0
    for truth, guess in zip(gold, predicted):
        if guess == truth and truth != NO_RELATION:
            tp += 1
        else:
            if guess != NO_RELATION:
                fp += 1
            if truth != NO_RELATION:
                fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_set: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        per_set[int(row["set_id"])].append(row)
    set_metrics = []
    for set_id in sorted(per_set):
        group = per_set[set_id]
        result = metrics(
            [row["gold_relation"] for row in group],
            [row["prediction"] for row in group],
        )
        result["set_id"] = set_id
        set_metrics.append(result)
    summary = {}
    for name in ("precision", "recall", "f1"):
        values = np.asarray([row[name] for row in set_metrics])
        summary[name] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    return {"sets": set_metrics, "summary": summary}


def paired_bootstrap(
    baseline: list[dict[str, Any]],
    system: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> dict[str, float]:
    key = lambda row: (
        row["set_id"], row["episode_id"], row["query_index"], row["query_id"]
    )
    base = {key(row): row for row in baseline}
    other = {key(row): row for row in system}
    if base.keys() != other.keys():
        raise ValueError("Prediction files do not contain the same query keys")
    keys = sorted(base)
    gold = np.asarray([base[item]["gold_relation"] for item in keys], dtype=object)
    base_pred = np.asarray([base[item]["prediction"] for item in keys], dtype=object)
    system_pred = np.asarray([other[item]["prediction"] for item in keys], dtype=object)
    rng = np.random.default_rng(seed)
    non_positive = 0
    deltas = []

    def f1_array(truth: np.ndarray, guess: np.ndarray) -> float:
        correct_positive = (truth == guess) & (truth != NO_RELATION)
        wrong = truth != guess
        tp = int(correct_positive.sum())
        fp = int((wrong & (guess != NO_RELATION)).sum())
        fn = int((wrong & (truth != NO_RELATION)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    for _ in range(samples):
        indices = rng.integers(0, len(keys), len(keys))
        base_f1 = f1_array(gold[indices], base_pred[indices])
        system_f1 = f1_array(gold[indices], system_pred[indices])
        delta = system_f1 - base_f1
        deltas.append(delta)
        non_positive += delta <= 0
    return {
        "observed_delta": f1_array(gold, system_pred) - f1_array(gold, base_pred),
        "bootstrap_mean_delta": float(np.mean(deltas)),
        "one_sided_p": (non_positive + 1) / (samples + 1),
        "samples": samples,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions")
    parser.add_argument("--baseline")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = list(read_jsonl(args.predictions))
    result = summarize(rows)
    if args.baseline:
        result["paired_bootstrap"] = paired_bootstrap(
            list(read_jsonl(args.baseline)), rows, args.bootstrap_samples, args.seed
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
