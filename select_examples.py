"""Select additional examples from candidates produced by the external pipeline.

This repository does not redistribute UMBC, SoftMatcher, embeddings, or model
weights.  The expected JSONL input contains one support per row with a
``query_embedding`` and a ``candidates`` list.  Each candidate has ``id``,
``score``, ``sentence``, and ``embedding`` fields.
"""

from __future__ import annotations

import argparse
import math
import random
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from utils import read_jsonl, write_jsonl


def _central_candidate(
    candidates: list[dict[str, Any]], labels: np.ndarray, centroid: np.ndarray, cluster: int
) -> dict[str, Any]:
    indices = np.flatnonzero(labels == cluster)
    embeddings = np.asarray([candidates[index]["embedding"] for index in indices])
    nearest = indices[np.linalg.norm(embeddings - centroid, axis=1).argmin()]
    return candidates[int(nearest)]


def _threshold_candidates(
    candidates: list[dict[str, Any]], threshold: float, count: int, step: float
) -> tuple[list[dict[str, Any]], float]:
    current = threshold
    selected = [item for item in candidates if float(item["score"]) >= current]
    while len(selected) < count and step > 0:
        current -= step
        selected = [item for item in candidates if float(item["score"]) >= current]
        if current <= min(float(item["score"]) for item in candidates):
            break
    if len(selected) < count:
        raise ValueError(f"Only {len(selected)} candidates available for {count} selections")
    return selected, current


def select(
    record: dict[str, Any],
    strategy: str,
    count: int,
    threshold: float,
    adaptive_step: float,
    seed: int,
    reported_cluster_count: bool,
) -> dict[str, Any]:
    candidates, effective_threshold = _threshold_candidates(
        list(record["candidates"]), threshold, count, adaptive_step
    )
    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    if strategy == "closest":
        chosen = candidates[:count]
        cluster_count = None
    else:
        embeddings = np.asarray([item["embedding"] for item in candidates], dtype=np.float32)
        query = np.asarray(record["query_embedding"], dtype=np.float32).reshape(1, -1)
        root = int(math.sqrt(len(candidates)))
        cluster_count = max(count, root) if reported_cluster_count else root
        if cluster_count < count:
            raise ValueError(
                f"paper cluster count floor(sqrt({len(candidates)}))={cluster_count} "
                f"cannot supply {count} distinct clusters"
            )
        init = "random" if strategy in {"kmeans_random", "kmeans_closest"} else "k-means++"
        kmeans = KMeans(n_clusters=cluster_count, init=init, random_state=seed, n_init=1)
        labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_
        if strategy == "kmeans_random":
            clusters = random.Random(seed).sample(range(cluster_count), count)
        elif strategy in {"kmeans_closest", "kmeanspp_closest"}:
            distances = np.linalg.norm(centroids - query, axis=1)
            clusters = np.argsort(distances)[:count].tolist()
        elif strategy == "kmeanspp_farthest":
            clusters = []
            centers = [query[0]]
            for _ in range(count):
                distances = np.stack(
                    [np.linalg.norm(centroids - center, axis=1) for center in centers]
                ).min(axis=0)
                distances[clusters] = -np.inf
                cluster = int(distances.argmax())
                clusters.append(cluster)
                centers.append(centroids[cluster])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        chosen = [
            _central_candidate(candidates, labels, centroids[cluster], cluster)
            for cluster in clusters
        ]

    return {
        "support_id": record["support_id"],
        "strategy": strategy,
        "requested_threshold": threshold,
        "effective_threshold": effective_threshold,
        "cluster_count": cluster_count,
        "examples": [
            {
                "id": item["id"],
                "score": float(item["score"]),
                "sentence": item["sentence"],
            }
            for item in chosen
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--strategy",
        required=True,
        choices=(
            "closest",
            "kmeans_random",
            "kmeans_closest",
            "kmeanspp_closest",
            "kmeanspp_farthest",
        ),
    )
    parser.add_argument("--count", type=int, choices=(4, 9), required=True)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument(
        "--adaptive-step",
        type=float,
        default=0.0,
        help="Set to 0.05 to reproduce the adaptive FewRel artifact behavior",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--paper-cluster-count",
        action="store_true",
        help="Use floor(sqrt(m)); default reproduces max(count, floor(sqrt(m)))",
    )
    args = parser.parse_args()
    rows = [
        select(
            record,
            strategy=args.strategy,
            count=args.count,
            threshold=args.threshold,
            adaptive_step=args.adaptive_step,
            seed=args.seed,
            reported_cluster_count=not args.paper_cluster_count,
        )
        for record in read_jsonl(args.input)
    ]
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()

