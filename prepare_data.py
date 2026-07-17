"""Inspect, validate, and map the released FewRel/TACRED identifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config import EPISODE_SEEDS, NO_RELATION
from utils import index_jsonl, read_json, read_jsonl, write_jsonl


ROOT = Path(__file__).resolve().parent


def _fewrel_paths(root: Path) -> dict[str, Path]:
    base = root / "datasets" / "fewrel"
    return {
        "source": base / "source_examples.jsonl.gz",
        "supports": base / "support_examples.jsonl.gz",
        "queries": base / "query_examples.jsonl.gz",
        "rules": base / "semantic_rules.jsonl.gz",
        "relations": base / "relations.json",
        "episodes": base / "episodes",
    }


def validate_release(root: Path) -> dict[str, Any]:
    paths = _fewrel_paths(root)
    source = index_jsonl(paths["source"])
    supports = index_jsonl(paths["supports"])
    queries = index_jsonl(paths["queries"])
    rules = index_jsonl(paths["rules"], key="source_example_id")
    relation_document = read_json(paths["relations"])
    selected_relations = {
        entry["name"] for entry in relation_document["relations"] if entry["selected_for_fsre"]
    }

    if len(source) != 70000:
        raise ValueError(f"Expected 70,000 source examples, found {len(source)}")
    if len(supports) != 4200 or len(rules) != 4200:
        raise ValueError(
            f"Expected 4,200 supports/rules, found {len(supports)}/{len(rules)}"
        )
    if len(queries) != 68771:
        raise ValueError(f"Expected 68,771 unique queries, found {len(queries)}")
    if supports.keys() != rules.keys():
        raise ValueError("FewRel support and semantic-rule IDs do not match")
    if not supports.keys() <= source.keys():
        raise ValueError("A support ID does not map to the released source data")
    if {row["relation"] for row in supports.values()} != selected_relations:
        raise ValueError("Support relation set does not match relations.json")

    total_rows = no_relation = 0
    for set_id, seed in enumerate(EPISODE_SEEDS):
        episode_path = paths["episodes"] / f"seed_{seed}.jsonl.gz"
        rows = list(read_jsonl(episode_path))
        if len(rows) != 30000:
            raise ValueError(f"Expected 30,000 rows in {episode_path}, found {len(rows)}")
        for row in rows:
            if row["set_id"] != set_id or row["seed"] != seed:
                raise ValueError(f"Incorrect set metadata in {episode_path}")
            if len(row["support_ids"]) != 5 or len(set(row["support_ids"])) != 5:
                raise ValueError(f"Invalid support set in episode {row['episode_id']}")
            if not set(row["support_ids"]) <= supports.keys():
                raise ValueError(f"Unknown support ID in episode {row['episode_id']}")
            if row["query_id"] not in queries:
                raise ValueError(f"Unknown query ID {row['query_id']}")
            expected_relations = [supports[item]["relation"] for item in row["support_ids"]]
            if row["candidate_relations"] != expected_relations:
                raise ValueError(f"Candidate relation mismatch in episode {row['episode_id']}")
            query_relation = queries[row["query_id"]]["relation"]
            expected_gold = (
                query_relation if query_relation in expected_relations else NO_RELATION
            )
            if row["gold_relation"] != expected_gold:
                raise ValueError(f"Gold relation mismatch for {row['query_id']}")
            no_relation += row["gold_relation"] == NO_RELATION
        total_rows += len(rows)

    return {
        "source_examples": len(source),
        "support_examples": len(supports),
        "query_examples": len(queries),
        "semantic_rules": len(rules),
        "episode_query_rows": total_rows,
        "episode_sets": len(EPISODE_SEEDS),
        "no_relation": no_relation,
        "no_relation_fraction": no_relation / total_rows,
    }


def show_fewrel_mapping(root: Path, rule_id: str) -> dict[str, Any]:
    paths = _fewrel_paths(root)
    source = index_jsonl(paths["source"])
    rules = index_jsonl(paths["rules"], key="source_example_id")
    if rule_id not in rules:
        raise KeyError(f"No FewRel rule with ID {rule_id!r}")
    return {"rule": rules[rule_id], "source_example": source[rule_id]}


def _authorized_examples(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix in {".jsonl", ".gz"}:
        return index_jsonl(path)
    value = read_json(path)
    if isinstance(value, list):
        return {row["id"]: row for row in value}
    if isinstance(value, dict):
        return value
    raise ValueError(f"Unsupported TACRED container in {path}")


def join_tacred(rules_path: Path, authorized_path: Path, output_path: Path) -> int:
    """Join public rules to a user's authorized TACRED records by source ID."""

    rules = list(read_jsonl(rules_path))
    examples = _authorized_examples(authorized_path)
    missing = [row["source_example_id"] for row in rules if row["source_example_id"] not in examples]
    if missing:
        raise KeyError(f"Authorized data is missing {len(missing)} source IDs; first={missing[0]}")
    write_jsonl(
        output_path,
        (
            {"rule": row, "source_example": examples[row["source_example_id"]]}
            for row in rules
        ),
    )
    return len(rules)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    mapping = subparsers.add_parser("map-fewrel-rule")
    mapping.add_argument("rule_id")
    tacred = subparsers.add_parser("join-tacred")
    tacred.add_argument("--rules", type=Path, default=ROOT / "datasets/tacred/semantic_rules.jsonl.gz")
    tacred.add_argument("--authorized-examples", type=Path, required=True)
    tacred.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "validate":
        result = validate_release(args.root)
    elif args.command == "map-fewrel-rule":
        result = show_fewrel_mapping(args.root, args.rule_id)
    else:
        result = {"joined_rules": join_tacred(args.rules, args.authorized_examples, args.output)}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
