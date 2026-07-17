"""Generate, paraphrase, summarize, or hybrid-select support examples."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from model import PaperLLM
from prompts import render
from utils import (
    parse_integer_list,
    parse_numbered_examples,
    read_json,
    read_jsonl,
    tagged_sentence,
    write_jsonl,
)


def _description_map(path: str | Path) -> dict[str, dict[str, str]]:
    value = read_json(path)
    relations = value.get("relations", value)
    return {entry["name"]: entry for entry in relations} if isinstance(relations, list) else relations


def _support_text(record: dict[str, Any]) -> str:
    return record.get("tagged_sentence") or tagged_sentence(record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("generate", "paraphrase", "summarize", "hybrid"))
    parser.add_argument("--input", required=True, help="Input JSONL(.gz)")
    parser.add_argument("--relations", required=True, help="Relation metadata JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--count", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = list(read_jsonl(args.input))
    relations = _description_map(args.relations)
    llm = PaperLLM(args.model, device_map=args.device_map)

    prompts: list[str] = []
    shuffled_candidates: list[list[dict[str, Any]]] = []
    rng = random.Random(args.seed)
    for record in records:
        relation = record["relation"]
        description = relations[relation]["description"]
        if args.mode in {"generate", "paraphrase"}:
            prompts.append(
                render(
                    args.mode,
                    relation=relation,
                    description=description,
                    support=_support_text(record),
                    count=args.count,
                )
            )
        elif args.mode == "summarize":
            prompts.append(render("summarize", support=_support_text(record)))
        else:
            candidates = list(record["candidates"])
            rng.shuffle(candidates)
            shuffled_candidates.append(candidates)
            candidate_text = "\n".join(
                f"{index}: {item['sentence']}"
                for index, item in enumerate(candidates, start=1)
            )
            prompts.append(
                render(
                    "hybrid_select",
                    relation=relation,
                    description=description,
                    count=args.count,
                    candidates=candidate_text,
                )
            )

    responses = llm.generate(
        prompts,
        batch_size=args.batch_size,
        max_new_tokens=60 if args.mode == "summarize" else 1000,
        seed=args.seed,
        sample=args.mode != "hybrid",
    )

    output = []
    for index, (record, response) in enumerate(zip(records, responses)):
        row = {"id": record["id"], "relation": record["relation"], "mode": args.mode}
        if args.mode in {"generate", "paraphrase"}:
            row["examples"] = parse_numbered_examples(response, args.count)
        elif args.mode == "summarize":
            row["summary"] = response
        else:
            selected = parse_integer_list(response, expected=args.count)
            candidates = shuffled_candidates[index]
            if len(set(selected)) != len(selected) or any(
                item < 1 or item > len(candidates) for item in selected
            ):
                raise ValueError(f"Invalid hybrid selection for {record['id']}: {selected}")
            row["examples"] = [candidates[item - 1] for item in selected]
        output.append(row)
    write_jsonl(args.output, output)


if __name__ == "__main__":
    main()

