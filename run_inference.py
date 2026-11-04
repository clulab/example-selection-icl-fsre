"""Run the paper's binary or multiclass in-context relation classifier."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from config import NO_RELATION
from entity_filter import filter_prompts as build_filter_prompts
from entity_filter import load_entity_filter_map
from model import PaperLLM, VLLMPaperLLM
from prompts import binary_relation_prompt, multiclass_prompt, numbered_supports, render
from utils import entity_text, index_jsonl, read_json, read_jsonl, tagged_sentence, write_jsonl


def relation_map(path: str | Path) -> dict[str, dict[str, str]]:
    value = read_json(path)
    entries = value.get("relations", value)
    if isinstance(entries, list):
        return {entry["name"]: entry for entry in entries}
    return entries


def extra_example_map(
    path: str | None,
    method: str | None = None,
    additional_count: int | None = None,
) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    mapped: dict[str, dict[str, Any]] = {}
    available: set[tuple[str, int]] = set()
    uses_unified_schema = False
    for row in read_jsonl(path):
        row_method = row.get("method")
        row_count = row.get("additional_count")
        if (row_method is None) != (row_count is None):
            raise ValueError(
                "An extra-example row must provide both method and additional_count"
            )
        if row_method is not None:
            uses_unified_schema = True
            selector = (str(row_method), int(row_count))
            available.add(selector)
            if method is None or additional_count is None or selector != (
                method,
                additional_count,
            ):
                continue

        examples = row["examples"]
        support_id = row.get("support_id", row.get("id"))
        if support_id is None:
            raise KeyError("An extra-example row must contain support_id or id")
        if support_id in mapped:
            raise ValueError(
                "Duplicate extra-example row for "
                f"support {support_id!r}, method {method!r}, count {additional_count!r}"
            )
        sentences = [
            item["sentence"] if isinstance(item, dict) else item for item in examples
        ]
        expected = int(row_count) if row_count is not None else additional_count
        if expected is not None and len(sentences) != expected:
            raise ValueError(
                f"Support {support_id!r} has {len(sentences)} additional examples; "
                f"expected {expected}"
            )
        summarized = bool(row.get("summarized", False))
        source_sentence = row.get("source_sentence")
        if summarized and not source_sentence:
            raise ValueError(
                f"Summarized row for support {support_id!r} has no source_sentence"
            )
        mapped[str(support_id)] = {
            "examples": sentences,
            "source_sentence": source_sentence,
            "summarized": summarized,
        }

    if uses_unified_schema and (method is None or additional_count is None):
        choices = ", ".join(
            f"{name}:{count}" for name, count in sorted(available)
        )
        raise ValueError(
            "This unified extra-example file requires --extra-method and "
            f"--additional-count. Available variants: {choices}"
        )
    if uses_unified_schema and not mapped:
        choices = ", ".join(
            f"{name}:{count}" for name, count in sorted(available)
        )
        raise ValueError(
            f"No rows found for method={method!r}, additional_count={additional_count!r}. "
            f"Available variants: {choices}"
        )
    return mapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--supports", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--relations", required=True)
    parser.add_argument("--extra-examples")
    parser.add_argument(
        "--extra-method",
        help="Method to select from a unified additional-examples JSONL file.",
    )
    parser.add_argument(
        "--additional-count",
        type=int,
        choices=(4, 9),
        help="Use four (5-shot) or nine (10-shot) additional examples.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--entity-filter",
        action="store_true",
        help="Recompute the paper's subject/object entity filter with the model.",
    )
    parser.add_argument(
        "--entity-filter-map",
        help=(
            "Trusted pickle containing saved query/relation filter decisions. "
            "This implies entity filtering and skips filter recomputation."
        ),
    )
    parser.add_argument("--mode", choices=("binary", "multiclass"), default="binary")
    parser.add_argument("--parametric", action="store_true")
    args = parser.parse_args()
    if args.backend == "vllm" and args.mode != "binary":
        parser.error("The vLLM backend supports --mode binary only")
    if (args.entity_filter or args.entity_filter_map) and args.mode != "binary":
        parser.error("Entity filtering supports --mode binary only")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be positive")
    if bool(args.extra_method) != (args.additional_count is not None):
        parser.error("--extra-method and --additional-count must be used together")
    if (args.extra_method or args.additional_count is not None) and not args.extra_examples:
        parser.error("--extra-method/--additional-count require --extra-examples")

    supports = index_jsonl(args.supports)
    queries = index_jsonl(args.queries)
    relations = relation_map(args.relations)
    extras = extra_example_map(
        args.extra_examples,
        method=args.extra_method,
        additional_count=args.additional_count,
    )
    episode_rows = list(read_jsonl(args.episodes))
    contexts: list[dict[str, Any]] = []
    for episode in episode_rows:
        query = queries[episode["query_id"]]
        support_records = [supports[item] for item in episode["support_ids"]]
        by_relation: dict[str, list[str]] = defaultdict(list)
        summarized_values: set[bool] = set()
        for support in support_records:
            relation = support["relation"]
            variant = extras.get(str(support["id"]))
            if args.extra_examples and variant is None:
                raise KeyError(
                    f"No selected additional examples for support {support['id']!r}"
                )
            if variant is None:
                by_relation[relation].append(tagged_sentence(support))
            else:
                by_relation[relation].append(
                    variant["source_sentence"] or tagged_sentence(support)
                )
                by_relation[relation].extend(variant["examples"])
                summarized_values.add(bool(variant["summarized"]))
        if len(summarized_values) > 1:
            raise ValueError("An episode mixes summarized and original support variants")
        contexts.append(
            {
                "episode": episode,
                "query": query,
                "by_relation": dict(by_relation),
                "candidate_relations": list(by_relation),
                "summarized": next(iter(summarized_values), False),
            }
        )

    if args.backend == "vllm":
        llm = VLLMPaperLLM(
            args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            seed=args.seed,
        )
    else:
        llm = PaperLLM(args.model, device_map=args.device_map)

    output: list[dict[str, Any]] = []

    if args.mode == "multiclass":
        for context in contexts:
            episode = context["episode"]
            query = context["query"]
            by_relation = context["by_relation"]
            candidate_relations = context["candidate_relations"]
            relation_entries = [relations[name] for name in candidate_relations]
            prompt = multiclass_prompt(
                relation_entries, by_relation, tagged_sentence(query)
            )
            response = llm.generate(
                [prompt], batch_size=1, max_new_tokens=20, seed=args.seed, sample=False
            )[0].strip()
            prediction = next(
                (name for name in candidate_relations if response == name), NO_RELATION
            )
            output.append(
                {
                    "set_id": episode["set_id"],
                    "seed": episode["seed"],
                    "episode_id": episode["episode_id"],
                    "query_index": episode["query_index"],
                    "query_id": query["id"],
                    "gold_relation": episode["gold_relation"],
                    "candidate_relations": candidate_relations,
                    "votes": None,
                    "probabilities": None,
                    "prediction": prediction,
                }
            )
    else:
        allowed_by_context = [
            {name: True for name in context["candidate_relations"]}
            for context in contexts
        ]
        if args.entity_filter_map:
            print(f"Loading entity-filter map from {args.entity_filter_map}", flush=True)
            saved_filter = load_entity_filter_map(args.entity_filter_map)
            for context_index, context in enumerate(contexts):
                query_id = str(context["query"]["id"])
                if query_id not in saved_filter:
                    raise KeyError(
                        f"Entity-filter map has no decisions for query {query_id!r}"
                    )
                for name in context["candidate_relations"]:
                    if name not in saved_filter[query_id]:
                        raise KeyError(
                            "Entity-filter map has no decision for "
                            f"query {query_id!r}, relation {name!r}"
                        )
                    allowed_by_context[context_index][name] = saved_filter[query_id][
                        name
                    ]
        elif args.entity_filter:
            filter_prompt_texts: list[str] = []
            filter_references: list[tuple[str, str]] = []
            unique_pairs: dict[tuple[str, str], tuple[dict[str, Any], str]] = {}
            for context in contexts:
                query_id = str(context["query"]["id"])
                for name in context["candidate_relations"]:
                    unique_pairs.setdefault(
                        (query_id, name), (context["query"], name)
                    )
            for pair, (query, name) in unique_pairs.items():
                pair_prompts = build_filter_prompts(query, relations[name])
                filter_prompt_texts.extend(pair_prompts)
                filter_references.extend([pair] * len(pair_prompts))
            print(
                f"Scoring {len(filter_prompt_texts)} entity-filter prompts "
                f"with {args.backend}",
                flush=True,
            )
            filter_scores = llm.score_yes_no_with_probabilities(
                filter_prompt_texts, batch_size=args.batch_size
            )
            if len(filter_scores) != len(filter_references):
                raise RuntimeError(
                    "The model returned an unexpected number of entity-filter scores"
                )
            decisions_by_pair = {pair: True for pair in unique_pairs}
            for pair, (decision, _) in zip(filter_references, filter_scores):
                if not decision:
                    decisions_by_pair[pair] = False
            for context_index, context in enumerate(contexts):
                query_id = str(context["query"]["id"])
                for name in context["candidate_relations"]:
                    allowed_by_context[context_index][name] = decisions_by_pair[
                        (query_id, name)
                    ]

        prompts: list[str] = []
        prompt_references: list[tuple[int, str]] = []
        for context_index, context in enumerate(contexts):
            query = context["query"]
            by_relation = context["by_relation"]
            for name in context["candidate_relations"]:
                if not allowed_by_context[context_index][name]:
                    continue
                if args.parametric:
                    prompt = render(
                        "parametric_knowledge",
                        relation=name,
                        description=relations[name]["description"],
                        supports=numbered_supports(by_relation[name]),
                        subject=entity_text(query, "subj"),
                        object=entity_text(query, "obj"),
                    )
                else:
                    prompt = binary_relation_prompt(
                        name,
                        relations[name]["description"],
                        by_relation[name],
                        tagged_sentence(query),
                        summarized=context["summarized"],
                    )
                prompts.append(prompt)
                prompt_references.append((context_index, name))

        print(
            f"Scoring {len(prompts)} binary relation prompts with {args.backend}",
            flush=True,
        )
        scores = llm.score_yes_no_with_probabilities(
            prompts, batch_size=args.batch_size
        )
        if len(scores) != len(prompt_references):
            raise RuntimeError("The model returned an unexpected number of scores")

        votes_by_context = [
            {name: False for name in context["candidate_relations"]}
            for context in contexts
        ]
        probabilities_by_context: list[dict[str, dict[str, float] | None]] = [
            {} for _ in contexts
        ]
        for (context_index, name), (decision, probability) in zip(
            prompt_references, scores
        ):
            votes_by_context[context_index][name] = decision
            probabilities_by_context[context_index][name] = (
                None
                if probability is None
                else {"yes": probability[0], "no": probability[1]}
            )

        for context_index, context in enumerate(contexts):
            episode = context["episode"]
            query = context["query"]
            candidate_relations = context["candidate_relations"]
            votes = votes_by_context[context_index]
            positive = [name for name, decision in votes.items() if decision]
            prediction = positive[0] if positive else NO_RELATION
            output.append(
                {
                    "set_id": episode["set_id"],
                    "seed": episode["seed"],
                    "episode_id": episode["episode_id"],
                    "query_index": episode["query_index"],
                    "query_id": query["id"],
                    "gold_relation": episode["gold_relation"],
                    "candidate_relations": candidate_relations,
                    "votes": votes,
                    "probabilities": probabilities_by_context[context_index],
                    "prediction": prediction,
                }
            )

    write_jsonl(args.output, output)


if __name__ == "__main__":
    main()
