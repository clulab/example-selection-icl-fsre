"""Run the paper's binary or multiclass in-context relation classifier."""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from config import NO_RELATION
from entity_filter import build_filter_prompt
from model import PaperLLM, VLLMPaperLLM
from prompts import binary_relation_prompt, multiclass_prompt, numbered_supports, render
from utils import entity_text, index_jsonl, read_json, read_jsonl, tagged_sentence, write_jsonl


def relation_map(path: str | Path) -> dict[str, dict[str, str]]:
    value = read_json(path)
    entries = value.get("relations", value)
    if isinstance(entries, list):
        return {entry["name"]: entry for entry in entries}
    return entries


def extra_example_map(path: str | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    mapped = {}
    for row in read_jsonl(path):
        examples = row["examples"]
        support_id = row.get("support_id", row.get("id"))
        if support_id is None:
            raise KeyError("An extra-example row must contain support_id or id")
        mapped[support_id] = [
            item["sentence"] if isinstance(item, dict) else item for item in examples
        ]
    return mapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--supports", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--relations", required=True)
    parser.add_argument("--extra-examples")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--entity-filter", action="store_true")
    parser.add_argument("--mode", choices=("binary", "multiclass"), default="binary")
    parser.add_argument("--parametric", action="store_true")
    args = parser.parse_args()
    if args.backend == "vllm" and args.mode != "binary":
        parser.error("The vLLM backend supports --mode binary only")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be positive")

    supports = index_jsonl(args.supports)
    queries = index_jsonl(args.queries)
    relations = relation_map(args.relations)
    extras = extra_example_map(args.extra_examples)
    episode_rows = list(read_jsonl(args.episodes))
    contexts: list[dict[str, Any]] = []
    for episode in episode_rows:
        query = queries[episode["query_id"]]
        support_records = [supports[item] for item in episode["support_ids"]]
        by_relation: dict[str, list[str]] = defaultdict(list)
        for support in support_records:
            relation = support["relation"]
            by_relation[relation].append(tagged_sentence(support))
            by_relation[relation].extend(extras.get(support["id"], []))
        contexts.append(
            {
                "episode": episode,
                "query": query,
                "by_relation": dict(by_relation),
                "candidate_relations": list(by_relation),
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

    rng = random.Random(args.seed)
    output: list[dict[str, Any]] = []
    episode_keys = [
        (int(episode["set_id"]), int(episode["episode_id"]))
        for episode in episode_rows
    ]
    remaining_queries = Counter(episode_keys)
    total_episodes = len(remaining_queries)
    completed_episodes = 0

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
        if args.entity_filter:
            filter_prompts: list[str] = []
            filter_references: list[tuple[int, str]] = []
            for context_index, context in enumerate(contexts):
                for name in context["candidate_relations"]:
                    filter_prompts.append(
                        build_filter_prompt(
                            context["query"],
                            name,
                            relations[name],
                        )
                    )
                    filter_references.append((context_index, name))
            print(
                f"Scoring {len(filter_prompts)} entity-filter prompts with {args.backend}",
                flush=True,
            )
            filter_scores = llm.score_yes_no_with_probabilities(
                filter_prompts, batch_size=args.batch_size
            )
            for (context_index, name), (decision, _) in zip(
                filter_references, filter_scores
            ):
                allowed_by_context[context_index][name] = decision

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
            prediction = rng.choice(positive) if positive else NO_RELATION
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

    for episode_key in episode_keys:
        remaining_queries[episode_key] -= 1
        if remaining_queries[episode_key] == 0:
            completed_episodes += 1
            if completed_episodes % 100 == 0 or completed_episodes == total_episodes:
                print(
                    f"{completed_episodes}/{total_episodes} episodes complete",
                    flush=True,
                )
    write_jsonl(args.output, output)


if __name__ == "__main__":
    main()
