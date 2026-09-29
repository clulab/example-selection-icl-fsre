"""Paper-faithful subject/object compatibility filtering."""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from prompts import render
from utils import entity_text


EntityFilterMap = dict[str, dict[str, bool]]


def tagged_entity_sentence(query: Mapping[str, Any], role: str) -> str:
    """Return the original filter input with one entity marked."""

    if role not in {"subj", "obj"}:
        raise ValueError("role must be 'subj' or 'obj'")
    tokens = list(query["token"])
    start = int(query[f"{role}_start"])
    end = int(query[f"{role}_end"])
    if not 0 <= start <= end < len(tokens):
        raise ValueError(f"Invalid {role} span in example {query.get('id')}")
    tokens[start] = "<entity>" + tokens[start]
    tokens[end] += "</entity>"
    return " ".join(tokens)


def build_filter_prompt(
    query: Mapping[str, Any], role: str, entity_type: str
) -> str:
    """Render the exact entity prompt used to create the paper's filter maps."""

    return render(
        "entity_filter",
        sentence=tagged_entity_sentence(query, role),
        entity=entity_text(query, role),
        entity_type=entity_type,
    )


def filter_prompts(
    query: Mapping[str, Any], relation_metadata: Mapping[str, str]
) -> tuple[str, str]:
    """Return the subject and object prompts for one query/relation pair."""

    return (
        build_filter_prompt(query, "subj", relation_metadata["subject_type"]),
        build_filter_prompt(query, "obj", relation_metadata["object_type"]),
    )


def load_entity_filter_map(path: str | Path) -> EntityFilterMap:
    """Load and validate a trusted paper entity-filter pickle."""

    with Path(path).open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Entity-filter map must be a dictionary")

    result: EntityFilterMap = {}
    for query_id, relation_decisions in value.items():
        if not isinstance(relation_decisions, dict):
            raise ValueError(
                f"Entity-filter entry for query {query_id!r} must be a dictionary"
            )
        validated: dict[str, bool] = {}
        for relation, decision in relation_decisions.items():
            if not isinstance(decision, bool):
                raise ValueError(
                    f"Entity-filter decision for {query_id!r}/{relation!r} "
                    "must be Boolean"
                )
            validated[str(relation)] = decision
        result[str(query_id)] = validated
    return result


def compatible_relations(
    llm: Any,
    query: Mapping[str, Any],
    relation_names: Sequence[str],
    metadata: Mapping[str, Mapping[str, str]],
    batch_size: int = 8,
) -> dict[str, bool]:
    """Recompute compatibility, requiring both entity-role checks to pass."""

    prompts: list[str] = []
    for name in relation_names:
        prompts.extend(filter_prompts(query, metadata[name]))
    decisions = llm.score_yes_no(prompts, batch_size=batch_size)
    if len(decisions) != 2 * len(relation_names):
        raise RuntimeError("The model returned an unexpected number of filter scores")
    return {
        name: decisions[2 * index] and decisions[2 * index + 1]
        for index, name in enumerate(relation_names)
    }
