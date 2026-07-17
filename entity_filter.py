"""LLM-based subject/object compatibility filter used before relation inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from model import PaperLLM
from prompts import render
from utils import entity_text


def build_filter_prompt(
    query: Mapping[str, Any], relation_name: str, relation_metadata: Mapping[str, str]
) -> str:
    return render(
        "entity_filter",
        relation=relation_name,
        subject_type=relation_metadata["subject_type"],
        object_type=relation_metadata["object_type"],
        subject=entity_text(query, "subj"),
        object=entity_text(query, "obj"),
    )


def compatible_relations(
    llm: PaperLLM,
    query: Mapping[str, Any],
    relation_names: Sequence[str],
    metadata: Mapping[str, Mapping[str, str]],
    batch_size: int = 8,
) -> dict[str, bool]:
    prompts = [build_filter_prompt(query, name, metadata[name]) for name in relation_names]
    decisions = llm.score_yes_no(prompts, batch_size=batch_size)
    return dict(zip(relation_names, decisions))

