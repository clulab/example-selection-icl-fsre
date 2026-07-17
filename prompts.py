"""Load and render the prompt templates distributed in ``prompts/``."""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Iterable, Mapping


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(name: str) -> str:
    path = PROMPT_DIR / f"{name}.txt"
    if not path.is_file():
        available = ", ".join(sorted(item.stem for item in PROMPT_DIR.glob("*.txt")))
        raise KeyError(f"Unknown prompt {name!r}; available prompts: {available}")
    return path.read_text(encoding="utf-8").strip()


def numbered_supports(supports: Iterable[str]) -> str:
    return "\n".join(
        f"Support Sentence {index}: {sentence}"
        for index, sentence in enumerate(supports, start=1)
    )


def render(name: str, **values: object) -> str:
    string_values = {key: str(value) for key, value in values.items()}
    return Template(load_prompt(name)).substitute(string_values)


def binary_relation_prompt(
    relation: str,
    description: str,
    supports: Iterable[str],
    query: str,
) -> str:
    return render(
        "binary_relation",
        relation=relation,
        description=description,
        supports=numbered_supports(supports),
        query=query,
    )


def multiclass_prompt(
    relations: Iterable[Mapping[str, str]],
    supports_by_relation: Mapping[str, Iterable[str]],
    query: str,
) -> str:
    blocks = []
    labels = []
    for index, metadata in enumerate(relations, start=1):
        name = metadata["name"]
        labels.append(name)
        blocks.append(
            f"Relation {index}: {name} ({metadata['description']})\n"
            + numbered_supports(supports_by_relation[name])
        )
    return render(
        "multiclass",
        relation_blocks="\n\n".join(blocks),
        labels=", ".join(labels),
        query=query,
    )

