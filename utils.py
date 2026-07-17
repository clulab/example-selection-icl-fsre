"""Small, dependency-light utilities shared by the release scripts."""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


def open_text(path: str | Path, mode: str = "rt"):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def read_json(path: str | Path) -> Any:
    with open_text(path) as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_text(path, "wt") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object on {path}:{line_number}")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_text(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def index_jsonl(path: str | Path, key: str = "id") -> dict[Any, dict[str, Any]]:
    indexed: dict[Any, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if key not in row:
            raise KeyError(f"{path} contains a row without {key!r}")
        if row[key] in indexed:
            raise ValueError(f"Duplicate {key}={row[key]!r} in {path}")
        indexed[row[key]] = row
    return indexed


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> random.Random:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return random.Random(seed)


def tagged_sentence(example: Mapping[str, Any]) -> str:
    tokens = list(example["token"])
    required = ("subj_start", "subj_end", "obj_start", "obj_end")
    if any(name not in example for name in required):
        raise KeyError(f"Example {example.get('id')} is missing entity spans")
    ss, se = int(example["subj_start"]), int(example["subj_end"])
    os, oe = int(example["obj_start"]), int(example["obj_end"])
    if not (0 <= ss <= se < len(tokens) and 0 <= os <= oe < len(tokens)):
        raise ValueError(f"Invalid entity span in example {example.get('id')}")
    tokens[ss] = "<subject>" + tokens[ss]
    tokens[se] += "</subject>"
    tokens[os] = "<object>" + tokens[os]
    tokens[oe] += "</object>"
    return " ".join(tokens)


def entity_text(example: Mapping[str, Any], role: str) -> str:
    if role not in {"subj", "obj"}:
        raise ValueError("role must be 'subj' or 'obj'")
    start = int(example[f"{role}_start"])
    end = int(example[f"{role}_end"])
    return " ".join(example["token"][start : end + 1])


def parse_integer_list(text: str, expected: int | None = None) -> list[int]:
    """Safely parse an LLM response containing a Python/JSON-style list."""

    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        raise ValueError(f"No list found in response: {text!r}")
    try:
        value = ast.literal_eval(text[start : end + 1])
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid list response: {text!r}") from exc
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise ValueError(f"Expected a list of integers: {text!r}")
    result = list(value)
    if expected is not None and len(result) != expected:
        raise ValueError(f"Expected {expected} selections, received {len(result)}")
    return result


def parse_numbered_examples(text: str, expected: int) -> list[str]:
    examples: dict[int, str] = {}
    for line in text.splitlines():
        head, separator, tail = line.partition(":")
        if not separator:
            continue
        try:
            index = int(head.strip().rstrip("."))
        except ValueError:
            continue
        if 1 <= index <= expected and tail.strip():
            examples[index] = tail.strip()
    missing = [index for index in range(1, expected + 1) if index not in examples]
    if missing:
        raise ValueError(f"Missing numbered examples: {missing}")
    return [examples[index] for index in range(1, expected + 1)]


def batched(values: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]
