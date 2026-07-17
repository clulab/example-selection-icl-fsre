"""Paper models and reproducibility defaults.

No model weights or credentials are distributed with this repository.  The
Hugging Face libraries automatically read ``HF_TOKEN`` when gated access is
needed.
"""

from __future__ import annotations

from dataclasses import dataclass


MODEL_IDS = {
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "gemma3-4b": "google/gemma-3-4b-it",
    "gemma3-12b": "google/gemma-3-12b-it",
}

SBERT_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"


@dataclass(frozen=True)
class GenerationSettings:
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int


GENERATION_SETTINGS = {
    "qwen": GenerationSettings(True, 0.6, 0.95, 20),
    "gemma": GenerationSettings(True, 1.0, 0.95, 64),
}

DEFAULT_SEED = 42
EPISODE_SEEDS = (160290, 160291, 160292, 160293, 160294)
SIMILARITY_THRESHOLD = 0.6
BOOTSTRAP_SAMPLES = 1000
NO_RELATION = "no_relation"


def model_family(model_id: str) -> str:
    lowered = model_id.lower()
    if "qwen" in lowered:
        return "qwen"
    if "gemma" in lowered:
        return "gemma"
    raise ValueError(f"No paper generation settings registered for {model_id!r}")

