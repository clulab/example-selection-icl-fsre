# Structured Example Selection for In-Context Few-Shot Relation Extraction

This is the compact release for **“Structured Semantic Information Helps Retrieve Better Examples for In-Context Learning Applied to Few-Shot Relation Extraction”** (Chakma, Surdeanu, and Blanco; ACL 2026).

The repository contains the paper's core Python implementation, executable prompt templates, complete FS-FewRel source and episode data, and the semantic rules for the TACRED support examples. It intentionally contains no model weights, embeddings, FAISS indices, UMBC sentences, SoftMatcher checkpoint, raw TACRED sentences, or experiment logs.

## Repository contents

```text
.
├── config.py                 # Paper models, seeds, and generation settings
├── utils.py                  # JSONL, tagging, parsing, checksums, and seeding
├── prompts.py                # Prompt loading and rendering
├── model.py                  # Hugging Face generation and yes/no scoring
├── prepare_data.py           # Dataset validation and ID mapping
├── entity_filter.py          # LLM entity-role compatibility filter
├── generate_examples.py      # Generate, paraphrase, summarize, hybrid-select
├── select_examples.py        # Closest and clustering-based candidate selection
├── run_inference.py          # Binary/multiclass ICL inference
├── evaluate.py               # Micro P/R/F1 and paired bootstrap
├── prompts/                  # Eight editable prompt templates
└── datasets/
    ├── fewrel/               # Complete paper-specific FewRel release
    └── tacred/               # Rule + source-ID pairs only
```

See [datasets/README.md](datasets/README.md) for the data policy and schemas.

## What is and is not included

| Resource | Included? | Contents |
|---|---:|---|
| FewRel source examples | Yes | 70,000 examples across the local train, validation, and test source splits |
| FS-FewRel episodes | Yes | Five sets, 150,000 flattened evaluation rows total |
| FewRel support/query details | Yes | 4,200 unique supports and 68,771 unique queries |
| FewRel semantic rules | Yes | 4,200 rules with direct source-example IDs |
| TACRED semantic rules | Yes | 804 paper support rules with original TACRED source IDs |
| TACRED sentences and labels | No | Users must obtain authorized TACRED data |
| UMBC and SoftMatcher | No | Follow the external NAACL 2024 release described below |
| Embeddings, indices, checkpoints, LLM weights | No | Created/downloaded by the user outside this repository |

## UMBC and SoftMatcher dependency

The UMBC preprocessing, candidate collection, SoftMatcher model, and corresponding embeddings originate from:

> Robert Vacareanu, Fahmida Alam, Md Asiful Islam, Haris Riaz, and Mihai Surdeanu. 2024. **Best of Both Worlds: A Pliable and Generalizable Neuro-Symbolic Approach for Relation Classification.** Findings of NAACL 2024.

- Paper: <https://aclanthology.org/2024.findings-naacl.165/>
- Associated release: <https://github.com/clulab/releases/tree/master/naacl2024-softrules>

Follow that work to prepare or request the UMBC corpus, SoftMatcher checkpoint, and rule/sentence embeddings. This repository starts after that pipeline has produced candidate IDs, scores, sentences, and embeddings. No part of those external artifacts is copied here.

`select_examples.py` expects one JSONL record per support:

```json
{
  "support_id": "P40_56",
  "query_embedding": [0.1, 0.2],
  "candidates": [
    {"id": 123, "score": 0.81, "sentence": "...", "embedding": [0.2, 0.3]}
  ]
}
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The scripts never contain access tokens. If a gated Hugging Face model requires authentication, set `HF_TOKEN` in your environment. Model weights are downloaded by `transformers` into the user's normal cache and are not part of this repository.

Paper model IDs are defined in `config.py`:

- `Qwen/Qwen3-4B`
- `Qwen/Qwen3-14B`
- `google/gemma-3-4b-it`
- `google/gemma-3-12b-it`

## Validate and inspect the included data

```bash
python prepare_data.py validate
```

Expected summary:

```text
70,000 source examples
4,200 support examples and semantic rules
68,771 unique query examples
150,000 evaluation rows in five sets
142,503 no_relation rows (95.002%)
```

Every FewRel rule maps directly to a source record. For example:

```bash
python prepare_data.py map-fewrel-rule P40_56
```

The rule ID `P40_56` means relation `P40`, example index `56` in the original relation list. See [datasets/fewrel/README.md](datasets/fewrel/README.md).

## Select retrieved candidates

Closest semantic-rule candidates:

```bash
python select_examples.py \
  --input external_candidates.jsonl.gz \
  --output selected.jsonl.gz \
  --strategy closest \
  --count 4 \
  --threshold 0.6
```

Available clustering strategies are `kmeans_random`, `kmeans_closest`, `kmeanspp_closest`, and `kmeanspp_farthest`. Use `--count 4` for the 5-shot setting and `--count 9` for the 10-shot setting.

By default, clustering records the executed artifact behavior `max(count, floor(sqrt(m)))`. Add `--paper-cluster-count` for the literal `floor(sqrt(m))` definition. The paper states a fixed threshold of `0.6`; `--adaptive-step 0.05` is available to reproduce the adaptive FewRel retrieval artifacts.

## Generate or transform examples

Input records must contain an `id`, `relation`, token/entity-span fields, and optionally a `candidates` list for hybrid selection.

```bash
# Generate nine new examples per support
python generate_examples.py generate \
  --input datasets/fewrel/support_examples.jsonl.gz \
  --relations datasets/fewrel/relations.json \
  --output outputs/generated.jsonl.gz \
  --model Qwen/Qwen3-4B \
  --count 9

# Paraphrase, summarize, or hybrid-select
python generate_examples.py paraphrase ...
python generate_examples.py summarize ...
python generate_examples.py hybrid ...
```

The Qwen and Gemma sampling settings are explicit in `config.py`. Qwen thinking mode is disabled for the paper prompts.

## Run FewRel inference

The command below runs the 1-shot binary baseline on the first episode set:

```bash
python run_inference.py \
  --episodes datasets/fewrel/episodes/seed_160290.jsonl.gz \
  --supports datasets/fewrel/support_examples.jsonl.gz \
  --queries datasets/fewrel/query_examples.jsonl.gz \
  --relations datasets/fewrel/relations.json \
  --model Qwen/Qwen3-4B \
  --entity-filter \
  --output outputs/qwen3_4b_seed_160290.jsonl.gz
```

For 5- or 10-shot ICL, pass the output of `select_examples.py` or `generate_examples.py` through `--extra-examples`. Other supported ablations are `--mode multiclass` and `--parametric`.

Binary inference compares the first-token `yes` and `no` logits. If more than one relation receives a positive vote, the implementation selects one using the configured random seed.

## Evaluate

```bash
python evaluate.py outputs/qwen3_4b_seed_160290.jsonl.gz
```

For a one-sided paired bootstrap comparison:

```bash
python evaluate.py outputs/system.jsonl.gz \
  --baseline outputs/baseline.jsonl.gz \
  --bootstrap-samples 1000 \
  --seed 42
```

Precision, recall, and F1 exclude `no_relation`. Means and population standard deviations are reported over the five episode sets.

## TACRED

`datasets/tacred/semantic_rules.jsonl.gz` contains only:

```json
{"source_example_id": "098f...", "rule": "[entity=PERSON]+ ..."}
```

The source ID is unchanged, allowing an authorized user to map the rule back to the original TACRED example. Raw sentences and relation labels are not included. See [datasets/tacred/README.md](datasets/tacred/README.md) for the join command.

## Citation

```bibtex
@inproceedings{chakma-etal-2026-structured,
  title     = {Structured Semantic Information Helps Retrieve Better Examples for In-Context Learning Applied to Few-Shot Relation Extraction},
  author    = {Chakma, Aunabil and Surdeanu, Mihai and Blanco, Eduardo},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics},
  year      = {2026}
}
```

