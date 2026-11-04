# Structured Example Selection for Few-Shot Relation Extraction

Code and data for:

> Aunabil Chakma, Mihai Surdeanu, and Eduardo Blanco. 2026.
>
> **[Structured Semantic Information Helps Retrieve Better Examples for In-Context Learning Applied to Few-Shot Relation Extraction](https://aclanthology.org/2026.acl-long.1664/).**
>
> ACL 2026.

This repository contains the core implementation, prompts, FS-FewRel data, and TACRED semantic rules used in the paper.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If a Hugging Face model requires authentication, set `HF_TOKEN` in your environment. Do not place tokens in source files.

## Validate the data

```bash
python prepare_data.py validate
python prepare_data.py map-fewrel-rule P40_56
```

## Select retrieved examples

`select_examples.py` accepts a file containing candidate examples, scores, and embeddings:

```bash
python select_examples.py \
  --input external_candidates.jsonl.gz \
  --output selected.jsonl.gz \
  --strategy kmeanspp_farthest \
  --count 4 \
  --threshold 0.6
```

Use `--count 4` for 5-shot experiments and `--count 9` for 10-shot experiments.

## Run inference

```bash
python run_inference.py \
  --episodes <episodes.jsonl.gz> \
  --supports <support_examples.jsonl.gz> \
  --queries <query_examples.jsonl.gz> \
  --relations <relations.json> \
  --model Qwen/Qwen3-4B \
  --entity-filter \
  --output outputs/predictions.jsonl.gz
```

The same command supports FewRel and authorized TACRED data using the common JSONL schema. Pass selected or generated examples through `--extra-examples` for 5- or 10-shot experiments. A unified file containing several variants is selected with `--extra-method` and `--additional-count`:

```bash
python run_inference.py \
  --episodes datasets/tacred/authorized/episodes_50k.jsonl.gz \
  --supports datasets/tacred/authorized/support_examples.jsonl.gz \
  --queries datasets/tacred/authorized/query_examples.jsonl.gz \
  --relations datasets/tacred/authorized/relations.json \
  --extra-examples datasets/tacred/authorized/additional_examples_qwen3_4b.jsonl.gz \
  --extra-method softmatcher_closest \
  --additional-count 4 \
  --model Qwen/Qwen3-4B \
  --entity-filter-map datasets/tacred/fs_tacred_entity_resolution_Qwen-Qwen3-4B.pkl \
  --output outputs/tacred_softmatcher_1plus4.jsonl.gz
```

Use `--additional-count 4` for `1+4` and `--additional-count 9` for `1+9`. The renderer automatically uses the exact 5-shot or 10-shot prompt from the original experiments, including the original summarized prompt for methods ending in `_summarized`.

Use `--entity-filter` to recompute the paper's subject/object compatibility filter. To reuse a trusted saved filter map without recomputing it, pass `--entity-filter-map /path/to/entity_resolution.pkl` instead; the map option enables entity filtering by itself.

## Evaluate

```bash
python evaluate.py outputs/predictions.jsonl.gz
```

## Citation

If you use our work, please cite our paper using the BibTeX below:

```bibtex
@inproceedings{chakma-etal-2026-structured,
    title = "Structured Semantic Information Helps Retrieve Better Examples for In-Context Learning Applied to Few-Shot Relation Extraction",
    author = "Chakma, Aunabil  and
      Surdeanu, Mihai  and
      Blanco, Eduardo",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.1664/",
    doi = "10.18653/v1/2026.acl-long.1664",
    pages = "35947--35971",
    ISBN = "979-8-89176-390-6"
}
```
