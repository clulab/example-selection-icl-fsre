# Complete FS-FewRel release

This directory contains the complete FewRel material required by the paper's FS-FewRel experiments.

| File | Records | Description |
|---|---:|---|
| `source_examples.jsonl.gz` | 70,000 | Source examples from the local FewRel train, validation, and test splits |
| `support_examples.jsonl.gz` | 4,200 | Unique 1-shot supports from the six selected relations |
| `query_examples.jsonl.gz` | 68,771 | Unique episode queries; includes episode-level `no_relation` candidates |
| `semantic_rules.jsonl.gz` | 4,200 | One shortest-path semantic rule per support |
| `episodes/seed_*.jsonl.gz` | 30,000 each | Five flattened evaluation sets, 150,000 rows total |
| `relations.json` | 16 relations | Validation relation descriptions and the six selected relation/type pairs |
| `manifest.json` | — | File sizes and SHA-256 checksums |

The six candidate relations are:

| ID | Relation | Subject → Object |
|---|---|---|
| `P25` | mother | PERSON → PERSON |
| `P26` | spouse | PERSON → PERSON |
| `P40` | child | PERSON → PERSON |
| `P206` | located in or next to body of water | LOCATION → LOCATION |
| `P410` | military rank | PERSON → TITLE |
| `P463` | member of | PERSON → ORGANIZATION |

## Rule mapping

A semantic-rule record has this form:

```json
{
  "source_example_id": "P40_56",
  "relation_id": "P40",
  "relation": "child",
  "rule": "[entity=PERSON]+ ... [entity=PERSON]+"
}
```

`source_example_id` is also the `id` in `source_examples.jsonl.gz` and `support_examples.jsonl.gz`. The ID splits at its final underscore: `P40_56` maps to relation `P40`, zero-based source index `56`.

```bash
python prepare_data.py map-fewrel-rule P40_56
```

## Episode format

Each compressed JSONL row contains one flattened evaluation query:

```json
{
  "set_id": 0,
  "seed": 160290,
  "episode_id": 0,
  "query_index": 0,
  "support_ids": ["P40_56", "P206_135", "P26_686", "P410_355", "P25_653"],
  "candidate_relations": ["child", "located in or next to body of water", "spouse", "military rank", "mother"],
  "query_id": "P1411_594",
  "query_relation": "...",
  "gold_relation": "no_relation"
}
```

`query_relation` is the source relation retained for traceability. `gold_relation` is the task label: it equals `query_relation` only when that relation is one of the five candidates, and is otherwise `no_relation`. This produces 142,503 `no_relation` rows out of 150,000 (95.002%).

The released pickle was converted to JSONL so users do not need to execute pickle data. The five files preserve the exact original order in contiguous 30,000-row sets with seeds `160290` through `160294`.

