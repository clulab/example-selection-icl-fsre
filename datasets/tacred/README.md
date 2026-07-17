# TACRED semantic rules

This directory contains the 804 unique semantic rules needed for the paper's TACRED support examples. A record contains only:

```json
{
  "source_example_id": "098f6439db54a6e6ae07",
  "rule": "[entity=PERSON]+ ... [entity=DATE]+"
}
```

The stable `source_example_id` is maintained deliberately. It maps the rule to the same original TACRED instance once the user has obtained authorized TACRED data. The release does **not** include the corresponding sentence, relation label, document ID, token sequence, entity offsets, or TACRED episodes.

## Obtain TACRED

Follow the official data-access instructions at <https://nlp.stanford.edu/projects/tacred/>. Store authorized files under `datasets/tacred/authorized/`; that directory is ignored by Git.

An authorized JSON or JSONL source must expose each original example under its `id`. Join all 804 public rules to those records with:

```bash
python prepare_data.py join-tacred \
  --authorized-examples datasets/tacred/authorized/examples.jsonl \
  --output datasets/tacred/authorized/rules_with_examples.jsonl
```

The command fails if even one of the 804 source IDs is missing. This verifies the mapping without redistributing protected TACRED content.

`manifest.json` records the checksum of the public rule file.

