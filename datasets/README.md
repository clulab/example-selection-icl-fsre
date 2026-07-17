# Datasets

This directory separates fully redistributable FewRel material from the restricted TACRED source text.

## FewRel

`fewrel/` is complete for the experiments in this paper. It contains every source record needed to resolve episode IDs, the paper's support and query views, the five evaluation sets, relation metadata, semantic rules, and checksums.

Rules use stable source IDs. `P40_56`, for example, maps to the original `P40` example at zero-based index `56`. Run `python prepare_data.py map-fewrel-rule P40_56` from the repository root to inspect the joined record.

The original FewRel project is available at <https://github.com/thunlp/FewRel>. Please retain the original dataset citation and license notice when redistributing these files.

## TACRED

`tacred/` contains 804 semantic rules used for the paper's unique TACRED support examples. Each rule retains its original source example ID. It does not contain the source sentence, source relation label, entity spans, or a complete TACRED episode.

Authorized TACRED users can join the rules to their local source records with `prepare_data.py`. TACRED access information is available from <https://nlp.stanford.edu/projects/tacred/>.

## UMBC and SoftMatcher

They are not distributed here. Follow the NAACL 2024 “Best of Both Worlds” paper and release linked from the main README. Embeddings, checkpoints, indices, and model weights must remain outside this Git repository.

