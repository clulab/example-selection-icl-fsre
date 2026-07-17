# Datasets

The `fewrel/` directory contains the FewRel source data, episode details, relation metadata, and semantic rules used in this work. The original example IDs are preserved so every rule can be mapped back to its source example.

The `tacred/` directory contains the semantic rules and their original TACRED source example IDs. TACRED sentences and relation labels are not included because they require authorized access. Users with TACRED access can join their local examples to the released rules using `prepare_data.py`.

UMBC data and SoftMatcher are not included. These resources are from **[Best of Both Worlds: A Pliable and Generalizable Neuro-Symbolic Approach for Relation Classification](https://aclanthology.org/2024.findings-naacl.165/)** by Vacareanu et al. (Findings of NAACL 2024). Please follow its [code and data release](https://github.com/clulab/releases/tree/master/naacl2024-softrules) to obtain or prepare those resources.

Embeddings, indices, checkpoints, and model weights are not distributed in this repository.
