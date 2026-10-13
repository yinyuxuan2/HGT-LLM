# HGT-LLM experiments

Code, configurations, tests, and small result records for the Amazon Reviews 2018 heterogeneous-graph recommendation experiments. The repository preserves historical versions separately; a later correction does **not** overwrite an earlier result.

## Versions

| Directory | Experiment | Status |
| --- | --- | --- |
| [`versions/legacy-sports-ours-min`](versions/legacy-sports-ours-min) | Initial single-category Sports & Outdoors BGE-M3 + standard HGT | Legacy protocol; not comparable with five-domain results |
| [`versions/five-domain-v1/pure-hgt`](versions/five-domain-v1/pure-hgt) | HGT with trainable ID features, no language model | Code/config archived; the local v1 export is incomplete, so no three-seed aggregate is asserted here |
| [`versions/five-domain-v1/bge-m3-hgt`](versions/five-domain-v1/bge-m3-hgt) | Frozen BGE-M3 node features + HGT, without type-aware prompts | Historical five-domain protocol |
| [`versions/five-domain-v1/type-aware-prompt-hgt`](versions/five-domain-v1/type-aware-prompt-hgt) | Type-Aware Prompt Encoding + frozen BGE-M3 + HGT | Historical five-domain protocol |
| [`versions/five-domain-v2/baselines`](versions/five-domain-v2/baselines) | Corrected frozen protocol and baseline trainers | Exported summary covers pure HGT, unprompted BGE-M3 + HGT, and Type-Aware BGE-M3 + HGT; uniform-prompt code exists but a complete local three-seed summary was not available at publication |
| [`versions/five-domain-v2/type-aware-cross-attention`](versions/five-domain-v2/type-aware-cross-attention) | Type-Aware BGE-M3 + HGT + semantic–structure cross-attention | Separate incremental experiment on v2 protocol |
| [`versions/five-domain-v2/unprompted-cross-attention`](versions/five-domain-v2/unprompted-cross-attention) | Unprompted BGE-M3 + HGT + semantic–structure cross-attention | Separate incremental experiment on v2 protocol |
| [`versions/five-domain-v3/rank-corrected`](versions/five-domain-v3/rank-corrected) | Type-Aware and pure HGT with supervision-edge masking and paired BPR | Six completed runs, seeds 42/43/44 |

The five Amazon categories are All Beauty, Books, Electronics, Sports & Outdoors, and Toys & Games. Review interactions are treated as a **purchase proxy**, not a verified purchase event. The graph uses User–Item interaction, Item–Category membership, and Item–Item co-purchase semantics; there is no view edge. Evaluation uses the frozen train/validation/test split, candidates, and negative samples, masks already-interacted training items, and reports Recall@10/20, NDCG@10/20, AUC, and AP. Protocol and negative-array fingerprints are in [`manifests`](manifests).

## Results

See [`RESULTS.md`](RESULTS.md) for the concise comparison and each version's `results_summary*.json` for full-precision values, three-seed sample standard deviations, per-seed records, timing, GPU memory, and hashes. **Do not mix v1 and v2 scores as though they shared one protocol; do not treat v2 and v3 as the same training configuration.** The v3 run changes both the training loss and sampled supervision-edge handling, while retaining the graph, features, candidate set, and HGT architecture.

## Reproduction

1. Create a Linux/PyTorch environment using [`environment`](environment). Install the CUDA-compatible PyTorch build separately for your GPU/driver. The public launchers call `python` from your environment; verify that it points to the intended interpreter.
2. Obtain Amazon Reviews 2018 review and metadata files for the five named categories and BAAI's BGE-M3 weights separately. Raw data and pretrained weights are **not** redistributed here.
3. Reconstruct the frozen protocol, graph, and features using the preparation/encoding scripts under `versions/five-domain-v2/baselines/code/scripts`. Check the manifest and array hashes before training. The archived run scripts expect sibling `protocol/`, `prepared/`, `inputs/`, and `features/` runtime directories; these large inputs are deliberately omitted from Git.
4. Run the appropriate version-specific launcher after adapting paths to your environment. The v3 trainer imports the baseline HGT modules from v2, so overlay its `code/scripts/*.py` on a copy of the v2 `code/` tree before running `run_rank_v3.sh`.
5. Keep seeds 42, 43, and 44; use the same candidate set and evaluation script for comparisons. Never substitute sampled results for the full frozen test evaluation.

The Type-Aware text templates exclude Item category information. Item and Category text is encoded once with frozen BGE-M3; User features do not invoke the language model. The cross-attention versions are incremental additions, **not** an implementation of a separate pretraining or graph prompt-tuning stage.

## What is not in Git

Raw Amazon data, user/item tables, encoded feature arrays, BGE-M3 weights, checkpoints, model files, server logs, SSH keys, and machine-specific deployment archives remain outside this repository. Model files exceed ordinary GitHub file limits and are preserved in the local experiment store, with SHA-256 values recorded in the result records. No trained checkpoint is claimed to be downloadable from this repository.

The HGT and text-encoder approaches build on [pyHGT](https://github.com/acbull/pyHGT) and [BGE-M3 / FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding); their upstream code and model weights are not vendored here.

The public copy replaces machine-specific absolute path prefixes with the placeholder `EXPERIMENT_ROOT`; set this to your own runtime path. Numeric metrics and recorded artifact SHA-256 values are unchanged, but those hash fields identify the **original private experiment artifacts**, not the path-sanitized JSON/config/launcher copies in Git. Original records remain in the local experiment store.
