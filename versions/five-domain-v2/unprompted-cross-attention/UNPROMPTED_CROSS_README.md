# Raw text BGE-M3 HGT with Cross Attention

User-requested new control: remove Type-Aware text prompts, keep frozen BGE-M3,
standard HGT and the exact Semantic-Structure Cross Attention implementation.
Reuse the existing unprompted cache; no re-encoding or graph/protocol changes.
HGT's node types and type-specific projections remain part of the architecture;
removing Type-Aware Prompt does not mean removing heterogeneous node types.

The primary comparison is unprompted BGE-M3 + HGT versus this new model: only
Cross Attention is added. Comparison against Type-Aware + Cross Attention also
changes the original text payload (the raw Category cache lacks parent/example
fields), so it is not a strict content-matched template-only ablation.

Seeds: 42, 43, 44. Use v2 protocol, all frozen candidates, identical graph and
training/evaluation settings. Source code is copied from the completed cross-v1
experiment without edits. Only feature path, feature manifest, experiment name
and output directory differ. No earlier result or checkpoint is overwritten.

Server: EXPERIMENT_ROOT/v2/unprompted-cross-v1
Local: EXPERIMENT_ROOT/experiments/unprompted-cross-v1
Runner: run_unprompted_cross_all.sh (shared parent flock lock).
