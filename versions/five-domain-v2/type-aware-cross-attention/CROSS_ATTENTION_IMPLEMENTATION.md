# Semantic Structure Cross Attention

This is an incremental extension of BGE-M3 + HGT, not a new pure-HGT model and
not the full document's self-supervised pretraining / graph prompt pipeline.
The document section 3.2 specifies two attention streams and weighted message
probabilities but leaves direction, gate granularity and User semantics open.
Those choices are explicitly frozen in the adjacent YAML, not claimed as quotes.

For an incoming edge s -> t and head h at layer l:

    a_struct = softmax over incoming edges of the original HGT logits
    a_cross  = MHA(Q = hidden_l[t], K = frozen_semantic[neighbors(t)]).weights
    lambda_l = sigmoid(gate_l), initialized to 0.5
    a_final  = (1-lambda_l) * a_struct + lambda_l * a_cross
    message  = sum_s a_final[s,t,h] * original_HGT_relation_value[s,t,h]

The same raw semantic cache is supplied at every layer and in full-neighbor
inference; it is never replaced by previous-layer hidden states. Item and
Category use frozen BGE vectors. User uses the unchanged train-only mean Item
vector, never a language-model call. This User vector is a derived semantic
proxy, not a direct LLM encoding of User. GNN states are structure-conditioned,
not mathematically pure structure because they start from semantic features.

PyTorch nn.MultiheadAttention performs Q/K projection, scaling, softmax and
masking. need_weights=True and average_attn_weights=False preserve per-head
edge weights. Its attention dropout is zero to keep normalized probabilities.
Since only weights are fused, dummy values are used and the unused V/output
projection parameters are frozen and excluded from trainable parameter counts.
The original HGT values, residual, activation, layer norm and dropout remain.

Edges are packed by destination degree and bucketed to bound padding. Padding
is masked. Each destination retains ALL of its incoming edges in the current
graph; no softmax is computed independently for single edges or neighbor chunks.
Isolated nodes skip cross attention. No global all-node attention matrix is built.

The standard HGT source gains only a default softmax extension hook; baseline
parameter names and computations are unchanged. Archived server v2 code and
the twelve baseline results are not overwritten. The new trainer has its own
revision and refuses baseline checkpoints. Both raw-text and Type-Aware caches
are supported with strict variant validation; the user confirmed Type-Aware
features for the formal experiment. The separate runner launches seeds 42/43/44.

Tests cover per-destination MHA reference equality, probability normalization,
weighted fusion, semantic sensitivity, edge ordering, input freezing, Q/K/gate
gradients, exact disabled-module HGT equality, isolated nodes, full inference,
and synthetic training/evaluation/resume. Synthetic values are never paper results.

The incremental comparison retains v2's two layers/128 dimensions, graph, splits,
candidates and optimizer. The document's four-layer/256-dimensional pretraining
configuration is a separate later experiment; changing it here would confound
the cross-attention comparison. GPU capacity/performance on full-scale training
still requires measurement; small tests do not establish production runtime.

API reference: https://docs.pytorch.org/docs/2.8/generated/torch.nn.MultiheadAttention.html
