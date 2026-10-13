# Five-domain test results

Values below are three-seed means on the frozen full candidate evaluation. Full precision, sample standard deviations, AUC/AP, runtime, memory, parameter counts, and hashes are in the linked JSON files. These are historical experiment records, not newly recomputed scores.

| Protocol/training | Method | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | Record |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| v2 | Pure HGT | 0.001883 | 0.003332 | 0.000907 | 0.001270 | [summary](versions/five-domain-v2/baselines/results_summary_v2.json) |
| v2 | Unprompted BGE-M3 + HGT | 0.001827 | 0.003466 | 0.000851 | 0.001261 | [summary](versions/five-domain-v2/baselines/results_summary_v2.json) |
| v2 | Type-Aware BGE-M3 + HGT | 0.002422 | 0.004108 | 0.001197 | 0.001620 | [summary](versions/five-domain-v2/baselines/results_summary_v2.json) |
| v2 + cross-attention | Type-Aware BGE-M3 + HGT | 0.002402 | 0.004219 | 0.001151 | 0.001607 | [summary](versions/five-domain-v2/type-aware-cross-attention/results_summary.json) |
| v2 + cross-attention | Unprompted BGE-M3 + HGT | 0.002358 | 0.003984 | 0.001126 | 0.001533 | [summary](versions/five-domain-v2/unprompted-cross-attention/results_summary.json) |
| v3 paired BPR + edge mask | Pure HGT | 0.001904 | 0.003188 | 0.001014 | 0.001337 | [summary](versions/five-domain-v3/rank-corrected/results_summary_rank_v3.json) |
| v3 paired BPR + edge mask | Type-Aware BGE-M3 + HGT | 0.003171 | 0.005463 | 0.001596 | 0.002171 | [summary](versions/five-domain-v3/rank-corrected/results_summary_rank_v3.json) |

The v3 Type-Aware Recall@10 exceeds its v2 historical value, but the change combines two training corrections; it is **not** an isolated effect estimate for either correction. The v2 cross-attention runs do not demonstrate a consistent Top-K improvement over their corresponding v2 baselines. The original v1 five-domain and legacy Sports runs use earlier protocols and are archived for traceability rather than placed in this comparison table.

No claim is made here that the full paper proposal's pretraining/graph-prompt stages were implemented.
