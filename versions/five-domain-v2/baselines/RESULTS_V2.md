# V2 test results

9/12 runs complete. Uniform-prompt group not trained: feature upload incomplete.
Mean ± sample standard deviation across seeds 42/43/44. All numbers below are percentages.

| Method | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | AUC | AP |
|---|---:|---:|---:|---:|---:|---:|
| pure-hgt | 0.1883 ± 0.0590 | 0.3332 ± 0.1103 | 0.0907 ± 0.0271 | 0.1270 ± 0.0393 | 73.0787 ± 3.6748 | 3.6463 ± 0.7408 |
| type-aware-prompt-hgt | 0.2422 ± 0.1003 | 0.4108 ± 0.1775 | 0.1197 ± 0.0501 | 0.1620 ± 0.0697 | 75.8026 ± 1.7410 | 4.9954 ± 1.2135 |
| unprompted-bge-m3-hgt | 0.1827 ± 0.0774 | 0.3466 ± 0.1154 | 0.0851 ± 0.0414 | 0.1261 ± 0.0509 | 76.7611 ± 1.1233 | 5.2450 ± 0.9266 |

All completed model/result hashes verified on server; local result hashes verified; models not downloaded by this export

Do not infer the isolated Type-Aware Prompt effect without the content-matched uniform baseline.
