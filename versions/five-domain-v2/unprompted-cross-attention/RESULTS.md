# Cross Attention test results

All three seeds completed. Mean ± sample standard deviation, percentages.

| Metric | unprompted-bge-m3-hgt | With Cross Attention | Relative mean change |
|---|---:|---:|---:|
| recall@10 | 0.1827 ± 0.0774 | 0.2358 ± 0.0960 | +29.12% |
| recall@20 | 0.3466 ± 0.1154 | 0.3984 ± 0.1508 | +14.96% |
| ndcg@10 | 0.0851 ± 0.0414 | 0.1126 ± 0.0489 | +32.33% |
| ndcg@20 | 0.1261 ± 0.0509 | 0.1533 ± 0.0627 | +21.63% |
| auc | 76.7611 ± 1.1233 | 74.3451 ± 1.8156 | -3.15% |
| ap | 5.2450 ± 0.9266 | 4.3157 ± 0.7892 | -17.72% |

All four ranking means increased, while AUC/AP decreased; no significance claim.
Model/result hashes verified on server; local metric hashes verified. Models not downloaded by this export.
Total run time: 13.60 hours.
Maximum recorded allocated GPU memory: 12296.20 MiB.
