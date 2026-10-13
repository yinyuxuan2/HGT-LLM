# Cross Attention test results

All three seeds completed. Mean ± sample standard deviation, percentages.

| Metric | Type-Aware HGT | With Cross Attention | Relative mean change |
|---|---:|---:|---:|
| recall@10 | 0.2422 ± 0.1003 | 0.2402 ± 0.0840 | -0.83% |
| recall@20 | 0.4108 ± 0.1775 | 0.4219 ± 0.1040 | +2.69% |
| ndcg@10 | 0.1197 ± 0.0501 | 0.1151 ± 0.0451 | -3.89% |
| ndcg@20 | 0.1620 ± 0.0697 | 0.1607 ± 0.0497 | -0.79% |
| auc | 75.8026 ± 1.7410 | 76.7289 ± 1.5643 | +1.22% |
| ap | 4.9954 ± 1.2135 | 5.3510 ± 1.0701 | +7.12% |

No consistent improvement in ranking metrics; no significance claim.
Model/result hashes verified on server; local metric hashes verified. Models not downloaded by this export.
Total run time: 23.35 hours.
Maximum recorded allocated GPU memory: 12296.20 MiB.
