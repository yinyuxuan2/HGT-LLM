# 五类目纯 HGT 对照实验

本实验为结构与 ID 驱动的 HGT 基线，不调用 BGE-M3 或其他语言模型，不使用文本向量、预训练权重、蒸馏特征或语言模型生成的用户历史聚合。

## 控制变量

- 数据版本：amazon-2018-5domain-core5-v1。
- 节点：99,640 User、200,000 Item、8,885 Category；总计 308,525。
- 语义边：purchase（review proxy）、belongs_to、co_purchase；共 3,686,238 条。与原实验一致地加入反向消息传递后为 7,372,476 条有向邻接边。
- 复用服务器原 protocol 与 prepared/graph_metadata.npz；启动时逐文件校验协议 manifest 中的 SHA-256。
- 相同划分、全部候选商品、每种子冻结负样本、训练交互屏蔽规则、验证 NDCG@10 选模规则。
- 标准 HGT 层与评价代码的 SHA-256 分别为 4a49ccdace62f0555dc009d4f928d8991bc6f03f9cad2b0b83a6a116c0e82bf4、690ed628c19f11769768e05b113c13e6566a157e64558524e3ef62787b544a40，与原服务器实验一致。

## 唯一实验因素：初始节点表示

三类节点使用互不重叠的全局 ID，随机初始化可训练的 128 维 embedding，normal(0, 0.02)。没有冻结文本编码器、文本投影或 BGE 用户均值特征。HGT 的类型/关系注意力保持不变，仍以点积预测 User–Item 链接。

可训练参数量 39,938,206；此前 BGE 对照为 13,594,782。本实验不是参数量匹配的消融，论文应同时报告参数量。节点 ID 基线属于同一转导式协议，不证明未知 ID 冷启动能力。

## 运行与交付

随机种子依次为 42、43、44。hidden=128，heads=4，layers=2，dropout=0.2，neighbors=[15,10]，batch=2048，inference_batch=4096，AdamW(lr=0.001, weight_decay=0.0001)，最多30轮，验证 NDCG@10 连续5轮无改进早停。

服务器根目录：EXPERIMENT_ROOT。

- 独立代码：code-pure-hgt；启动器：run_pure_hgt_all.sh（防重复锁）。
- 结果：runs-pure-hgt/seed-{42,43,44}。
- 日志：logs-pure-hgt，含 runner.log、runner.stderr 和每种子日志。
- 每种子保存 run_config.json、best_model.pt、last_checkpoint.pt、best_hidden.npy、ranking_metrics.json、experiment_record.json。最后两项仅在种子正式完成后生成。
- 汇总：pure_hgt_results_summary.json 和 PURE_HGT_RESULTS.md；仅在全部种子完成且模型/结果哈希通过验证后生成。标准差为样本标准差（ddof=1）。
- 本地交付目录：EXPERIMENT_ROOT\server_bundle\pure-hgt-5domain。训练产物完成后同步并核验，不将部分下载标记为完整。

已通过本地及服务器的合成单元测试：文本占位输入完全不影响输出，前向与梯度有限，节点 ID 和 HGT 参数正常反向传播。合成测试不产生实验指标。
