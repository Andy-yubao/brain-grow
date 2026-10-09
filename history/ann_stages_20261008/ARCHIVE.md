# ANN 阶段实现与运行归档（2026-10-08）

此处保存固定节点剪边主线之前的已结束 ANN 实验实现、配置、执行登记和完整本地运行目录。它们用于追溯与按需恢复，不是当前入口。对应 2026-10-07 至 2026-10-08 的报告保存在 `docs/reports/`。

归档按原工作区相对路径收纳：`configs/`、`tools/`、`src/braingrow/`、`tests/`、`docs/`、`experiments/runs/`。恢复某阶段前先阅读该阶段登记和根目录 `AGENTS.md`，核对当前用户指令、数据权限与源码/配置哈希；不要直接启动旧训练。

当前保留在工作区的代码是 `train_aligned_edges` 及其共享的 `train_capacity_baseline`、`train_capacity_pruning`、数据加载和报告/审计入口。
