# BrainGrow

研究普通 ANN 的参数与结构联合优化：在任务功能保持的前提下学习更有效、紧凑的连接组织。当前 H1 阶段固定 FC 为 784→3136→1568→128→10，只优化参数和连接，不物理删点。CNN 只作外部参照，核心优化器不使用图像几何或 CNN 分层配额。

固定节点剪边阶段已将真实非零边由 7,577,856 降至 236,808，验证准确率 97.125%。下一轮需登记新机制，再按证据持续推进。完整状态见[状态](docs/status.md)，阶段结果见[报告索引](docs/reports/README.md)。

## 项目入口

- [目标与边界](docs/research_goal.md) · [当前框架](docs/current_framework.md) · [研究计划](docs/research_plan.md)
- [实验原则](docs/experiment_plan.md) · [数据协议](docs/data_protocol.md) · [想法登记](docs/user_ideas.md)
- [当前剪边实现登记](docs/h1_aligned_edges_v1_execution.md)
- [历史归档](history/README.md) · [清理维护流程](docs/workspace_cleanup.md)

在项目根使用现有 `.venv`。工作区轻量检查：

```powershell
.venv/Scripts/python.exe -B -m tools.check_workspace
```
