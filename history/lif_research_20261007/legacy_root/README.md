# BrainGrow

研究能够自行调整连接、形成有用结构并淘汰无效连接的LIF网络。上一轮学习策略失败，当前只保留数据、网络输入输出、结构编辑和评价框架；尚未选定新学习方案。

- [当前状态](docs/status.md)、[研究约束](docs/research_goal.md)
- [输入输出框架](docs/current_framework.md)、[数据协议](docs/data_protocol.md)
- [历史与恢复](docs/history.md)、[实验原则](docs/experiment_plan.md)
- [新生长策略设计](docs/growth_strategy_design.md)、[最小诊断计划](docs/growth_strategy_diagnostics.md)（尚未实现或训练）

## 使用

在根目录使用独立项目环境。当前本地`.venv`可用；新环境执行`python -m pip install -e ".[dev]"`，或依照`environment.yml`建立braingrow环境。当前框架依赖NumPy和Pillow，检查使用pytest；不修改其它项目环境。

```powershell
.venv/Scripts/python.exe -B -m tools.prepare_data
.venv/Scripts/python.exe -B -m tools.check_framework
.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider --basetemp=.tmp
.venv/Scripts/python.exe -B -m tools.check_workspace
```

框架检查只让7个训练视频经过未训练网络，不是训练或性能实验。完整数据审计用`python -B -m tools.audit_dynamic_v5`；预览用`python -B -m tools.preview_data`。`python -m braingrow --status`显示状态。

## Git维护

首次历史提交保存清理前完整现场，随后另一次提交保存干净框架。后续每次机制修改和重要实验结论分别提交，提交前检查差异。数据缓存、环境、临时图和运行目录不进入日常提交；可比较实验的配置、结论、成本、来源及必要证据应显式保存。大规模生成物须选择适当归档方式，不默认无限累积进Git。
