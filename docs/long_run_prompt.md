# 长任务入口

开始新一轮前，先读根目录 `AGENTS.md`、[目标](research_goal.md)、[状态](status.md)、[框架](current_framework.md)、[数据协议](data_protocol.md)、[实验原则](experiment_plan.md)、[研究计划](research_plan.md) 和[想法登记](user_ideas.md)。

当前持续主线为 H1：普通 ANN 参数与固定节点连接优化。保持 FC 784→3136→1568→128→10；不物理缩点，不向优化器输入图像几何或 CNN 配额。阶段目标须保持任务功能。新实验先登记机制、对照、预算、接受和停止规则；运行结束后据证据登记下一轮。

固定模型根 11、数据根 20261004 和原始 ID 隔离。官方测试和剩余封存数据保持关闭。保留成本、恢复点和完整失败证据，不重复已完成运行。近期报告按日期/实验整理在[报告索引](reports/README.md)；已归档实现与运行材料见[历史索引](../history/README.md)。
