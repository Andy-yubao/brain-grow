# H1-L0逐边可训练门控：执行登记

2026-10-08，主代理已认可[完整固定提案](h1_l0_gates_v1_proposal.md)，正式实施授权覆盖实现、唯一小CPU检查、提交、单GPU训练与报告。当前本文是训练前登记，不表示研究有效；完成结果另写报告及状态，不修改被锁定的本执行文件。

配置：[h1_l0_gates_v1.json](../configs/h1_l0_gates_v1.json)。源：[门控](../src/braingrow/l0_gates.py)、[独立runner](../src/braingrow/l0_experiment.py)、[训练入口](../tools/train_l0_gates.py)、[报告入口](../tools/report_l0_gates.py)。旧model/codec/训练源码不改，已有模型/无损编码/事务/GPU启动证据复用。新增仅[单项真实CPU检查](../tests/test_l0_gates.py)，门梯度、确定性折入等价及完整门值置乱；启动前执行一次，不跑额外GPU检查或恢复压力重放。

唯一根11、数据20261004，从未剪枝FC25同state开始，784→512→256→10；3200fit/800select，旧1000control在四最终臂冻结且接受判断完成后各重复探索一次。官方测试和剩余封存ID不访问。

固定三实际前缀fixed/learned/gate_no_l0各20轮；learned、gate_no_l0每边门变量随CE反传，learned另加λ=min(epoch/5,1)×.001乘sum(p_open)/3200。β2/3、γ-.1、ζ1.1、a初值2.5；a学习率.02、θ学习率.00025第16轮减半，AdamW eps1e-8、betas(.9,.999)、wd0。每层每批一次E形随机门，均匀eps1e-6，两门臂共用根11派生独立门流；不退火或扫描。

learned20轮母态分叉theta×z_det和theta×逐层完整置乱z_det（含零），随机臂前20轮成本共享，不重复执行或计账。另fixed及gate_no_l0共四最终臂，各5轮/250更新，同batch、Adamreset、θlr.000125。冻结不再门控、不额外阈值硬剪、不恢复边；原宽度/槽稠密gather执行保留，真实活跃/非零边、独立参数和文件字节分别报告，不能宣称物理节点缩小或加速。

唯一末态接受：learned相对fixed25选择准确率下降≤1pp、CE增加≤.05、fit≥90%、codec_v1实际文件严格变小；20%是另报阶段规模信号。失败25轮候选预算/成绩照报，完整20轮母態恢复仅作继续入口，不替代公平25轮比较；没有fallback补训练、选最佳轮、改λ或追加种子。随机不依据选择成绩接受或更换身份。旧control不参与任何接受。

实际新增总4000update/256000优化前向和反传样本；前缀3000、分支1000。每轮选择48000+16000，随机冻结边界800、最终fit12800、探索4000，评价上限81600。fold/编码不forward，报告只读预测与文件；如新增真实forward必须事前登记额外成本。本轮没有新增激活拟合或候选标签搜索。

工程1800秒跨会话累计，完整轮持久化预算；非有限值、数据权限或来源不符立即保存停止，不换参数补救。checkpoint包含phase/arm/下一轮、权重/组映射/梯度、Adam、日程、全RNG/独立门流、行记录及账本；完整轮resume不会重跑已完成轮，已保存fit/control预测幂等补记账本。硬中断未落盘的中间计算成本无法完整追溯须披露。

每个新epoch开始前与完成后均检查当前累计预算，不仅检查启动时的past值；完整轮边界才停止，单个已开始轮可能跨过墙钟阈值，其实际成本仍记录，不称毫秒级硬实时截止。p_open使用理想未截断Uniform(0,1)解析期望；端点保护采样后它作为近似统计/惩罚，不冒称截断采样的精确期望。

启动前仅提交归属文件，保存实际提交/所有源配置哈希于training_commit.json，再启动；runner比对源、父checkpoint、原GPU启动和配置。训练开始后这些锁定文件不改。只读报告复核真实预测、门值多重集、掩码/文件、更新和评价账本，不加载模型/optimizer；图标题轴名中文。

```powershell
.venv/Scripts/python.exe -B -m pytest tests/test_l0_gates.py -q
.venv/Scripts/python.exe -B -m tools.check_workspace
.venv/Scripts/python.exe -u -B -m tools.train_l0_gates
.venv/Scripts/python.exe -B -m tools.report_l0_gates
```

完整轮恢复：同一源码/配置/设备下使用训练入口加--resume；完成summary后直接退出，无新增训练和模型评价。运行文件在experiments/runs/h1_l0_gates_v1/mnist_static，真实模型/预测/登记/门值证据与汇总报告显式入Git，完整checkpoint/逐轮日志留本地。

既有L0机制的逐边应用，不称首创；随机差异是门位置及衰减幅度对应关系的联合身份信号。λ0门仍有任务梯度，控制显式L0项，不能单独分离所有随机噪声作用。开放初值、Adam eps和门梯度尺度是固定方案风险；无删边、概率与确定性稀疏有差距或不能保持任务均如实报告，不人为追加硬剪。
