# 当前输入输出框架

2026-10-05用户授权5000原图扩充、仅8帧CNN，实际14轮早停；同1000控制笔迹的联合81.4964%→97.3929%，冻结原验证82.1429%→97.5714%，见[报告](../reports/cnn_5000_20261005.md)。4000拟合/1000控制、原v6渲染与根11保持，训练更新量同时增加，非纯ID等预算消融。主SNN与其余ANN未训练，当前本轮进程已退出。

2026-10-05[v6三个ANN对照入口](../tools/train_ann_baselines_v6.py)已独立接通；架构复用此前获授权MLP/8帧CNN/CNN＋GRU，默认累计25轮、控制损失动态降率及早停、根11从头训练。详见[执行登记](ann_baselines_v6_execution.md)。用户本人执行，本轮无模型前向或训练。此对照不改变主研究LIF边界，历史v5入口及检查点保持。

2026-10-05用户授权[调整当前数据为v6](data_v6_execution.md)：保持28×28×8原灰度，放大原MNIST前景、平移8–10像素、净旋转60–90°、初始角±15°并允许最多15%灰度笔画出画。新数据独立生成，原v5与以下全部历史模型/入口保持；本轮只生成和展示，未来训练须明确接入v6，不默认续训旧入口。

2026-10-05用户另行授权[同数据非SNN对照](ann_comparison_execution.md)，模型见[ann_controls.py](../src/braingrow/network/ann_controls.py)。对照可使用卷积、池化、ReLU和GRU；这一例外不替换下文主研究的LIF框架。

框架实现见[src/braingrow/network/growing.py](../src/braingrow/network/growing.py)。保留仿真、结构编辑、深拷贝、检查点保存/恢复与计数。2026-10-04用户启动后新增[A事件学习器](../src/braingrow/network/discrete.py)与[B群落发育](../src/braingrow/network/communities.py)，已完成有限小诊断，均未建立有效功能；它们是可比较机制实现，不是有效默认策略。实施与实际证据见[A登记](discrete_growth_execution.md)、[B登记](community_growth_execution.md)及[诊断报告](../reports/growth_diagnostics_20261004.md)。

最新另有[A2事件学习](../src/braingrow/network/discrete_v2.py)与[B2保留修正](../src/braingrow/network/communities_v2.py)，固定预算忠实复核完成，仍无继续/毕业信号；不是已有效默认策略。见[登记](growth_revision_execution.md)与[实际报告](../reports/growth_v2_diagnostics_20261004.md)。

后续新增[两层梯度LIF实现](../src/braingrow/network/layered_gradient.py)，完成GPU固定图、随机出生/活跃拆分与容量诊断，见[实际节点报告](../reports/layered_neuron_effect_20261004.md)。旧入口的两层实现保留。用户最新要求实施[5000/3000/2000固定基线](large_lif_baseline_plan.md)，已新增[任意层数固定LIF图](../src/braingrow/network/fixed_layered.py)及[专用训练入口](../tools/train_large_baseline.py)，新入口目标固定为三层，不运行生长/剪减；运行与实际诊断见[执行登记](large_lif_baseline_execution.md)。完整训练由用户在PowerShell启动，尚未记录训练准确率。

2026-10-05更新：大容量固定图已由用户运行15遍后暂停，控制数字/运动/联合29.732%/69.732%/21.429%。新增[控制反馈学习率接续](adaptive_learning_rate_execution.md)，用户授权代理修复后继续启动GPU训练，模型动力学与连接保持原基线。

2026-10-05基线停止记录：完成75遍并包含第76遍512视频，最新控制数字/运动/联合36.250%/71.607%/27.411%，最新第70遍完整拟合三项100%。模型5000/3000/2000保持固定，见[训练报告](../reports/large_lif_baseline_training_20261005.md)。随后用户授权自主处理泛化，按[新登记](generalization_tuning_execution.md)实现同图/同第20遍权重的固定轨迹、在线刷新、刷新加标签平滑配对实验。前向模型保持，平滑仅影响训练损失；实际16视频GPU梯度诊断已通过、新实验已启动，见[证据](../reports/lif_generalization_startup_20261005.json)，尚无新完整控制成绩，不能称泛化已改善。

## 输入与神经时间

2026-10-05泛化调优已完成：同起点固定轨迹/刷新/刷新加平滑6轮比较，选中平滑组扩展至12轮，见[报告](../reports/lif_generalization_tuning_20261005.md)。固定图与LIF动力学保持，全部边权训练；控制联合36.161%，冻结探索性验证33.643%。原固定拟合76.853%与实际当轮动态拟合47.790%分别披露；增强后仍有欠拟合信号。候选模型和完整学习状态留在lif_generalization_v1/refresh_smooth，当前无训练进程。

每视频8帧，每帧28×28灰度图展平为784维float32、范围[0,1]。灰度直接成为输入电流，经可编辑输入边进入隐藏LIF；不提供卷积特征、姿态、运动标签或原始ID。每帧重复4个仿真步，每步5ms，共32步/160ms神经仿真时间；不代表视频实际播放时长。每视频隐藏电位、上一步脉冲及输出电位均从0开始，视频之间没有状态延续。

## 隐藏与输出

隐藏电位按泄漏系数和输入电流更新，递归边只传递上一仿真步实际Fire脉冲。越过阈值后发放并归零。输出固定17个电位累积单元，只积分实际隐藏脉冲经输出边的加权信号，序列内没有泄漏、发放或复位；末态前10维、后7维分别argmax，相同电位取组内第一个类别。

`GrowingNetwork.create(seed=11,n=24)`仅是沿用的随机初始化诊断入口，不代表已学结构或推荐新学习策略。类名中的Growing表示具有增删节点等操作，不表示存在自动学习。`split_node`是复制动力学并平分出边的结构原语，保留供验证，不要求新策略采用它。

## 计数与限制

隐藏LIF为n，输出LIF为0，输出累积单元为17，总计算单元为n+17。分别统计输入边、隐藏有向图边及密度、输出边和有效权重总数；不存在分时读出权重。报告实际每帧/整视频Fire数与每LIF每步发放率，输出发放为0。

权重仍按稠密数组执行，掩码决定有效连接；边数与Fire数不是实际能耗。默认节点和边上限为None；模型数组512MiB保护是工程限制，不包含数据、中间批次和整个进程峰值。未实现稀疏执行和连续流策略。

评价见[evaluation/metrics.py](../src/braingrow/evaluation/metrics.py)，只依据直接输出电位统计数字、运动、联合准确率及混淆矩阵。交叉熵是诊断指标，不是梯度训练器。

`forward(record_steps=True)`新增兼容诊断接口，提供逐步真实Fire、复位前膜值与每帧计数；记录不参与前向动力学，也不传播膜值。资格只用于本单元训练管理；标签/ID/议程不进入`forward`。正常推断只加载实际网络，末态电位可按整视频真实Fire计数乘实际输出边复核（float32积分允许明确浮点误差）。
