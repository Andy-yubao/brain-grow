# 文献依据与方法定位

## 2026-10-08固定节点剪边新轮：原始来源核对

本次定向核对原始摘要，非完整复现或通用性能保证。

- Renda、Frankle、Carbin，2020，[Comparing Rewinding and Fine-tuning in Neural Network Pruning](https://arxiv.org/abs/2003.02389)：比较固定低学习率微调与权重/学习率回退；学习率回退保留末态权重、重新采用早期学习率日程。启发大幅剪边后充分恢复，不把本项目短周期学习率重启称精确复现。
- Evci等，2019/ICML 2020，[Rigging the Lottery: Making All Tickets Winners](https://arxiv.org/abs/1911.11134)：用幅度与间歇梯度改变稀疏拓扑。为后续固定节点动态换边提供机制依据，不由论文结果推断本项目会达标。
- Frantar、Alistarh，2023，[SparseGPT](https://arxiv.org/abs/2301.00774)：研究大规模GPT的局部二阶剪枝重构。训练激活补偿可作为机制候选，LLM中50–60%稀疏结果不能外推当前FC所需93%以上剪边，也不把方法本身称新发明。

检索/核对日期2026-10-07。优先使用本地arXiv MCP已有收藏，再用Paper Search MCP与论文原始页面核对；Paper Search本次组合标题查询返回空结果，并不表示没有相关论文。以下是针对计划的定向核对，不是穷尽检索或最新文献覆盖保证。

本轮读取Weight-Sharing Regularization的正文前段及MNIST相关片段、Towards Learning Convolutions from Scratch相关正文片段。其余核对原始摘要/出版页及已有本地元数据，未将未全文复核的细节写成确定结论。年份采用arXiv首次公开年份，会议年份不同则注明。

| 文献 | 与当前目标的关系 | 解读边界 |
| --- | --- | --- |
| Ingrosso、Goldt，2022，[Data-driven emergence of convolutional structure](https://arxiv.org/abs/2202.00565) | 在受控任务上研究初始FC的局部化和重复感受野 | 数据高阶统计重要；不能外推普通MNIST必然局部化，相似模式不自动等于参数绑定 |
| Neyshabur，2020，[Towards Learning Convolutions from Scratch](https://arxiv.org/abs/2007.13657) | β-LASSO学习稀疏局部连接，讨论描述长度 | 作者指出其实现不直接利用稀疏降低存储/计算；剪枝收益与实际执行收益须分开 |
| Pellegrini、Biroli，2021，[Sifting out the features by pruning](https://arxiv.org/abs/2104.13343) | 迭代剪枝后的FC出现局部感受野 | 是稀疏局部结构证据，不能替代严格共享与在线联合优化验证 |
| Shakerinava等，2023，[Weight-Sharing Regularization](https://arxiv.org/abs/2311.03096) | 成对差值惩罚、近端映射得到精确相等权重；MNIST环面及置乱图像 | 直接覆盖“从FC学习共享”的部分目标，不能把同类正则组合称首创；作者区分任意共享与对称性 |
| Redman等，2024，[On How Iterative Magnitude Pruning Discovers Local Receptive Fields](https://arxiv.org/abs/2412.06545) | 分析剪枝与局部感受野的机制联系 | 支持机制诊断和数据对照，不提供本项目算法有效性保证 |
| Lufkin、Saxe、Grant，2025，[Nonlinear dynamics of localization](https://arxiv.org/abs/2501.17284) | 非线性学习局部化动力学 | 机制研究可启发受控实验，不等于真实数据架构自动发现已解决 |
| Ullrich、Meeds、Welling，2017，[Soft Weight-Sharing for Neural Network Compression](https://arxiv.org/abs/1702.04008) | 软共享结合剪枝/量化与MDL动机 | 存储压缩不是卷积模块发现，组映射也要计入 |
| Louizos、Welling、Kingma，2017/ICLR 2018，[Learning Sparse Neural Networks through L0 Regularization](https://arxiv.org/abs/1712.01312) | 可学习门控与参数共同优化 | 门控为连续路线参考，不自动发现共享关系或CNN |
| Yeh等，2022，[Equivariance Discovery by Learned Parameter-Sharing](https://proceedings.mlr.press/v151/yeh22b.html) | 共享方案作为优化对象，以双层优化探索已研究的离散群等变性族，使用partition distance定量测量 | 其共享参数化与适用等变性族仍有假设，不能称无先验任意架构发现 |
| Zhou、Knowles、Finn，2020/ICLR 2021，[Meta-Learning Symmetries by Reparameterization](https://arxiv.org/abs/2007.02933) | 元学习重参数化可作为后期跨任务候选 | 需要跨任务设计，不适合作首轮低成本机制方案 |
| Evci等，2020，[Rigging the Lottery: Making All Tickets Winners](https://proceedings.mlr.press/v119/evci20a.html) | 动态稀疏拓扑更新，可启发删边后恢复边 | 原研究在给定架构上更新稀疏拓扑；不等于从FC中发现卷积原语 |

## 具体待比较点

Weight-Sharing Regularization已研究任务损失加共享与L1惩罚及近端更新，其MNIST实验并不说明完全恢复CNN；因此本项目必须比较任务敏感分组能否比值域聚类更好地保留功能与形成组织。

软共享及MDL压缩已有先例；本项目描述计数必须把拓扑、组ID和调用映射算进去，而不是只减少数值表。学习共享方案的双层优化也已有先例，第一版启发式候选选择不能冒称比它更一般或等价。

初步可研究的增量：可逆删边/拆组与功能保持的组合；利用梯度冲突而非像素坐标发现共享组；从标量共享走向可编译重复子图；固定核心规则的多任务差异化结构。以上均是候选贡献，是否新颖仍需算法与实现级的进一步比较。

## 2026-10-08：主动路径组稀疏方案定位

本次定向核对[Path-SGD: Path-Normalized Optimization in Deep Neural Networks](https://arxiv.org/abs/1506.02617)原始摘要。Neyshabur、Salakhutdinov、Srebro于2015年讨论不改变网络输出的权重缩放不变几何，以及基于路径正则的近似最速下降。该依据说明路径尺度不变思想已有研究，不把它称为本项目首创；本次未全文复核论文。

本项目新尝试为Adam训练目标加入节点完整路径组惩罚，再作可拒绝的物理删节点，不是复现Path-SGD更新公式，也不直接继承其效果结论。偏置路径、固定初始化归一化、节点组求和、真实文件接受条件和五臂对照是本轮实现选择。只有运行后比较无正则同剪枝机制，才能判断该正则是否促进可剪性。
