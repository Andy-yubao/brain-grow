# 扩容FC真实稀疏执行诊断登记

2026-10-08，依据[适应末态](h1_capacity_adaptation_v1_execution.md)主剪枝图保持约97%，但此前只验证文件缩小，训练/推断均dense。此轮只读部署检查与测量，不训练、不改连接、不设权重阈值，不增加数据。保存task16/fixed16及CNN25权重、已有完整位图与预测是唯一输入。

## 假设与转换

task现有10400411活边，占原25391616的40.96%；测试这些任意稀疏连接转成真实CSR后能否维持同任务输出且产生实际延迟/内存收益。直接从BGM每层位图、活FP32、偏置构建原out×in矩阵CSR，FP32值/int32行指针及列索引，不丢弃任何活边（含数值恰零）。同权重dense Linear参照移除训练mask元数据，仅保存含零权重/偏置；另列fixed完整dense与CNN手工参照，结构/容量/权重不同不能当纯执行因果对照。

CSR使用`torch.sparse.mm(Wcsr, x.T)`、偏置、前三层ReLU；dropout评价关闭。稀疏数值/索引及偏置单独计实际存储字节，CSR torch文件实际字节另列；先前位图文件44.8MB不是CSR运行内存。采用[官方 sparse.mm 文档](https://docs.pytorch.org/docs/2.14/generated/torch.sparse.mm.html)列出的CSR×dense前向接口。网站稳定版现指2.14，本机2.13.0+cu130的CPU/CUDA支持与性能必须靠本轮实际验证，不能由文档推定GPU兼容或加速。

## 对照、测量及预算

CPU及CUDA各串行4种独立工作进程：task CSR、同权重task dense、fixed dense、CNN。每种进程仅驻留一种模型；分别报进程初始化工作集、载入后工作集、最终工作集/峰工作集，Windows GetProcessMemoryInfo实测；CPU峰包含PyTorch/载入转换，不能把它当模型数值字节。GPU报模型载入分配、评价/计时阶段峰分配和保留缓存，CUDA上下文外开销不包含在torch分配；各项分开，均不推断能耗。

只读已有验证集缓存select.npz（800×784），校验既有manifest SHA及隔离ID；不加载训练集、探索性测试集或官方测试/封存ID。每进程验证一次/800样本、batch64，保存logits/混淆矩阵/CE/准确率，与对应task16/fixed16/CNN25既有保存预测比较最大绝对误差、最大相对误差、argmax一致性。最大绝对误差≤1e-3且800个argmax完全一致才测延迟；超标保留失败，停止对应格式的速度比较，不改阈值。

端到端计时batch1和64：固定验证集前1/64个原灰度host FP32输入（包含单数字偏置，执行量仍固定），host→目标设备→前向→10类输出回CPU并同步。模型载入/预处理/解码/验证不计入端到端；CPU同设备输入无复制，GPU输入pageable非pinned，输出转换计入。每尺寸10预热+100次逐次计时，报告原始100时序、均值/p50/p95及每样本延迟；单次请求，不循环摊薄、并发或CUDA Graph。总最多1760计时/预热前向、57200样本；验证8×800=6400样本，最多合计63600推断样本。TF32关闭，FP32/CPU线程2，推断inference_mode，无梯度、更新或测试集评价。

每worker120秒、整次20分钟、CUDA峰8GiB；超时/OOM/非有限/格式不支持保存失败，可继续其他已登记设备，不加参数扫描或阈值调节。一次新增小CPU真实CSR↔dense前向与int32索引检查，其余旧导出/数据检查复用；完整模型输出一致性是本轮核心测量，不是重复训练诊断。实现/登记测量前提交；结束保存源/配置/父模型/数据SHA、实际CSR模型、预测/时序、中文报告并提交重要证据，大型部署文件本地保留。

入口项目根`.venv/Scripts/python.exe -m tools.measure_capacity_deployment`，目录`experiments/runs/h1_capacity_deployment_v1/mnist_static`拒绝覆盖。若CSR更慢或更占内存，如实记录；下一轮优先级依据实际瓶颈，不能把边数/文件下降称实际加速。

## 实际完成

测量前实现b79b5d437d3623d714129ebbf5c4e5aa0045fba1；一次新增小CPU CSR/dense/int32检查通过0.071秒，workspace47文档191链接57语法通过。CPU/CUDA各4独立进程均通过对应保存预测一致性，最大绝对误差≤1.15e-5、800argmax全相同，验证97.125%。[实际报告](../reports/h1_capacity_deployment_v1_mnist_static_20261008.md)附完整原始时序/资源JSON与验证预测，报告0模型前向。

CSR GPU batch1/64比同权重dense慢9.77/2.57倍；CPU batch1慢4.61倍、batch64快约18.5%。真实CSR运行张量83.28MB、文件83.29MB；位图文件44.81MB、dense运行数值101.61MB，各格式区别报告。GPU模型/峰分配和CPU工作集实际值见完整JSON，不概括全场景加速/节能。会话129.538秒，0训练/反传，验证6400样本及1760计时/预热调用57200样本，exit0/GPU释放。

用户最新主结构目标改为有效非零边接近CNN；基本97%已达，后续压缩各档不硬守97%、不以独立参数或文件作主要目标。CNN展开有效非零边独立核对433680（排除padding/偏置），当前FC10400411为23.982倍。下一轮10×/5×/2×/1×边数曲线另行登记，此轮不改图/阈值。
