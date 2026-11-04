# Clover 官方 Q8 GGUF 与块去噪运行时实测

> **R0 历史报告，保留原始测量。** 当前入口为 [性能记录](clover-performance.md)，
> review 修复、冻结 ABAB 复测、长上下文及成本分析见 [R1](clover-runtime-review.md)。
> 下文 runtime-before 的 causal 对照存在明显波动，旧整体加速范围不能全部归因于代码修复。
> R1 的旧源码交替复测未重现原先低值；不覆盖这批原始证据，也不把 R0 llama.cpp 数字称为最终源码的重测。
> “按实际行数捕获”等描述对应 R0；R1 已改为固定物理宽度与有效行屏蔽。

本报告针对 Windows、RTX 5070 Ti 16 GB、Ryzen 7 9700X、64 GB 内存、单请求 TP1。
Q8 分支从 main `f417397` 创建；NVFP4 分支从 `2fd00ce` 追加提交。
完整测量、输入、源码版本、输出、缓存与内存记录见 [evidence](evidence/clover-runtime/sources.json)。

性能T1～T4通过：Q8 causal相对较快的stock基线为1.38～1.54倍；W16去噪额外取
native热/冷较慢、stock跨宽度/验收规则最快，仍为1.24～1.64倍。
**正确性不是全项通过：HF去噪进度4/15未满足容差，NVFP4有2个W32输出差异未完全隔离。**
本报告同时保留质量失败与回归跳过项，不以速度收益代替正确性结论。
实测Python3.12.10、Torch2.11.0+cu130、Triton Windows3.6.0.post26；完整依赖与原生扩展摘要在provenance中。

## 权重、基线与计量

Clover GGUF 只使用 `Diffuspace/Clover-35B-A3B-Preview-GGUF` revision
`32e22a02b885e173177b7ce7661ad13448b4f89e` 的
`Clover-35B-A3B-Preview-Q8_0.gguf`，37,863,343,264 字节，完整 SHA256 为
`834529e7aae000305ae94656782726bcf0ed77976f95a56383447d8903afddab`。
文件单独放在 `models/inputs/clover-gguf-diffuspace/<revision>/`，没有使用社区 Clover 量化权重。
官方 llama 能运行 Q8，所以没有降级到 Q4_K_M 或 Q4_0。

NVFP4的前后控制使用此前从官方HF revision `4b0402b6` 转换的同一个本地
`clover-nvfp4/4b0402b6-routed-v1` bundle：专家NVFP4 W4A16，其他参数BF16。
其五提示矩阵统一4096槽、memory_ratio0.85，另附5120槽causal容量审计。
这组N项数据比较相同NVFP4权重的运行时修复前后，不把它当成与官方Q8同精度的跨框架比较。

llama 基线固定为 `Maphist0/llama.cpp-pub@cb09d4355d8bf968bb907c5de03cf42d8aa073ce`。
源码工作树保持干净；VS2022、CUDA toolkit13.2、Release、sm_120，CUDA开启。
构建按模型卡设置，并额外构建 server 以测试驻留热缓存。没有修改算法。
模型卡 PR 中后来的 AR-only 版本不是本报告的 draft-self 基线。

两边均关闭思考，短提示 token 数分别为21/23/36/48/36，完整 token ID 数组逐项验证相同。
FreeToken 使用2560个 Q8专家缓存槽；每槽3,342,336字节，共8160MiB。
llama 使用 `--cpu-moe --moe-cache-mib 8160 -ngl 99 -c 4096 -ctk bf16 -ctv bf16 --flash-attn on -t 8 -tb 8`。
它的日志实际给出2559个可用槽；额外槽表开销与FreeToken不同，共同约束是8160MiB预算。
P0关闭惩罚用于跨框架比较；P1保留模型默认惩罚用于内部模式矩阵。
每个性能配置2次预热、5次测量，报告中位数；EOS后停止，吞吐按实际可见输出token数除以端到端时间。
不把每块宽度当作每次真正生成的token数，也不把同一流式突发内的日志间隔当作decode速度。

严格 draft-self 使用驻留 llama-server，显式 `cache_prompt=false` 并检查 `cache_n=0`。
entropy0.9/0.6只由 stock CLI暴露，每次请求启动新进程，单独标为进程冷缓存。
`-f`文件补一个结尾换行，抵消CLI去掉一个LF的行为；保留内置含EOS计数和原始输出日志。
最终跨框架主表统一使用较快的 `--load-mode none`，causal、严格selfspec、entropy、预览和长提示都重新测量。
最初的mmap结果保留为A/B：本机普通加载大幅改善吞吐，所以原先约2.4～2.8倍的causal相对值不作为最终结论。
两种加载方式的算法源码和二进制完全相同。CLI计时从prompt echo结束到生成结束，排除加载/进程启动；
冷缓存指新进程中的专家缓存，不能把它解释为将35GiB模型加载耗时算入生成速度。
另外对FreeToken逐请求显式清空GPU专家缓存，仍保留2次预热与5次测量，用于threshold/entropy主比较；
清空与同步开销保守计入native计时。W16去噪也测这个配置，T3同时查看热/冷两侧并给更保守的择慢比值。

## 三种解码的含义

`causal`逐token生成。`selfspec`进行一次masked-block预览，再因果验证，提交接受的前缀。
W8/W16分别对应stock的 `--spec-draft-n-max 7/15`；threshold按约定对比entropy0.9/0.6。
stock entropy的原始参数为 `--spec-accept-ratio 0.9 --spec-accept-delta 0.6`，不是降低采样temperature。
“严格”描述逐位置验收规则；批量验证与单token前向存在BF16/量化归约差异，不能据此保证输出与独立causal逐ID相同。
原始矩阵的 `selfspec_exact_match` 保留实际结果，不能把这个字段的false隐去。
`denoise`按官方HF的LowConfidence语义反复预览，固定置信度严格大于0.95的位置；没有位置达标时固定最高置信度位置，整块完成后提交。
本次锁定的llama版本有块预览/GDN/coreset和draft-self，没有完整多轮LowConfidence控制器。
T3因此比较的是FreeToken去噪与llama draft-self两种策略，不是同一去噪算法的直接速度比。

## 实现与设计文档的实测修订

官方文件的853个张量经过重新核对。专家gate/up/down、注意力和GDN投影、共享专家保留Q8打包字节；
BF16的output、embedding和ssm_alpha/beta保留相应路径。
适配器恢复V-head顺序、原始A_log和HF的zero-centered norm约定。
专家格式和行字节数按层读取；混合Q4_K/Q6_K/Q8_0/Q4_0格式用独立数值测试验证，未声称跑过完整官方Q4模型。
Q8 dense在单行使用MMVQ，多行使用MMQ；MoE单行MMVQ，多行按专家分组MMQ。
实测packed dense为1,492,910,080字节，其他BF16参数缓冲为2,088,008,576字节。

专家准入、LRU、缺失专家的mapped-host拷贝沿用原有GPU路径，放在CUDA graph内；不把CPU专家准入夹在逐层graph之间。
coreset排序/topk可在GPU graph内执行，真实冷缓存重放通过。
预览、验证、提交按实际行数捕获，块内复用静态位置、页表和注意力元数据。
第一次遇到某个宽度仍须warmup与capture；端到端测量在两次预热后进行，单次预览不包含首次捕获。
私有clean GDN状态与事务状态隔离，去噪保留下一块需要的pending状态，合并独立flush。
greedy惩罚、mask、argmax和softmax置信度融合；每轮去噪只回读一次计数结果。
五项采样运算由一个pick kernel完成，固定位置/计数另用一个小resolve kernel。
graph覆盖模型前向及GPU专家卸载，包括提交时的causal replay；控制器、一次计数回读及最终选中状态行的拷贝仍在graph外。
接受长度由控制器决定，保留图外的小型提交拷贝以维持事务语义，其开销计入端到端吞吐；没有把整个生成循环声称为单个graph。

实测揭示了设计文档未列出的两个问题：

- Windows的varlen卷积fallback每个GDN层调用 `seq_lens.max().item()`。Clover验证现在传已知行数，移除隐藏的逐层同步；其他模型保持原默认行为。
- 借用的GGUF MMQ kernel把大于255的专家编号跳过，而FreeToken传入的是全局缓存槽号；专家偏移乘法还会在2GiB以上溢出。
  修复槽号假设、64位地址计算和padding边界。独立旧源码复现失败；修复后包含槽256/512及2GiB偏移的83项kernel测试通过。
  早期产生重复Count或NaN的Q8运行明确排除，不计作速度收益。

31.875GiB专家锁页主存申请实测成功。专家载荷用readinto流式拷进银行，避免把整个专家mmap页再驻留一份；
小型文件释放测试和真实进程工作集同时留证。保留W32事务窗口可避免短请求反复捕获；
长prefill会释放它。这与设计的“立即释放”假设不同，采用统一2560槽预算并报告实际显存峰值。

后端无关运行时提交 `69acf2f`、`25fb44b` 已分别追加到NVFP4为 `40e399f`、`40f96c0`。
NVFP4 fused MoE CUDA源码没有修改。GGUF的MMQ槽号修复仅在Q8分支。

执行顺序按本机条件调整：等待官方Q8下载时先在已有NVFP4上准备共享运行时与控制实验；
随后重新校验官方权重和已有固定提交sm_120构建，完成官方基线。不是本次重新编译了一个算法修改版。
发现Windows加载方式影响后，先补齐更快的none基线，再做最终比较。
HF对拍最初将dense展开BF16会额外损失Q8精度，因此改为独立精确反量化和Q8_1数学参考；
这个修订只改变参考工具，没有为提高一致率更改服务模型的权重或生成文本。

## 性能、正确性、质量与回归结果

除单独注明的ms、秒和字节外，吞吐单位均为tok/s。主性能表使用未加NVML采样的运行；
资源诊断与单次预览实验另列计量范围。

### T1：causal，P0

| 提示 | FreeToken | llama none | llama mmap | 相对none |
|---|---|---|---|---|
| Count | 58.40 | 42.17 | 24.15 | 1.38x |
| Sky | 61.06 | 42.36 | 22.26 | 1.44x |
| Python | 72.99 | 48.95 | 25.87 | 1.49x |
| JSON | 63.62 | 41.40 | 24.81 | 1.54x |
| Chinese | 83.82 | 56.73 | 33.33 | 1.48x |

### T2：selfspec，P0

每格为FreeToken / stock tok/s。严格两边均驻留；threshold对entropy时FreeToken每次显式清空专家缓存，stock为新CLI进程。Native清空和同步开销保守计入时间。原驻留threshold矩阵另保留。

| 提示 | 严格W8 | 严格W16 | threshold/entropy W8 | threshold/entropy W16 |
|---|---|---|---|---|
| Count | 65.42 / 35.24 | 62.35 / 34.56 | 56.29 / 27.85 | 54.83 / 22.53 |
| Sky | 40.63 / 20.59 | 29.88 / 14.75 | 45.71 / 23.03 | 31.13 / 13.78 |
| Python | 78.74 / 42.36 | 70.93 / 33.35 | 77.55 / 37.48 | 75.40 / 31.71 |
| JSON | 59.25 / 29.47 | 50.91 / 21.03 | 58.72 / 27.69 | 53.85 / 18.35 |
| Chinese | 59.59 / 22.72 | 40.68 / 15.54 | 58.82 / 23.34 | 44.26 / 16.08 |

### T3：W16去噪，P0

同时给出同宽度比较和更严格的跨W8/W16择快比较；stock均取严格/entropy较快值。最后一列额外取native热/冷较慢值，避免用热native对冷CLI的收益冒充同缓存范围收益。

| 提示 | FreeToken denoise16 热/冷 | stock最佳W16 | stock最佳任意宽度 | 保守相对任意宽度 |
|---|---|---|---|---|
| Count | 70.83 / 57.70 | 34.56 | 35.24 | 1.64x |
| Sky | 30.71 / 29.26 | 14.75 | 23.03 | 1.27x |
| Python | 52.37 / 52.32 | 33.35 | 42.36 | 1.24x |
| JSON | 44.31 / 43.09 | 21.03 | 29.47 | 1.46x |
| Chinese | 37.95 / 36.72 | 16.08 | 23.34 | 1.57x |

### T4：单次初始块预览，ms

同提示、同anchor、同mask输入；每格FreeToken CUDA event / stock llama_perf decode。

| 提示 | W8 | W16 | FreeToken causal step |
|---|---|---|---|
| Count | 11.49 / 19.35 | 14.59 / 48.79 | 10.80 |
| Sky | 11.54 / 19.64 | 14.72 / 49.74 | 10.32 |
| Python | 11.31 / 19.55 | 14.42 / 48.70 | 12.08 |
| JSON | 11.61 / 19.88 | 14.76 / 48.87 | 11.50 |
| Chinese | 11.50 / 18.66 | 14.56 / 48.27 | 8.88 |

### 1K/4K TTFT，秒

| 提示 | Q8 | NVFP4 | llama none ub4096 | llama none ub256 | llama mmap ub256 |
|---|---|---|---|---|---|
| Long1K | 0.78 | 0.40 | 1.02 | 2.52 | 3.87 |
| Long4K | 2.24 | 1.12 | 2.96 | 8.49 | 13.49 |

主TTFT比较取统一ubatch4096、batch4096；256/1024/4096的完整对照保留：

| stock ubatch | 状态 | 1K TTFT秒 | 4K TTFT秒 |
|---|---|---|---|
| 1024 | complete | 1.02 | 3.24 |
| 4096 | complete | 1.02 | 2.96 |

这些长提示配置不改变短提示T1～T4的已完成输入、缓存与计时协议；不把ubatch256的结果称为stock最佳prefill性能。

### 官方Q8六模式矩阵，P1

| 提示 | causal | selfspec8 | selfspec16 | denoise8 | denoise16 | denoise32 |
|---|---|---|---|---|---|---|
| Count | 56.85 | 63.35 | 60.00 | 58.47 | 70.97 | 56.42 |
| Sky | 59.44 | 42.48 | 33.06 | 40.22 | 36.72 | 24.96 |
| Python | 70.82 | 72.31 | 70.47 | 60.48 | 50.33 | 42.05 |
| JSON | 61.84 | 51.12 | 51.03 | 50.95 | 45.28 | 38.31 |
| Chinese | 80.73 | 49.60 | 36.25 | 45.65 | 38.83 | 27.31 |

### NVFP4六模式矩阵，P1

| 提示 | causal | selfspec8 | selfspec16 | denoise8 | denoise16 | denoise32 |
|---|---|---|---|---|---|---|
| Count | 94.53 | 129.09 | 117.31 | 118.62 | 148.00 | 106.28 |
| Sky | 77.37 | 62.74 | 52.09 | 55.40 | 49.83 | 35.54 |
| Python | 94.43 | 136.53 | 145.53 | 106.58 | 95.32 | 78.43 |
| JSON | 86.19 | 110.11 | 81.50 | 116.20 | 112.95 | 88.83 |
| Chinese | 95.72 | 71.02 | 55.37 | 67.21 | 50.92 | 37.10 |

### Q8运行时修复前 → 后

每格为前 / 后端到端tok/s，P1。Q8前控制保留正确MMQ和packed dense，仅还原旧运行时。

| 提示 | causal | selfspec8 | selfspec16 | denoise8 | denoise16 | denoise32 |
|---|---|---|---|---|---|---|
| Count | 58.07 / 56.85 | 54.01 / 63.35 | 46.35 / 60.00 | 44.75 / 58.47 | 62.01 / 70.97 | 47.48 / 56.42 |
| Sky | 43.05 / 59.44 | 25.28 / 42.48 | 16.56 / 33.06 | 18.18 / 40.22 | 17.82 / 36.72 | 8.07 / 24.96 |
| Python | 36.68 / 70.82 | 46.52 / 72.31 | 54.86 / 70.47 | 38.64 / 60.48 | 39.92 / 50.33 | 31.31 / 42.05 |
| JSON | 55.06 / 61.84 | 34.55 / 51.12 | 41.47 / 51.03 | 32.47 / 50.95 | 33.43 / 45.28 | 27.93 / 38.31 |
| Chinese | 73.55 / 80.73 | 27.84 / 49.60 | 23.47 / 36.25 | 21.20 / 45.65 | 21.11 / 38.83 | 17.06 / 27.31 |

上述25个非causal配置的加速范围：1.14～3.09倍。

### NVFP4修复前 → 后

每格为前 / 后端到端tok/s，P1。Q8前控制保留正确MMQ和packed dense，仅还原旧运行时。

| 提示 | causal | selfspec8 | selfspec16 | denoise8 | denoise16 | denoise32 |
|---|---|---|---|---|---|---|
| Count | 84.73 / 94.53 | 55.34 / 129.09 | 62.76 / 117.31 | 46.22 / 118.62 | 83.76 / 148.00 | 73.08 / 106.28 |
| Sky | 63.79 / 77.37 | 22.36 / 62.74 | 28.30 / 52.09 | 14.19 / 55.40 | 16.69 / 49.83 | 16.49 / 35.54 |
| Python | 86.21 / 94.43 | 49.42 / 136.53 | 72.81 / 145.53 | 32.10 / 106.58 | 39.34 / 95.32 | 30.67 / 78.43 |
| JSON | 78.54 / 86.19 | 39.62 / 110.11 | 33.38 / 81.50 | 57.17 / 116.20 | 64.10 / 112.95 | 61.71 / 88.83 |
| Chinese | 94.30 / 95.72 | 32.60 / 71.02 | 30.28 / 55.37 | 20.52 / 67.21 | 21.20 / 50.92 | 24.28 / 37.10 |

上述25个非causal配置的加速范围：1.44～3.90倍。

### 性能解释

运行时前后控制保留同一MoE后端和权重，非causal配置普遍加速；NVFP4 fused MoE CUDA代码未改，支持瓶颈主要在调用、元数据、同步和块间状态处理的判断。Q8多行分组MMQ和packed dense解决另一组带宽问题，不能把这两类收益混称为单一kernel加速。单次前向、端到端吞吐和可见token进度分别留证。

Q8 P1中Count的D16为70.97 tok/s，高于causal56.85；其余四提示D16仍慢于causal。去噪必须反复预测再提交整块，普通文本每轮确定的位置较少，较快的单次前向并不保证较快的生成。W32计算量更大，本机五提示矩阵中均未带来优于W16的去噪吞吐。

NVFP4每槽1,775,616字节，4096槽载荷6936MiB，覆盖的专家数量更多；Q8每槽3,342,336字节，2560槽8160MiB。两者权重精度、输出与缓存覆盖不同，NVFP4绝对速度不能直接作为同权重的llama对比结论。

### 输出稳定性与NVFP4对照诊断

| 分支 | 模式 | 逐ID一致请求数 | 有差异的提示/宽度 |
|---|---|---|---|
| q8 | causal/selfspec | 75/75 | [] |
| q8 | denoise | 35/75 | [('Chinese', 8), ('Chinese', 16), ('JSON', 8), ('JSON', 32), ('Python', 16), ('Python', 32), ('Sky', 8), ('Sky', 16)] |
| nvfp4 | causal/selfspec | 75/75 | [] |
| nvfp4 | denoise | 40/75 | [('Chinese', 8), ('Chinese', 16), ('Chinese', 32), ('JSON', 32), ('Python', 8), ('Python', 32), ('Sky', 8)] |

| 控制实验 | 恢复旧输出数 | 仍有差异的提示/宽度 |
|---|---|---|
| nvfp4-no-ahead-diagnostic | 13/15 | [('Chinese', 32), ('Python', 32)] |
| nvfp4-old-commit-shape | 0/2 | [('Chinese', 32), ('Python', 32)] |
| nvfp4-old-shape-reference-pick | 0/2 | [('Chinese', 32), ('Python', 32)] |
| nvfp4-old-controls-eager-preview | 0/2 | [('Chinese', 32), ('Python', 32)] |
| nvfp4-old-controls-eager-both | 0/2 | [('Chinese', 32), ('Python', 32)] |

no-ahead恢复逐块单独pending前向；old-commit-shape同时恢复只验证W-1个已提交token；reference-pick进一步恢复Torch惩罚/softmax/argmax。这些是因果隔离实验，不混入速度主表；batch形状和置信度浮点归约会改变后续固定顺序，不承诺去噪跨实现逐ID相同。

恢复独立pending前向使7个变化配置中的5个恢复旧输出，证明这5个与pending合并的数值路径有关。其余Python/Chinese W32在恢复旧提交形状、Torch pick、eager预览和eager验证后仍未完全恢复，不能仅归因于CUDA graph或融合pick。

| 仍有差异的提示 | 首次差异token下标（从0） | 旧/控制输出长度 |
|---|---|---|
| Python | 71 | 197/215 |
| Chinese | 86 | 106/104 |

这两例属于尚未完全隔离的数值复现限制；N的25配置速度收益已验证，W32逐ID复现问题仍明确保留，不能宣称所有输出差异都已定位到单独算子。

NVFP4五提示单次预览：W8 10.85～10.92ms；W16 14.53～14.58ms。此前分支evidence中的eager预览约34ms。

### C：正确性

同权重独立HF强制历史对拍 156/158，98.73%；旧分支为155/158。

参考保留精确F32反量化Q8 dense并独立模拟Q8_1激活；权重解码与NumPy bitwise核对，不使用服务端GGUF CUDA kernels。
两个causal差异均为HF rank2；HF gap均0.125，其中一个native tie，另一个native gap0.25。

| 提示 | W | selfspec平均每轮token相对差异 | ±20% |
|---|---|---|---|
| Count | 8 | 0.00% | True |
| Count | 16 | 0.00% | True |
| Sky | 8 | 17.95% | True |
| Sky | 16 | 15.38% | True |
| Python | 8 | 0.00% | True |
| Python | 16 | 0.00% | True |
| JSON | 8 | -16.84% | True |
| JSON | 16 | 1.89% | True |
| Chinese | 8 | 12.32% | True |
| Chinese | 16 | 10.00% | True |

| 提示 | W | 去噪平均每轮确定位置相对差异 | ±30% |
|---|---|---|---|
| Count | 8 | 0.00% | True |
| Count | 16 | 0.00% | True |
| Count | 32 | 33.33% | False |
| Sky | 8 | 2.78% | True |
| Sky | 16 | 11.76% | True |
| Sky | 32 | 0.00% | True |
| Python | 8 | 3.85% | True |
| Python | 16 | 22.22% | True |
| Python | 32 | 3.70% | True |
| JSON | 8 | -35.71% | False |
| JSON | 16 | -37.04% | False |
| JSON | 32 | -33.33% | False |
| Chinese | 8 | 1.64% | True |
| Chinese | 16 | 1.61% | True |
| Chinese | 32 | 1.61% | True |

进度比较统一64-token上限，保留逐轮轨迹；均值容差不等于逐步文本完全一致。两分支预览KV/GDN哈希不变、冷cache重放及每行验证状态审计通过。

相同前缀与相同mask/已固定位置的HF预览对拍：

| 提示 | W | 预览序号 | argmax一致率 | 平均KL | HF/native >0.95行数 | 跨0.95阈值的行 |
|---|---|---|---|---|---|---|
| Count | 32 | 0 | 84.38% | 0.04 | 6/11 | [2, 8, 9, 10, 11] |
| Count | 32 | 1 | 100.00% | 0.01 | 29/31 | [30, 31] |
| JSON | 8 | 0 | 87.50% | 0.07 | 0/0 | [] |
| JSON | 8 | 1 | 100.00% | 0.02 | 2/2 | [] |
| JSON | 16 | 0 | 93.75% | 0.03 | 0/0 | [] |
| JSON | 16 | 1 | 75.00% | 0.07 | 2/2 | [] |
| JSON | 32 | 0 | 84.38% | 0.05 | 0/0 | [] |
| JSON | 32 | 1 | 78.12% | 0.05 | 1/1 | [] |

该实验固定native输入块，隔离自由生成历史分歧；它不把自由轨迹进度不达标改判为通过。

两边的原始logits统一使用Torch softmax/argmax后，首次预览固定的位置仍不同：

| 提示 | W | HF固定位置 | native固定位置 |
|---|---|---|---|
| Count | 32 | [0, 1, 4, 5, 6, 7] | [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11] |
| JSON | 8 | [3] | [0] |
| JSON | 16 | [5] | [0] |
| JSON | 32 | [3] | [1] |

因此这4个容差失败已在采样前的模型logits中出现，不能仅归因于融合pick。Count W32有5行跨过0.95阈值，最终token相同但预览次数3对4，离散计数使均值差达到33.33%。JSON首次均无位置超过阈值，HF/native选择的最高置信度位置不同，随后排版、内容与自由生成历史分叉。独立HF数学参考与服务端量化、BF16 attention/GDN归约路径不同是数值来源；此实验定位到logits/置信度差异，尚未把剩余误差归因到某个单独算子，C去噪进度仍判未达标。

### R：回归

| 分支 | pytest |
|---|---|
| q8 | 451 passed, 147 skipped, 47 subtests passed in 202.26s (0:03:22) |
| nvfp4 | 343 passed, 147 skipped, 47 subtests passed in 17.11s |

运行范围是regression evidence命令列出的模型、内核、状态池与调度器回归集合，不是整库所有pytest。


147跳过项中89是参数布局不适用，46缺完整Qwen4原始权重，4缺专用HF环境，4需vLLM，3需FlashInfer，1缺Gemma样本。
另有真实Flash-Next MTP开/关、main与两feature分支12个配对案例，token和轮次全部一致；真实Qwen3.5族NVFP4三提示在两feature分支也与main逐ID一致。详细输入及保护文件哈希在regression evidence。

### 质量：greedy causal，前50题

统一1024-token上限，关闭思考，checkpoint默认惩罚；这是小样本检查，不是完整榜单。

| 分支 | GSM8K | HumanEval | 达到token上限 GSM/HE |
|---|---|---|---|
| q8 | 48/50 | 48/50 | 1 / 0 |
| nvfp4 | 49/50 | 49/50 | 0 / 0 |

GSM按最终数值评分；HumanEval使用受限子进程执行题目测试，5秒超时。保留所有输出、失败与评分原因。
Q8五提示P1行为检查中，causal/selfspec Python执行通过，三个去噪宽度失败（shadowed sorted或语法错误）；JSON六模式通过。Python输出带Markdown围栏，执行评分先剥除围栏，不能称作严格遵守only-code格式。

NVFP4 causal与此前同1024-token协议的100题输出逐ID一致：100/100；旧评分与来源SHA保留在quality evidence。

五提示代码/JSON行为检查，256-token协议：

| 配置 | 提示 | causal | selfspec8 | selfspec16 | denoise8 | denoise16 | denoise32 |
|---|---|---|---|---|---|---|---|
| q8 | Python | True | True | True | False | False | False |
| q8 | JSON | True | True | True | True | True | True |
| nvfp4_before | Python | True | True | True | False | False | False |
| nvfp4_before | JSON | True | True | True | False | True | True |
| nvfp4_after | Python | True | True | True | False | False | False |
| nvfp4_after | JSON | True | True | True | False | True | True |

NVFP4的JSON D8失败在修复前后都存在；三个Python去噪宽度的失败同样存在于旧运行时，不能说是本次加速后才出现。

### 显存与PCIe

下列显存为独立资源矩阵的整卡采样峰值，含桌面；不把带采样的吞吐混入主比较。

| 配置 | NVML整卡峰值 GiB | 实际采样间隔中位数 ms |
|---|---|---|
| q8 | 15.71 | 93.81 |
| nvfp4 | 15.76 | 93.93 |
| stock-q8 | 12.44 | 94.09 |
| stock-q8-mmap | 12.36 | 94.35 |

逐请求hit率、逻辑H2D字节、物理收发估算和工作集/private峰值在matrices/resources/sources evidence，单位及采样范围分别标注。
旧统计字段cpu_per_layer在GPU offload下沿用miss计数，不能据此断言专家在CPU计算。

同一资源矩阵进程树的CPU内存采样，GiB：

| 配置 | 峰值工作集RSS | 峰值private | 系统可用最低 |
|---|---|---|---|
| q8 | 35.73 | 20.01 | 18.68 |
| nvfp4 | 19.93 | 18.16 | 34.49 |
| stock-q8 | 34.18 | 46.47 | 20.02 |
| stock-q8-mmap | 30.17 | 13.59 | 23.43 |

五提示资源诊断的缓存命中率与逻辑专家传输范围：

| 配置 | 模式 | hit率范围 | 逻辑H2D GiB/请求范围 |
|---|---|---|---|
| q8 | causal | 70.30–83.81% | 10.49–63.81 |
| q8 | selfspec8 | 61.13–84.44% | 13.21–88.84 |
| q8 | selfspec16 | 52.69–81.71% | 14.65–106.97 |
| q8 | denoise8 | 62.82–89.17% | 14.61–101.40 |
| q8 | denoise16 | 54.07–89.72% | 12.87–131.36 |
| q8 | denoise32 | 39.92–89.09% | 16.43–137.77 |
| nvfp4 | causal | 81.50–100.00% | 0.00–17.87 |
| nvfp4 | selfspec8 | 80.99–92.49% | 3.13–25.88 |
| nvfp4 | selfspec16 | 75.24–92.75% | 4.03–27.61 |
| nvfp4 | denoise8 | 81.39–94.53% | 3.72–33.13 |
| nvfp4 | denoise16 | 77.06–94.77% | 3.31–29.57 |
| nvfp4 | denoise32 | 73.23–95.87% | 4.44–39.23 |

stock的ubatch≤8、ubatch>8命中/上传汇总由未修改的日志给出，包含预热，不能直接冒充与native逐请求同口径hit率；完整backend_receipts随matrices提交。

NVML整卡逐请求PCIe采样积分估算范围，GiB；**不是精确硬件字节计数**：

| 配置 | RX范围 | TX范围 |
|---|---|---|
| q8 | 20.20–246.21 | 2.96–44.48 |
| nvfp4 | 0.68–70.24 | 0.02–13.34 |
| stock-q8 | 10.90–204.28 | 1.33–23.99 |
| stock-q8-mmap | 13.57–200.07 | 1.24–23.71 |

### NVFP4额外容量配置

共同前后比较固定4096槽；额外5120槽、memory_ratio0.95的causal审计状态：complete.

| 提示 | causal tok/s |
|---|---|
| Count | 79.13 |
| Sky | 89.21 |
| Python | 94.50 |
| JSON | 88.67 |
| Chinese | 100.95 |

更大缓存不是所有提示的通用加速配置；例如已充分命中的Count仍可能变慢，主结论使用固定4096槽。

### 验收状态

| 项 | 通过 |
|---|---|
| T1 | True |
| T2 | True |
| T3 同W16 | True |
| T3 另取跨宽度最快 | True |
| T4 | True |
| C selfspec进度 | True |
| C 去噪进度 | False |

**C去噪进度有4项未达标，NVFP4有2个W32输出差异未完全隔离；不能将本轮交付写成所有验收条目通过。** 完整状态见acceptance.json。生成质量失败与输出差异单列，不将速度通过写成所有文本/代码质量通过。

## 原始数据边界

CUDA event测量FreeToken预览，stock预览由外部harness调用未修改的 `common_speculative_draft`，读取 `llama_perf`。
两边使用相同提示、相同首token anchor和W-1个mask；捕获不计入单次预览时间。
外部harness也保留同步wall time，不修改stock库或算法。

缓存hit率、逻辑专家传输字节来自实际路由/缺失计数；逻辑字节不等于物理PCIe流量。
另跑NVML资源矩阵，保留整卡PCIe收发计数积分估算及物理显存峰值，不混入主性能中位数。
NVML计数器覆盖20ms窗口，但本机实际轮询间隔约94ms；积分包含桌面、协议与非专家流量，不能称为精确的逐请求硬件字节计数。
长prefill的整层传输不含在decode/sparse-prefill逻辑字节指标内。
Torch allocated/reserved、NVML整卡显存、进程工作集和private bytes分别列出，不能相加当作物理内存。
工作集/private表为采样的进程树求和，子进程之间共享页可能重复计数；系统可用最低值另列，不把RSS当作独占物理RAM。

1K/4K提示分别精确为1024/4096个token；两边统一8192上下文以容纳4K提示加输出。
两边重算完整prompt；FreeToken基准默认禁止前缀复用，llama检查实际prompt计数。
短提示均不足64token，早期未显式禁用前缀复用的P0矩阵日志同样显示cached-token为0。

原始日志、全部测量和失败运行保留在本机 `runs/T-20261102-clover-q8/`。
提交中的evidence包含每次请求指标、输出字节/token哈希、去重后的完整输出、输入ID、命令与源码回执以及原始文件SHA。
剥除Markdown围栏后的代码执行结果与输出格式遵循情况分别解释；不修改模型生成文本来让检查通过。

## 在本机运行官方Q8

使用已验证的Windows原生扩展、MSVC/CUDA编译环境和任务私有cache；软件版本见provenance。
普通安装的Transformers不能直接据此推断支持Clover GGUF，HF参考是有界专家的数学对拍工具。
使用Q8工作树的Python包和绝对解释器路径。

```python
from freetoken.core import SamplingParams
from freetoken.llm import LLM

model = "C:/workspace/models/inputs/clover-gguf-diffuspace/32e22a02b885e173177b7ce7661ad13448b4f89e/Clover-35B-A3B-Preview-Q8_0.gguf"
llm = LLM(
    model, moe_strategy="offload", moe_cache_size=2560, expert_load="serial",
    max_running_req=1, memory_ratio=0.85,
    num_token_override=4096, max_seq_len_override=4096,
    max_extend_tokens=4096, cuda_graph_bs=[1],
)
try:
    ids = llm.tokenizer.apply_chat_template(
        [{"role": "user", "content": "Count from 1 to 10."}],
        tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=False,
    )
    answer = llm.generate([ids], SamplingParams(
        temperature=0, max_tokens=256, clover_mode="denoise", clover_width=16,
    ))[0]
    print(answer["text"])
finally:
    llm.engine.shutdown()
```

其他模式为 `causal`、`selfspec` W8/16、`denoise` W8/16/32。
块模式本次资格范围为单请求、TP1、greedy、GPU专家卸载、hybrid_radix。
自动缓存规划尚不预留W32事务窗口，使用上述显式预算；显存峰值已接近本机上限。

## 来源与许可证

- HF语义：`Diffuspace/Clover-35B-A3B-Preview@4b0402b65e523d0183a14c431955438714cb16ff`，`modeling_clover_moe.py`、`sampling_clover_moe.py`，Apache-2.0。
- 预览卷积/GDN/coreset：`WilliamZR/sglang@2243dc6edb416073b60be0765904cabae9eb61cf`，文件头给出完整路径，Apache-2.0。
- GGUF布局、解码及stock探针：固定llama来源与MIT许可证在相应文件头及 `LICENSE.clover-gguf-reference` 中记录。
- GPU MMQ/MMVQ沿用vLLM引入的GGML算术；相关头文件记录来源提交、路径和Apache-2.0/MIT来源。
- [NVML计数口径](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)：物理流量为采样积分估算。
