# Clover GGUF / NVFP4 / llama.cpp：5070 Ti 实测分析

本报告针对 Windows、RTX 5070 Ti 16 GB、Ryzen 7 9700X、64 GB 内存上的
Clover-35B-A3B-Preview 单请求推理。FreeToken 已能直接加载下载的混合 GGUF，
并运行 causal、自投机和 LowConfidence 块去噪。默认建议是 causal；块宽更大
并不保证更快。以下把算法正确性、输出质量、吞吐和交互延迟分别列出。

## 测试对象与边界

| 项目 | FreeToken NVFP4 | FreeToken GGUF | 官方 llama.cpp Clover PR |
|---|---|---|---|
| 源码 | `feat/clover-nvfp4@2fd00ce` | `feat/clover-gguf`，基于 `main@f417397` | PR31306，`cb09d4355d8bf968bb907c5de03cf42d8aa073ce` |
| 权重 | 本地 BF16 转换，路由专家 NVFP4，其余 BF16 | AtomicChat AD-Q4_K-IQ4_XS，18.75 GiB | 同一个 AtomicChat 文件，无转换 |
| 路由专家 | NVFP4 W4A16 | gate/up IQ4_XS、down Q4_K；激活 Q8_1 | 同一打包权重；GGML 后端自身计算路径 |
| dense | BF16 | 文件 Q8_0，加载时还原成 BF16 | 保留量化格式，GGML 算子执行 |
| 专家卸载 | 主存打包专家 + GPU LRU | 主存打包专家 + GPU LRU | 官方 `--cpu-moe --moe-cache-mib` |
| 模式 | causal、selfspec、denoise | causal、selfspec、denoise | causal、draft-self；无独立 LowConfidence 循环 |

GGUF 文件 SHA256 是
`04ca113c97ce6f225b5846ea9a8bc0c19c0e22f83b8eae40bd6dd967951c3cc4`，
ModelScope 文件 revision 为 `556eebd05adcae7c29e6ce005eea7220f0e19eaa`。
这是 [AtomicChat 社区量化](https://modelscope.cn/models/AtomicChat/Clover-35B-A3B-Preview-GGUF)，
不能标成 Diffuspace 官方 Q4_K_M 的实测。
[官方模型卡](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview-GGUF)
推荐 [llama.cpp PR31306](https://github.com/ggml-org/llama.cpp/pull/31306)。本次固定该 PR
的 head，编译未修改的 `llama-server` 和 `llama-speculative-simple`，不用主线中
尚未包含的模型支持。工具链为 VS2022/MSVC14.44、CUDA13.2、SM120a。

原 NVFP4 分支已经按用户要求推送至
[wenlive/FreeToken](https://github.com/wenlive/FreeToken/tree/feat/clover-nvfp4)。
它确实实现了块去噪，不只有普通 decode 和自投机；入口为
`python/freetoken/models/clover_moe/runtime.py::_denoise`。本次 GGUF 复用该控制器，
另行验证其量化算术和端到端行为。两条分支都没有修改现有 fused NVFP4 MoE 内核。

## 能否在本机运行

可以，但需要专家放主存、工作集缓存到 GPU。20,128,892,352 字节的文件本身已大于
16 GB 显存，不能据“4 bit”推断全 GPU 驻留。FreeToken 的路由专家原始打包数据
为16.25 GiB，运行时不从 SSD 调入专家；dense Q8_0 在加载时展开成 BF16。

默认4096个专家槽合计6656 MiB。全五提示、六模式矩阵的递归进程工作集峰值
37.90 GiB、private bytes 19.19 GiB，系统可用内存最低15.97 GiB。Windows 的
工作集包含文件映射，private bytes 不是进程全部驻留内存；不能混用两个指标。
另一次六模式生命周期的 CUDA allocator 峰值为 allocated 13.66 GiB、reserved
14.71 GiB；这不是整卡显存，也不是与上述主存峰值同时发生的一个样本。
[资源证据](evidence/clover-gguf/resources.json)保留两次实验的来源。

本机安装的 Transformers5.16.1 在只读取 GGUF metadata 时就拒绝 `clover_moe`，
因此不能直接用原样 `AutoModelForCausalLM.from_pretrained(..., gguf_file=...)`。
自行适配的 HF 对拍器能够运行，使用有界专家缓存，GPU allocated 峰值约10.7 GiB；
这是正确性参考程序，不是 stock HF 支持，也不是性能基线。原 BF16 约65 GiB，
不能整体装入64 GB主存；主存/显存混合 BF16 的其他方案没有在本任务中评测。

## 可比较的测试协议

固定五个提示：Count、Sky、写合并区间 Python 函数、生成三本书 JSON、中文说明文。
HF chat template 关闭 thinking；所有框架使用完全相同的 prompt token ID。
最大输出256，实际遇 EOS 停止；上下文4096，单请求、TP=1。每组2次预热、5次测量，
报告中位数；区间和原始输出保存在 JSON。专家缓存保留，KV/prompt cache 不复用。

两套采样协议不能合并成一张“相同输出”排行榜：

- **P1**：模型发布的 presence/repetition、长度和思考结束偏置等默认设置，
  用于 FreeToken NVFP4/GGUF 的六模式矩阵和下游质量测试。
- **P0**：关闭 Clover 惩罚，greedy，用于 FreeToken 与 stock llama.cpp 的接近对齐测试。
  llama.cpp 的服务器会把 prompt 纳入 sampler penalty history，而 FreeToken 的
  Clover 惩罚按生成 history 工作；关闭惩罚避免把这项语义差异算成框架速度差。

“tok/s”均优先采用可见输出 token 数 / 请求总耗时，包含 prefill 和首 token 延迟，
排除模型加载。FreeToken 是离线 scheduler；llama-server 的结果另含本地 HTTP/SSE
和服务器调度。CLI 用生产端日志时间计算生成阶段，属于新进程冷专家缓存，
不能直接冒充热服务器结果。对单个 burst 不计算虚高的稳态 decode 速度。

发现并处理的官方示例细节：

- `-f` 会删掉末尾一个 LF。测试文件补偿一个 LF，并核对逐字回显；服务器额外核验
  tokenizer 的完整 ID 数组。仅比较相同 token 数不足以发现最后一个 token 变化。
- GGUF metadata 自带 repeat1.05，与官方 entropy acceptance 的无约束 sampler
  检查冲突；显式 repeat1/presence0/frequency0 后可运行，没有修改上游源码。
- CLI 内置 `n_predict` 包含 EOS/未打印尾部，Count 曾显示30个 token 却记32。
  报告保留内置计数，同时独立统计可见输出，避免吞吐虚高。
- 服务器可在末包重复所有 token，客户端去重并识别真实 EOG；固定
  `cache_prompt=false`，逐请求断言 `cache_n=0`、prompt长度正确。

## FreeToken：NVFP4 与 GGUF 的六模式结果（P1）

以下每格是 NVFP4 / GGUF 的可见端到端 tok/s，默认均4096专家槽。
权重与激活算术不同，不能解释为相同输出上的纯内核 A/B。

| 提示 | causal | selfspec8 | selfspec16 | denoise8 | denoise16 | denoise32 |
|---|---:|---:|---:|---:|---:|---:|
| Count | 97.7 / 93.8 | 86.3 / 83.1 | 93.7 / 89.0 | 72.1 / 74.8 | 102.0 / 107.5 | 102.9 / 98.8 |
| Sky | 78.9 / 76.7 | 41.8 / 47.8 | 43.7 / 41.4 | 26.9 / 30.6 | 29.3 / 32.7 | 28.4 / 18.4 |
| Python | 95.8 / 92.8 | 85.2 / 81.1 | 111.1 / 98.0 | 57.9 / 58.6 | 65.2 / 66.2 | 48.0 / 48.4 |
| JSON | 87.5 / 83.6 | 69.3 / 61.6 | 60.7 / 61.5 | 66.0 / 64.9 | 81.1 / 72.6 | 75.1 / 63.4 |
| 中文 | 97.5 / 91.6 | 36.7 / 41.8 | 36.5 / 36.1 | 27.5 / 26.8 | 28.9 / 28.1 | 28.4 / 22.3 |

来源：[GGUF矩阵](evidence/clover-gguf/performance.json)、
[NVFP4矩阵](https://github.com/wenlive/FreeToken/blob/feat/clover-nvfp4/docs/evidence/clover-nvfp4/performance-v3.json)。
GGUF causal 接近 NVFP4，同属本机约百 token/s 的档位。GGUF 文件小约2.7 GiB，
主要是存储收益，不自动换来更高吞吐：dense 仍展开 BF16，专家只有约4.04%的
单专家字节节省，且引入 Q8_1 激活量化与不同 dot 路径。

GGUF 的 TTFT 随模式明显变化：

| 提示 | causal ms | selfspec8 ms | selfspec16 ms | denoise8 ms | denoise16 ms | denoise32 ms |
|---|---:|---:|---:|---:|---:|---:|
| Count | 55 | 89 | 86 | 183 | 197 | 303 |
| Sky | 112 | 98 | 98 | 380 | 573 | 1353 |
| Python | 143 | 139 | 141 | 266 | 433 | 981 |
| JSON | 165 | 163 | 163 | 269 | 310 | 536 |
| 中文 | 130 | 124 | 126 | 377 | 638 | 1436 |

Count denoise16 的吞吐比 causal 高约15%，但首字等候多约142 ms。
denoise32 把 Count 几乎一次输出，不能用 burst 内日志间隔推导稳态速度。
中文 denoise32 等1.4秒才开始输出，最终吞吐也更低；增加块宽无助于交互。

## 工作集、PCIe 与打平门槛

GGUF 每专家1,703,936字节，NVFP4 每专家1,775,616字节。GGUF 的4096/4608/5120槽
分别占6656/7488/8320 MiB；相同槽数不等于相同显存预算，更不等于相同命中率。
相邻 token 激活专家的重叠、preview 的跨位置并集，以及缓存淘汰都影响传输。

| GGUF 提示/模式 | cache hit | 专家 H2D payload GB/请求 | 有效并行进度 |
|---|---:|---:|---:|
| Count causal | 100% | 0 | 1 token/步 |
| Count selfspec16 | 77.2% | 3.96 | 10 emitted tokens/轮 |
| Count denoise16 | 77.4% | 3.48 | 10 visible tokens/preview |
| Python causal | 86.7% | 18.85 | 1 token/步 |
| Python selfspec16 | 80.4% | 27.72 | 9 emitted tokens/轮 |
| Python denoise16 | 83.1% | 35.90 | 3.03 visible tokens/preview |
| 中文 causal | 89.5% | 6.05 | 1 token/步 |
| 中文 selfspec16 | 91.4% | 11.73 | 2.88 emitted tokens/轮 |
| 中文 denoise16 | 95.1% | 11.28 | 0.98 visible tokens/preview |

payload 是算法上的专家搬运量，含短提示 sparse prefill，排除总线协议开销；
不是 PCIe 硬件计数器。hit 按每层请求的唯一专家累计，重复 carrier ID 只算一次。
高命中率也可能很慢：中文去噪反复预览，在命中率95%的情况下，有效进度仍接近1。
同一个输出请求搬运量可以超过模型大小，因为专家会淘汰、重新调入。

用整轮 wall time / causal 稳态一步时间估算所需有效 token 数，计算和 dispatch
也包含在内，不能把全部额外耗时归因于 PCIe。selfspec 统计从普通 prefill 后开始，
去噪排除含 prompt 的第一块，后续块包含 pending-token flush 和 causal commit。
[代价模型](evidence/clover-gguf/cost-model.json)提供每提示、每宽度的实测门槛。
Python selfspec16 有机会跨过门槛；JSON、Sky 和中文的有效进度普遍不足。
去噪应比较 visible yield，不能把 EOS 后仍被解析的 block slots 当成用户输出。
整轮门槛高于设计文档引用的 Spark 估计，支持“桌面卡更难打平”的预测方向；
本次没有在 Spark 重跑同权重同协议，不能给出严格的两机性能比，也没有 profiler
证据把额外开销全部归因于 PCIe。直接保留事务前缀已经避免了额外 commit replay，
仍不足以弥补低接受率任务反复执行整模型 preview 的代价。

把未用于 W32 快照的显存给专家缓存，比盲目增加并行宽度更有用：

| 提示 | causal5120 tok/s | selfspec16/4608 tok/s |
|---|---:|---:|
| Count | 93.8 | 109.1 |
| Sky | 95.9 | 42.7 |
| Python | 97.8 | 100.3 |
| JSON | 88.7 | 64.4 |
| 中文 | 99.9 | 36.5 |

[缓存实验](evidence/clover-gguf/cache-profiles.json)中的每次输出均与对应默认模式相同。
GGUF Python 自投机对调优 causal 只领先约2.5%，不足以作普遍建议；NVFP4 对应实验
约有13.7%优势。跨时段百分之几的差异也应谨慎看待。大缓存配置只验证了相应模式，
不能拿来保证 W32、并发或更长上下文也能装下。切换全部模式保留4096/.85配置。

## 官方 llama.cpp：语义和实现差异

`draft-self` 构造 `[anchor, mask, ...]` 做一次非 causal preview，然后用 causal logits
验证；W8/16对应 `--spec-draft-n-max 7/15`，不是8/16。CLI 的官方 entropy
acceptance ratio0.9/delta0.6 会放宽接受标准；严格 greedy 自投机与之分列。
服务器没有暴露这两个 entropy 参数，不能给 server 强加不支持的旗标。

上游预览保存/恢复 target state，拒绝部分草稿时可能从 checkpoint 恢复并重放前缀。
FreeToken 用事务化逐行状态快照，直接保留被接受的前缀，避免额外 commit 前向。
FreeToken 块去噪另有完整 LowConfidence 循环和因果 commit；官方 PR 中没有这个控制器，
因此官方 CLI 的成绩不能填写到“纯块扩散”列。
[上游源码](https://github.com/Maphist0/llama.cpp-pub/blob/cb09d4355d8bf968bb907c5de03cf42d8aa073ce/common/speculative.cpp)
与[CLI验证逻辑](https://github.com/Maphist0/llama.cpp-pub/blob/cb09d4355d8bf968bb907c5de03cf42d8aa073ce/examples/speculative-simple/speculative-simple.cpp)
固定了这一判断的版本。

官方 cache 用 SLRU，默认6656 MiB实际记录4095槽，并非恰好4096；对大于32行的
ubatch 绕过小块 slot-map 路径。Python36/JSON48 token 的 prefill 与 FreeToken
支持到128行的稀疏并集 prefill 并不等价。上游 Clover top-k 是 CPU custom op，
完整排序256个专家并按较小 ID 处理 ties；FreeToken 路由留在 GPU。
这些是源码确认的结构差异，但在没有分阶段 profiler 的情况下，不宣称某一个
就是全部性能差距。cache 日志按<=8行/>8行分桶，生命周期累计包含预热，不能
当成每个测量请求的 payload。
[cache源码](https://github.com/Maphist0/llama.cpp-pub/blob/cb09d4355d8bf968bb907c5de03cf42d8aa073ce/src/llama-moe-cache.cpp)
和[模型图](https://github.com/Maphist0/llama.cpp-pub/blob/cb09d4355d8bf968bb907c5de03cf42d8aa073ce/src/models/clover-moe.cpp)。

## 官方 llama.cpp：完整实测（P0）

每格依次为 causal / 严格selfspec8 / 严格selfspec16 的可见端到端 tok/s。
FreeToken 是离线 scheduler；llama.cpp 是 localhost server，均2预热+5测量。
“普通加载”是 `--load-mode none`；10 GiB 是候选调优配置，并不在所有任务上胜出。

| 提示 | FreeToken GGUF/4096槽 | llama 6656 MiB/mmap | llama 10240 MiB/普通加载 |
|---|---:|---:|---:|
| Count | 82.4 / 89.3 / 96.5 | 64.8 / 44.8 / 41.3 | 62.2 / 55.0 / 41.3 |
| Sky | 79.3 / 47.4 / 49.6 | 46.5 / 23.9 / 16.2 | 59.0 / 27.4 / 16.8 |
| Python | 95.0 / 95.8 / 110.5 | 54.4 / 46.7 / 32.6 | 68.6 / 65.0 / 33.4 |
| JSON | 84.7 / 71.4 / 48.0 | 48.5 / 36.8 / 23.7 | 52.5 / 42.6 / 21.9 |
| Chinese | 96.8 / 36.2 / 32.5 | 58.8 / 23.7 / 14.7 | 68.7 / 25.7 / 13.1 |

同序 TTFT（ms）：

| 提示 | FreeToken | llama 6656/mmap | llama 10240/普通加载 |
|---|---:|---:|---:|
| Count | 94.2 / 87.3 / 84.1 | 97.2 / 199.7 / 205.3 | 122.8 / 155.3 / 137.6 |
| Sky | 110.6 / 96.8 / 97.7 | 227.1 / 263.2 / 242.1 | 131.2 / 145.9 / 138.9 |
| Python | 150.4 / 136.6 / 141.5 | 356.9 / 342.8 / 362.4 | 181.9 / 186.6 / 184.0 |
| JSON | 172.6 / 162.0 / 175.6 | 441.9 / 462.2 / 456.0 | 527.3 / 561.9 / 568.2 |
| Chinese | 134.3 / 124.2 / 124.7 | 273.2 / 290.1 / 292.9 | 132.7 / 147.6 / 167.9 |

causal 的缓存/加载 A/B如下。最后一列是随后复测的 FreeToken P0 causal，
全部25次输出ID与初测对应结果相同。Count跨时段变化19.4%，其余约0.2%-2.2%；
因此短请求上百分之几的差别不应解释成稳定优势。

| 提示 | llama6656/mmap | llama10240/mmap | llama10240/普通加载 | FreeToken复测 |
|---|---:|---:|---:|---:|
| Count | 64.8 | 60.7 | 62.2 | 98.3 |
| Sky | 46.5 | 64.3 | 59.0 | 79.5 |
| Python | 54.4 | 65.7 | 68.6 | 96.2 |
| JSON | 48.5 | 58.5 | 52.5 | 86.5 |
| Chinese | 58.8 | 74.2 | 68.7 | 98.7 |

单独固定6656 MiB缓存时，Python causal由mmap54.4变为普通加载65.9 tok/s，
这是追加完整普通加载矩阵的依据。最终矩阵表明它主要改善Python，JSON等提示
反而受损；不能给整个框架宣称统一21%收益。缓存加大也没有改善所有自投机行。
调参范围有限，报告没有寻找每提示的独立最优点再拼榜。

各框架causal输出长度（可见token）如下；不同文本不构成等质量、等工作量的证明：

| 提示 | FreeToken P0 | llama P0/6656 |
|---|---:|---:|
| Count | 30 | 30 |
| Sky | 35 | 35 |
| Python | 237 | 237 |
| JSON | 87 | 87 |
| Chinese | 118 | 98 |

相同P0设置下，15个“提示×模式”的首个测量输出有11/15与FreeToken逐ID相同。
尤其中文trajectory有差异；dense算术、shape相关舍入和EOG定义仍有区别。
上表描述这些实际运行配置的速度，不宣称相同数值算法的纯后端加速倍数。

官方CLI entropy策略按模型卡 ratio0.9/delta0.6运行；每提示一个新进程、一个样本，
文件页缓存可能已经热，GPU专家缓存是新进程状态。它没有两次请求预热，不与上表
当作同缓存口径。计时包含生成前的prompt处理，排除模型加载：

| 提示 | 可见tokens | E2E tok/s | TTFT ms | 内置计数 |
|---|---:|---:|---:|---:|
| Count | 30 | 9.4 | 1391 | 32 |
| Sky | 46 | 8.6 | 1533 | 48 |
| Python | 255 | 23.5 | 1757 | 256 |
| JSON | 88 | 12.8 | 2160 | 90 |
| Chinese | 125 | 12.0 | 1891 | 127 |

Count的内置decode32/1.878≈17.0 tok/s，换成30个可见token并计入prefill后是9.4。
两个数字回答不同问题，不能选择更大的一个冒充请求吞吐。官方 entropy策略
也不能标成严格greedy等价或纯LowConfidence块去噪。

[完整比较证据](evidence/clover-gguf/official-comparison.json)保留所有配置、区间、
输出计数、cache日志、精确上游commit、构建参数和二进制SHA；
[主存记录](evidence/clover-gguf/comparison-host-resources.json)保留各实验的递归进程峰值。
stock缓存的上传计数含预热且按生命周期累计，无法从它推算每次实测请求的精确PCIe字节。

主存方面，两种框架都留有余量。下面是不同完整实验的峰值，不能相加或当作同时采样：

| 配置 | 进程工作集 GiB | private bytes GiB | 系统最低可用 GiB |
|---|---:|---:|---:|
| FreeToken GGUF P1全部六模式 | 37.90 | 19.19 | 15.97 |
| FreeToken GGUF P0三模式 | 37.70 | 17.04 | 16.20 |
| llama6656/mmap | 16.87 | 11.43 | 37.38 |
| llama10240/mmap | 16.87 | 14.94 | 37.39 |
| llama10240/普通加载 | 18.10 | 31.69 | 36.16 |

Windows private/commit accounting不等于物理驻留，不能把它与工作集相加。FreeToken
把打包专家复制进pinned主存银行，文件映射也会进入工作集；llama.cpp保留量化dense，
模型图记录GPU model buffer约2030.6 MiB，另有KV80 MiB、recurrent62.8 MiB、
compute362 MiB及专家cache。这与FreeToken展开BF16 dense的显存分配不同。
这些源码/分配记录解释了预算结构，不能替代整卡实时峰值：本次没有采集stock每模式
的整卡VRAM峰值，能确定的是全部候选请求实际完成、未发生OOM。


复现实测时使用提交中的客户端，避免手工重新引入换行和计数差异。例如官方 CLI：

```powershell
$python = 'C:/workspace/envs/T-20261031-clover-gguf/runtime/Scripts/python.exe'
$exe = 'C:/workspace/builds/T-20261031-clover-gguf/llama-official-cb09-sm120/bin/llama-speculative-simple.exe'
$model = 'C:/workspace/models/inputs/clover-gguf-atomic/556eebd05adcae7c29e6ce005eea7220f0e19eaa/Clover-35B-A3B-Preview-AD-Q4_K-IQ4_XS.gguf'
& $python benchmarks/bench_clover_llama.py --executable $exe --model $model `
  --prompts-json C:/workspace/tasks/T-20261031-clover-gguf/llama-prompts/manifest.json `
  --frontend cli --modes official:16 --prompts Count --tokens 256 `
  --warmups 0 --repeats 1 --cache-mib 6656 --output C:/workspace/runs/clover-cli-new
```

该客户端调用未修改的官方可执行文件，参数包括
`--spec-type draft-self --spec-draft-n-max 15 --spec-accept-ratio 0.9
--spec-accept-delta 0.6 -ctk bf16 -ctv bf16 --temp 0 --top-k 1 --seed 42
-ngl 99 --cpu-moe --moe-cache-mib 6656 -t 8 -tb 8 --flash-attn on`，以及明确的
惩罚重置和逐请求日志。将 executable 改为 `llama-server.exe`、frontend改为server、
modes改为 `causal,selfspec:8,selfspec:16` 即复现严格热缓存矩阵；客户端只绑定localhost，
测完自动退出自己启动的进程。输出目录必须是新的。

## 正确性与质量不能从 tok/s 推断

同权重 HF causal forced-history 决策155/158一致，三处中文分歧有实际非零 logit gap，
不能一概称为 ties。只读预览的全部 KV/GDN 哈希不变。selfspec10组、denoise15组
均满足设计20%/30%的进度差异门槛，文本分别8/10、7/15完全相同。HF 对拍 horizon
是64或EOS，性能 horizon256；不能声称每个256-token输出都已有完整逐 token gold。
[数值证据](evidence/clover-gguf/correctness.json)保留边界和差异。

causal 在固定首50 GSM8K、首50 HumanEval 上均49/50，无1024-token截断，与此前
BF16/NVFP4 aggregate相同，但 paired结果不同。GGUF 的 HumanEval/32 失败，恢复
了基线的另一题；因此不是“量化零损失”的证据，也不足以得出统计显著优劣。
块去噪必须另外检查生成代码和 JSON；算法进度正常不等于程序通过行为测试。
[质量记录](evidence/clover-gguf/quality.json)保留 prompt一致性和真实失败。
本次合并区间函数的具体行为检查中，GGUF causal 和两种 selfspec 全部通过，
三种 denoise 全部失败，W32含语法错误。GGUF六种模式的 JSON schema 检查均通过；
旧 NVFP4 矩阵中 denoise Python 同样全部失败，denoise8 还生成过无效 JSON。
因此不能把“是代码/JSON”当成适合块扩散的充分条件。stock llama.cpp 严格模式
以及官方 entropy CLI 的本次 Python/JSON 检查通过，五个提示不构成通用质量结论。

403项回归通过、147项跳过、47个subtests通过；另有5项新测量计数测试。
现有 qwen3_5_moe/NVFP4、qwen4_exp 路径均纳入。真实 Flash-Next 模型在 main 和
特性分支上开/关 MTP 的 token 和轮次一致，31个MTP张量、4个保留槽一致。
缺少完整 Qwen4 HF 权重和可选 backend 的测试仍明确记为 skip，不能写成已通过。

## 使用建议和后续优先级

- 日常文本、中文、JSON、短交互：causal。它就是普通逐 token 自回归，无 MTP。
- 可预测代码：可试 selfspec16，但要测试目标任务正确性和实际 cache预算；本 GGUF
  的收益比此前 NVFP4 小。不要仅凭格式“是代码”就默认高接受率。
- Count类模板化高置信输出：可试 denoise16，吞吐与首字延迟之间做取舍。
  W32 在这五个提示上没有普遍优势，尤其不适合中文交互。
- 需要官方直接运行：使用指定 Clover PR 的 stock llama.cpp，并采用主存专家+
  GPU cache；不能照搬足够显存机器上的全 GPU 参数。
- 下阶段先优化目标工作集、路由与缓存传输，再考虑更多块宽、CPU混合执行或更复杂
  调度。本次没有把尚未测量的 CPU/GPU hybrid 路径标为完成优化。

原始日志、token、logits、失败、精确命令和环境位于
`C:/workspace/runs/T-20261031-clover-gguf`；提交中的精简 JSON 以 SHA256绑定这些
不可变实验。跨框架的输出长度、sampler与缓存差异均保留，不把不同质量目标的
最高速度拼成单一排名。
