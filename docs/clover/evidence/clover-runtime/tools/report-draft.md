# Clover 官方 Q8 GGUF 与块去噪运行时实测

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

<!-- RESULT_TABLES -->

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
