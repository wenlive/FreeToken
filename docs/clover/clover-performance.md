# Clover 性能与验证记录

适用环境：Windows、RTX 5070 Ti 16 GB、Ryzen 7 9700X、64 GB 主存，单请求 TP1。
这是一份按轮次维护的记录；不同轮次的配置、计时边界和源码版本不能省略后直接比较。

| 轮次 | 内容 | 阅读入口 |
|---|---|---|
| R0，2026-11-02 | 官方 Q8 GGUF、llama.cpp 对照、六模式矩阵、HF/质量与资源验收 | [历史报告](clover-runtime-performance.md)、[原始证据](evidence/clover-runtime/sources.json) |
| R1，2026-11-02 | review 修复、同次开机 ABAB、去噪成本预算、长上下文和容量实验 | [本轮报告](clover-runtime-review.md)、[证据索引](evidence/clover-runtime-review/README.md) |

R0 的跨框架结果仍对应 R0 源码；R1 没有重新运行 llama.cpp，不能把 R1 的新数值代入旧表后称为新跨框架验收。
R1 也没有重跑 GSM8K/HumanEval；R0 前 50 题小样本质量结果仅作为历史记录。
R0 内部 runtime-before 的 causal 对照有明显波动；本轮重新控制后的结果见 R1，不继续沿用旧的整体加速范围。
HF 去噪进度原有 4/15 容差失败、NVFP4 原有两个 W32 输出差异尚未全部定位。
“严格 selfspec”指验收规则；多行 MMQ 与单行 MMVQ 有数值差异，不承诺与 causal 逐 token 相同。

## 如何读数字

| 指标 | 定义与用途 |
|---|---|
| 端到端 tok/s | 可见输出 token / 从请求开始到结束的时间，包含 TTFT，排除模型加载 |
| TTFT | 请求开始到首次可见输出；块模式可能一次发出多个 token |
| 流式 decode tok/s | 首批之后的可见 token / 首次到末次可见输出间隔；不含首批和 TTFT，单批输出时无定义 |
| 单次固定预览 ms | 相同前缀、mask、缓存条件下的 graph replay；排除首次捕获，不能代表全程动态专家流量 |
| 去噪生成轮耗时 | 外层 round 的 CUDA event 区间减去 advance 区间；含轮内主机等待造成的 GPU 空隙，排除轮间调度空隙及 prompt advance |
| 专家逻辑 H2D | 缺失专家数 × 每专家载荷字节；不含总线协议开销，不等于物理 PCIe 计数器 |
| 显存峰值 | 必须注明 PyTorch allocated/reserved 或 NVML 整卡值；三者不能互换 |

性能主表使用 2 次预热、5 次测量，列中位数。ABAB 表另保留每进程中位数、样本范围和两个成对比值。
P0 表示关闭惩罚的跨框架实验；P1 表示使用模型默认惩罚的内部实验。
输出上限 64、128、256 的数据分别标注，不能把 token 数较短、TTFT 占比更高的结果解释成 decode 回退。

## 去噪分析方法

用 HF 的可见 token 数 N、预览次数 P、提交块数 B，和原生运行的平均预览成本 p、整块提交成本 c，计算

`HF 进度条件预算 = N / (P × p + B × c)`。

它可用于回答“假如达到 HF 的进度，以当前实现成本能有多快”。它不依赖 HF 的运行速度，
但依赖当前硬件、上下文、专家路由和缓存分布，因此是条件成本预算，不是普适理论上界。
整块 c 包括因果验证、状态提交和 ahead 状态保存，不能用一次 causal 单步代替。
可见 token 排除 EOS 和被截断的块尾，不能把 W 个已解析位置全部算成输出。

同时计算原生自己的进度预算与实际生成轮吞吐：前者用于检查实现额外成本，
与 HF 条件预算的差异用于检查进度差异。超过 HF 条件预算不等于超越物理上限；
低于它也不必然是 Python 开销。完整公式、边界与逐提示数字见 [R1](clover-runtime-review.md)。

## 使用建议与范围

默认保持 causal。当前小样本中 Count 更适合 denoise16、Python 更适合 selfspec，普通文本和中文通常适合 causal；
这不是通用分类器，本轮没有加入自动模式切换。
本轮块路径仅验证 TP1、单运行请求、GPU expert offload 和 greedy：selfspec 支持 W8/16，denoise 支持 W8/16/32。
denoise 的最终短尾仍预览完整 W，须保证 `prefix + W <= max_seq_len`；它不是任意长度上下文下的无限追加接口。

| 参数 | 默认 | 含义 |
|---|---|---|
| `clover_mode` | `causal` | 请求可指定 `selfspec` 或 `denoise` |
| `clover_width` | `8` | 请求使用的物理宽度；不能超过引擎预留值 |
| `clover_max_width` | `32` | 引擎预留的最大块窗口；`0` 只允许 causal，块请求超过已声明值会拒绝 |
| `clover_split_kv` | `False` | 长上下文实验选项；有整模型数值差异，尚未作为默认路径 |
| `gguf_embedding_host` | `False` | 将 untied input embedding 的精确字节映射到主存，便于自动规划更多专家槽 |

本轮把事务窗口和私有 clean 状态纳入预算，并拒绝超出已声明容量的请求。
固定槽数、`memory_ratio=.98` 的压力对照是受控实验配置；不能据此推断桌面变化时始终有相同显存余量。

下面与本轮生命周期核查使用同一预算配置；模型路径选择本分支支持的官方 Q8 文件或已转换 NVFP4 目录：

```python
from freetoken.llm import LLM
from freetoken.core import SamplingParams

llm = LLM(
    "LOCAL_QUALIFIED_MODEL_PATH",
    moe_strategy="offload", moe_cache_auto=True, kv_reserve_tokens=4096,
    expert_load="serial", max_running_req=1, memory_ratio=.9,
    num_token_override=4096, max_seq_len_override=4096,
    max_extend_tokens=4096, cuda_graph_bs=[1], clover_max_width=32,
)
ids = llm.tokenizer.apply_chat_template(
    [{"role": "user", "content": "Count from 1 to 10."}],
    tokenize=True, add_generation_prompt=True, enable_thinking=False,
    return_dict=False,
)
answer = llm.generate([ids], SamplingParams(
    temperature=0, max_tokens=64, clover_mode="causal", clover_width=8,
))
llm.engine.shutdown()
```

下一轮新增数据应追加独立证据批次，记录父版本、最终源码、原始命令、输入 token、预热/重复数、
计时边界、缓存容量、输出签名、未通过项和排除项。旧证据不覆盖，新结论在本页和对应报告中链接。
