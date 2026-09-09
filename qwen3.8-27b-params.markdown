# Qwen3.8-27B 参数速查(harness 涉及部分)

> 日期:2026-09-09 ｜ 状态:**服务当前未运行**(上次 2026-09-04 SIGTERM 停止),本文是"起得来之后的参数底账"。
> 用途:harness(阶段 0 起)所有与模型打交道的地方都对着这张表。参数分三类——**模型能力参数**(config 原生,写死)、**引擎参数**(起服务时定,改要重启)、**请求参数**(每次调用可改)。
> 勘误:上一版把 `--context-length` 写成 8192 是**错的**——那是计划初稿值且与 `chunked_prefill_size=8192` 混淆。**9-04 实测跑通的是 `context_length=32768`**;模型能力上限是 262144,详见 §1.3。

---

## 0. 一条命令快速起服务

```bash
# conda env: sglang(python 3.11, 独立于 finetune/vLLM)
nohup /home/dongshuang/miniconda3/envs/sglang/bin/python -m sglang.launch_server \
  --model-path /home/dongshuang/llm/qwen3.8 \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port 8003 \
  --context-length 32768 \
  --mem-fraction-static 0.5 \
  --attention-backend triton \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  > ~/laser_rag/logs/qwen38_sglang.log 2>&1 &
```

冒烟:`curl localhost:8003/v1/models` → `id = "qwen3.8-27b"`。
(此命令 = 9-04 实测跑通的那条,端口 8003、context 32768。)

---

## 1. 模型真实参数(config.json / 官方 README 溯源)

### 1.1 模型身份(你是谁、在哪个栈)

| 项 | 值 | 备注 |
|---|---|---|
| 命名 | **Qwen3.8-27B**(served-name `qwen3.8-27b`) | API 里 `model` 字段填这个 |
| 权重目录 | `/home/dongshuang/llm/qwen3.8` | modelscope 下载,29G,64 个 FP8 shard ≈24.4GB |
| 架构(config) | `Qwen3_5ForConditionalGeneration`(model_type `qwen3_5`) | dense 多模态(image+text+video),非 MoE |
| 注意力 | **混合 GDN**:64 层 = **48 linear_attention + 16 full_attention**,`full_attention_interval=4`(每 4 层 1 个全注意力) | SGLang 走自有 Triton `GDNAttnBackend`;linear 层 KV 状态小 → 长上下文主要占 full 层 |
| 多模态 | 视觉 `vision_config`:depth 27 / hidden 1152 / patch 16 | 支持 `image_url` / video 传图 |
| 精度 | 权重 **FP8**(quant_method `fp8`,fmt `e4m3`,activation `dynamic`);882 个模块豁免(multimodal 投影/qkv 等) | 8-21 vLLM 曾 auto-fp8 成功加载 |
| 部署机 | DGX Spark(GB10,aarch64,sm_121a),统一内存 128GB | 与 laser_qa 的 vLLM 35B(:8002)同机不同栈 |

### 1.2 文本骨干(text_config)完整参数

| 参数 | 值 |
|---|---|
| `hidden_size` | 5120 |
| `num_hidden_layers` | 64(48 linear + 16 full,见上) |
| `num_attention_heads` / `head_dim` | 24 / 256 |
| `num_key_value_heads`(full 层 GQA) | 4 |
| linear 层 | `linear_num_key_heads=16` / `linear_num_value_heads=48`,KV head dim 128,`linear_conv_kernel_dim=4` |
| `intermediate_size` | 17408 |
| `vocab_size` | 248320(padded) |
| **`max_position_embeddings`** | **262144**(= tokenizer `model_max_length`;原生上下文) |
| `rope_parameters` | `rope_type=default`,theta 1e7,`mrope_interleaved`,mrope_section [11,11,10],partial_rotary 0.25 |
| 激活 | silu;`output_gate_type=swish`;`attn_output_gate=True` |
| MTP | `mtp_num_hidden_layers=1`(多 token 预测) |
| 特殊 token | `bos/eos/pad=248044`,真 eos 另有 `248046`;`image_token_id=248056`、video `248057`、vision start/end `248053/248054` |

### 1.3 上下文长度 —— 三层概念(别再搞混)

| 层 | 值 | 谁定的 | 能不能改 |
|---|---|---|---|
| **① 模型原生能力** | **262144**(config `max_position_embeddings`、tokenizer `model_max_length`,官方 README 同) | 模型/config,写死 | 原生不可超;超了要 RoPE 缩放 |
| **② SGLang 服务开窗** | **32768**(9-04 实测跑通值) | 启动参数 `--context-length` | **可以改**,改完重启;受显存/内存约束 |
| **③ 可扩展上限** | **1,000,000** | 官方声明,需 **YaRN** RoPE 缩放 + 引擎 override | 见 §1.4 命令 |

官方 README 原文:`Context Length: 262,144 natively and extensible up to 1,000,000 tokens.`

> **对 harness 的意义**:真正的"上下文预算"是 ②(当前 32768),不是 8192,也不是 262144。多轮工具循环 + thinking 历史 + 工具结果要挤在 32K 内——Context 工程(CP 摘要/截断)仍是刚需,但余量比我上一版说的宽松 4 倍。**thinking 默认全保留**(preserve_thinking),会显著吃②,harness 要留意。

### 1.4 想开更长上下文(≥262144 或上 1M)的启动方式

官方 README(SGLang 段落)给的方法——**只在确实需要长上下文时才用**,平时别开:

```bash
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 /home/dongshuang/miniconda3/envs/sglang/bin/python -m sglang.launch_server \
  --model-path /home/dongshuang/llm/qwen3.8 \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port 8003 \
  --context-length 1000000 \
  --mem-fraction-static 0.5 \
  --attention-backend triton \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --json-model-override-args '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
```

- 提示词 key 是 `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`(vLLM 对应 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`)。
- **注意**:所有主流引擎的 YaRN 是**静态缩放**(factor 恒定不随输入长度变)→ 开长窗会损害短文本性能。官方建议 `factor` 按你的典型场景设:若常跑 ~524K 用 factor 2.0;上 1M 才用 4.0。
- **权衡**:`--context-length` 开多大,预分配的 KV cache 就多大(linear 层状态小、16 个 full 层才吃 KV),128GB 统一内存 + FP8 权重 ~24GB 下能开多少需实测,不是无脑开。

---

## 2. 引擎参数(9-04 实测,起服务时定,harness 侧只读参考)

> 这些不在请求里改;要改 → 重启 SGLang。下表是**从运行日志 server_args 读出的真值**,不是我计划里的初稿。

| 参数 | 实测值 | harness 含义 |
|---|---|---|
| `--context-length` | **32768** | 服务端上下文窗(可调,见 §1.3);当前比上一版写的 8K 宽 4 倍 |
| `--chunked-prefill-size` | 8192(默认) | 长 prompt 分块预填充,不是上下文上限 —— 上一版误当成它了 |
| `--mem-fraction-static` | 0.5 | 显存占用率;跑不起来调它 + 先卸 ollama 122B |
| `--attention-backend triton` | triton | sm_121a 上默认 flashinfer 会崩,**不能改** |
| `--reasoning-parser qwen3` | qwen3 | thinking 的 reasoning 走 `reasoning_content` |
| `--tool-call-parser qwen3_coder` | qwen3_coder | 让模型原生 tool_calls 被正确解析 |
| 端口 | 8003 | OpenAI 兼容端点 `http://localhost:8003/v1`,`api_key` 随便("not-needed") |
| 吞吐(实测 9-04) | ~7.9 token/s(full,thinking 下) | 长任务单轮生成要按几十秒~分钟预算 |
| `dtype/quantization` | auto / None | 权重本身是 FP8,引擎 auto 加载 |

**改动风险**:context 想开大/想上 1M → 改 `--context-length` + YaRN override 重启,并留意显存/吞吐/短文本损伤。

---

## 3. 请求参数(每次调用可改,harness 高频打交道)

### 3.1 thinking 开关(最常碰)

- **默认开**(`default_enabled=True`,引擎 auto-detect 确认):agent 复杂决策先出 `reasoning_content`,再给 `content`。协调者路由/判型可以要 thinking。
- **保留 thinking 历史**:Qwen3.8 默认保留所有历史消息的 thinking 块(`preserve_thinking`),agent 场景决策一致性好,但**吃上下文预算**。
- **关**:每次请求传
  ```python
  extra_body={"chat_template_kwargs": {"enable_thinking": False}}
  ```
  → 不再出 reasoning,响应更快、token 更省。适合"快问快答"子调用。

### 3.2 工具/function calling(阶段 0 核心)

- 格式:OpenAI 兼容 `tools=[{type:"function", function:{name, description, parameters}}]`;模型返回合法 `tool_calls`(name + JSON 参数)。
- 回填:assistant 带 `tool_calls` → 追加 `{"role":"tool","tool_call_id":..., "content": <工具结果 JSON>}` → 模型续答。可多轮。
- **死循环防护**:客户端要设 `max_rounds`(aux 用 4,demo 用 3)。
- 采稳:工具循环建议 `temperature=0.0`(schema 稳定,少瞎编参数)。

### 3.3 采样参数

| 参数 | config 默认(generation_config.json) | harness 建议覆盖 |
|---|---|---|
| `temperature` | 1.0(`do_sample: true`) | 工具循环 0.0;开放生成 0.3(aux 实测) |
| `top_p` | 0.95 | 一般不动 |
| `top_k` | 20 | 一般不动 |
| `max_tokens` | — | aux 默认 1500;长方案类调大(注意 32K 总预算,见 §1.3) |

### 3.4 特殊 token(config)

- `eos_token_id`:**[248046, 248044]**;`bos/pad = 248044`。
- 影响:JSON/结构化输出要防提前截断——能包进 tools 的走 tool_calls,别靠裸文本抽 JSON。

### 3.5 视觉(激光看图专家预留)

- OpenAI `image_url` + base64(dataURL/JPEG)直传即可(demo 已顺带验证视觉 200)。
- **注意**:图经 VL 编码后要占 token;上下文预算按 32K 算,不是 8K。

---

## 4. 客户端代码种子(harness 阶段 0 直接抄)

| 文件 | 提供了什么 |
|---|---|
| `Laser_assistant/aux_assistant/llm.py` | `QwenClient`: `complete()`、`reasoning_of()`、`run_tools(messages, tools, execute, max_rounds, on_event)` —— **单轮工具循环已经封装好**,这是 harness 引擎的起点 |
| `~/sglang/demo_tool_call.py` | 最简 demo:普通对话(thinking 开)、工具往返、关 thinking 三场景,`temperature=0.0` |

aux 的关键取值:`base_url=http://localhost:8003/v1`、`model=qwen3.8-27b`、`temperature=0.3`、`max_tokens=1500`、thinking 由 `enable_thinking` 控制。

---

## 5. 参数来源与溯源

- **模型能力/config 真值**:`/home/dongshuang/llm/qwen3.8/config.json` + `generation_config.json` + `tokenizer_config.json` + 模型目录 `README.md`(官方声明,含 YaRN/1M/SGLang 段落)。
- **引擎实测值**:`~/laser_rag/logs/qwen38_sglang.log` 里 9-04 的 `server_args`(context 32768、chunked-prefill 8192、triton、parsers 等全部来自这里)。
- **启动命令初稿**:`~/.claude/plans/cached-wandering-corbato.md`(写的是 8192,是**计划初稿**,与实际跑通值不符,以此文档为准)。
- 环境隔离:conda env `sglang`(`/home/dongshuang/miniconda3/envs/sglang`),**不 import `laser_qa` 业务代码**,不碰 vLLM 0.26 / finetune env。

---

## 6. 运行前提与坑(起服务前必读)

1. **先卸 ollama 122B**(`ollama stop`):它占 RAM 会让 SGLang memory profiling 失败——和 vLLM 35B 同款坑。
2. 端口 8003 别和别的服务撞(laser_qa 用 8000/8002,ollama 11434)。
3. 起服务 → `curl /v1/models` 确认 `qwen3.8-27b` 就位,再跑 `demo_tool_call.py` 冒烟。
4. 日志:`~/laser_rag/logs/qwen38_sglang.log`(起服务重定向)。排查 SIGTERM/吞吐/队列都在这里。
5. 想验证长上下文:先不开 YaRN 跑 32768 冒烟;确实需要 >262K 再照 §1.4 开,并留意短文本退化。

---

## 7. Harness 要盯的其它 Qwen 参数(除上下文外)

> 上下文(§1.3)只是第一项。做多智能体编排时,下面这些直接影响**每轮行为、上下文预算、成本、门禁稳定性**。分四组:thinking 控制 / 采样确定性 / 结构化输出 / 引擎吞吐。每项标了"harness 场景 / 当前状态 / 建议"。

### 7.1 Thinking 的细粒度控制(最影响预算和时延)

| 参数 | 取值 | harness 场景 | 当前状态 | 建议 |
|---|---|---|---|---|
| `enable_thinking` | true/false | 协调者路由/判型开;快问快答子调用关 | 逐请求可开可关(`extra_body.chat_template_kwargs`) | 默认开,按角色粒度开关 |
| `preserve_thinking` | true/false | thinking 历史**默认全保留**——多轮工具循环里每轮 thinking 都留在上下文 | 官方默认 **true**;`chat_template_kwargs.preserve_thinking=false` 可关 | 长任务若上下文吃紧,对不需要推理记忆的专家关掉,能省一大截预算 |
| `reasoning_effort` | **xhigh / medium / low** | 调推理深度/成本 | 模型官方支持;**但当前 SGLang auto-detect `effort_kwarg=None`**,逐请求设未必生效 | 别依赖它做节流;先实测 `xhigh/medium` 是否有差别,没差别就用 thinking 开关代替 |

**为什么 preserve_thinking 在 harness 里是重点**:默认它把**每一轮**的 reasoning 都追加回 messages。你的 run_tools 循环跑 3–4 轮 → 一轮 deep thinking 可能就几千 token 留在历史里 → 很快吃掉 32K。这也是上一版"8K 很紧"直觉的来源——**真正吃上下文的是 thinking 历史,不是对话本身**。

### 7.2 采样确定性(门禁/Gate 稳定性)

| 参数 | 取值 | harness 场景 | 建议 |
|---|---|---|---|
| `temperature` | 0–2 | 工具调用/结构化产出要**可复现** | 工具循环 `temperature=0`;但注意 GPT 风格 temp≠纯贪心,Qwen 下仍可能采样 |
| `top_p` / `top_k` | 0.95 / 20(默认) | 内容生成多样性 | 一般不动;工具循环可 `top_p=1, top_k=-1` 尽量去随机化 |
| `max_tokens` | — | 每轮输出预算,`max_tokens` 设太小 thinking 可能被截断 | thinking 开着时把输出预算给足(见 3.3);截断的 reasoning 会污染下一轮 |

**对 harness 的实际含义**:Gate 用自动校验(必填项/JSON schema)会比"让模型保证格式"可靠。确定性参数是辅助,不该当门禁本身。

### 7.3 结构化输出(直接喂给 Gate / Spec 文件)

| 项 | 说明 |
|---|---|
| 语法后端 | SGLang 这个启动的 `grammar_backend='xgrammar'`(日志 server_args 确认) |
| 用途 | 让专家 Agent 直接产出**符合 schema 的 JSON**(如 spec 文件、工具入参),而不是靠提示词"请输出 JSON"再抽 |
| 当前状态 | 引擎已带 xgrammar;还没在 harness 里用 |
| 建议 | 阶段 3 做 Gate 时,配合 OpenAI `response_format={"type":"json_schema", ...}` 或工具入参约束,能显著减少"模型给的 JSON 抽不出来"类故障——这正好对应 requirements 里"SQL 语法校验"那类自动门禁 |

### 7.4 引擎吞吐与并发(多专家并行时)

| 参数 | 实测/默认 | harness 场景 | 建议 |
|---|---|---|---|
| `chunked-prefill-size` | 8192 | 长 prompt(角色文件+历史+工具结果)分块预填,避免大请求占满 | 保持默认;prompt 超 8K 会自动分块 |
| `max_running_requests` | None(未设) | 协调者派多个专家并行时的队列行为 | 阶段 2 做并行前先压测:几个并发请求时吞吐怎么掉 |
| 单请求吞吐 | ~7.9 token/s(full,thinking) | 一个长任务=几十秒~几分钟 | 预算耗时;并行专家是"总吞吐换单请求时延" |
| 前缀缓存 | 引擎默认开启(radix cache) | 多个专家共享**同一段 system prompt/角色前缀**时可复用 KV | **把公共约束(红线+身份)放 system 最前且字节级一致**,专家越多省得越多——这也是"角色文件复用"的一个隐藏收益 |

### 7.5 一份"harness 每次调用都该显式带上"的清单(建议模板)

```python
client.chat.completions.create(
    model="qwen3.8-27b",
    messages=...,                      # 顺序:共享红线+身份 → 阶段上下文 → 本轮
    temperature=0.0,                   # 工具循环;开放生成再改 0.3
    max_tokens=4096,                   # thinking 开着要留足
    extra_body={"chat_template_kwargs": {
        "enable_thinking": True,        # 按角色开关
        # "preserve_thinking": False,   # 上下文吃紧时按专家关
    }},
    # tools=..., tool_choice="auto",   # 阶段 0 起
    # response_format={"type": "json_schema", ...}   # 阶段 3 Gate 起
)
```

> 原则:**能默认的别每次都传,会变的分角色传**。上面只有 `temperature`/`max_tokens`/`enable_thinking`/`preserve_thinking` 是按调用变化的,其余由起服务参数与代码模板固定。
