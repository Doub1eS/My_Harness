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
