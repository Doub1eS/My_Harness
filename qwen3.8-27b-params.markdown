# Qwen3.8-27B 参数速查(harness 涉及部分)

> 日期:2026-09-09 ｜ 状态:**服务当前未运行**(上次 2026-09-04 SIGTERM 停止),本文是"起得来之后的参数底账"。
> 用途:harness(阶段 0 起)所有与模型打交道的地方都对着这张表。参数分两类——**引擎参数**(起服务时定的,改要重启)和**请求参数**(每次调用可改)。

---

## 0. 一条命令快速起服务

```bash
# conda env: sglang(python 3.11, 独立于 finetune/vLLM)
nohup /home/dongshuang/miniconda3/envs/sglang/bin/python -m sglang.launch_server \
  --model-path /home/dongshuang/llm/qwen3.8 \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port 8003 \
  --context-length 8192 \
  --mem-fraction-static 0.5 \
  --attention-backend triton \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  > ~/laser_rag/logs/qwen38_sglang.log 2>&1 &
```

冒烟:`curl localhost:8003/v1/models` → `id = "qwen3.8-27b"`。

---

## 1. 模型身份(你是谁、在哪个栈)

| 项 | 值 | 备注 |
|---|---|---|
| 命名 | **Qwen3.8-27B**(served-name `qwen3.8-27b`) | API 里 `model` 字段填这个 |
| 权重目录 | `/home/dongshuang/llm/qwen3.8` | modelscope 下载,FP8 |
| 架构(config) | `Qwen3_5ForConditionalGeneration`(model_type `qwen3_5`) | dense 多模态(image+text),非 MoE |
| 注意力 | **混合 GDN**:linear_attention(3 层)↔ full_attention(1 层)循环,`full_attention_interval=4` | SGLang 走自有 Triton `GDNAttnBackend` |
| 多模态 | 带视觉(`vision_config`: depth 27 / hidden 1152 / patch 16) | 支持 `image_url` 传图 |
| 精度 | 权重 FP8(e4m3,dynamic);config 声明 bf16 | 8-21 vLLM 曾 auto-fp8 成功加载 |
| 部署机 | DGX Spark(GB10,aarch64,sm_121a),统一内存 128GB | 与 laser_qa 的 vLLM 35B(:8002)同机不同栈 |

---

## 2. 引擎参数(起服务时定,harness 侧只读参考)

> 这些不在请求里改;要改 → 重启 SGLang。

| 参数 | 当前值 | harness 含义 |
|---|---|---|
| `--context-length` | **8192** | **硬上下文上限**。整条多轮工具循环 + 历史 + 工具结果都挤在 8K 内 → harness 的 Context 工程(CP 摘要/截断/渐进加载)是刚需,不是优化。config 原生 `max_position_embeddings=262144`,但服务只开 8K |
| `--mem-fraction-static` | 0.5 | 显存占用率;跑不起来调它 + 先卸 ollama 122B |
| `--attention-backend triton` | triton | sm_121a 上默认 flashinfer 会崩,**不能改** |
| `--reasoning-parser qwen3` | qwen3 | 让 thinking 的 reasoning 走 `reasoning_content` 字段 |
| `--tool-call-parser qwen3_coder` | qwen3_coder | 让模型原生 tool_calls 被正确解析 |
| 端口 | 8003 | OpenAI 兼容端点 `http://localhost:8003/v1`,`api_key` 随便("not-needed") |
| 吞吐(实测 9-04) | ~7.9 token/s(full,thinking 下) | 长任务单轮生成要按几十秒~分钟预算 |

**改动风险**:context 想开大 → 改 `--context-length` 重启,并留意显存/吞吐。

---

## 3. 请求参数(每次调用可改,harness 高频打交道)

### 3.1 thinking 开关(最常碰)

- **默认开**(Qwen3.5 系):agent 复杂决策会先出 `reasoning_content`,再给 `content`。harness 的协调者路由/判型可以要 thinking。
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
| `max_tokens` | — | aux 默认 1500;长方案类调大(注意 8K 总预算) |

### 3.4 特殊 token(config)

- `eos_token_id = [248046, 248044]`,`pad/bos = 248044`。
- 影响:JSON/结构化输出要防提前截断——能包进 tools 的走 tool_calls,别靠裸文本抽 JSON。

### 3.5 视觉(激光看图专家预留)

- OpenAI `image_url` + base64(dataURL/JPEG)直传即可(demo 已顺带验证视觉 200)。
- **注意**:图经 VL 编码后要占不少 token,别爆 8K 上下文。

---

## 4. 客户端代码种子(harness 阶段 0 直接抄)

| 文件 | 提供了什么 |
|---|---|
| `Laser_assistant/aux_assistant/llm.py` | `QwenClient`: `complete()`、`reasoning_of()`、`run_tools(messages, tools, execute, max_rounds, on_event)` —— **单轮工具循环已经封装好**,这是 harness 引擎的起点 |
| `~/sglang/demo_tool_call.py` | 最简 demo:普通对话(thinking 开)、工具往返、关 thinking 三场景,`temperature=0.0` |

aux 的关键取值:`base_url=http://localhost:8003/v1`、`model=qwen3.8-27b`、`temperature=0.3`、`max_tokens=1500`、thinking 由 `enable_thinking` 控制。

---

## 5. 引擎参数与请求参数的完整对应(config 溯源)

- 真值文件:`/home/dongshuang/llm/qwen3.8/config.json` + `generation_config.json`(上面 3.3/3.4 全部来自这里)。
- 启动参数来源:`~/.claude/plans/cached-wandering-corbato.md`(SGLang 源码部署 + qwen3.8 冒烟计划),9-03/9-04 已实际跑通过。
- 环境隔离:conda env `sglang`(`/home/dongshuang/miniconda3/envs/sglang`),**不 import `laser_qa` 业务代码**,不碰 vLLM 0.26 / finetune env。

---

## 6. 运行前提与坑(起服务前必读)

1. **先卸 ollama 122B**(`ollama stop`):它占 RAM 会让 SGLang memory profiling 失败——和 vLLM 35B 同款坑。
2. 端口 8003 别和别的服务撞(laser_qa 用 8000/8002,ollama 11434)。
3. 起服务 → `curl /v1/models` 确认 `qwen3.8-27b` 就位,再跑 `demo_tool_call.py` 冒烟。
4. 日志:`~/laser_rag/logs/qwen38_sglang.log`(起服务重定向)。排查 SIGTERM/吞吐/队列都在这里。
