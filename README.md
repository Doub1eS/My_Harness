# My_Harness

自研通用多智能体 **harness 框架**的学习项目。目标:从零搭出一个可运行的 Orchestrator + Specialist 编排框架,在过程中掌握六支柱方法论(Identity / Orchestration / Context / Gate / Recovery / Evolution),最终长出适配激光助手(Laser Assistant)的形态。

基座:**Qwen3.8-27B**(FP8,多模态)+ **SGLang**(OpenAI 兼容端点 :8003),与 laser_qa 的 vLLM 35B 在线栈独立并存;理念后期再带回激光项目后期迭代。

## 文件

| 文件 | 内容 |
|---|---|
| `requirements.markdown` | 六支柱需求蓝本(理想形态,源自 Qoder/SQL 数仓场景)——设计参考,不是最终形态 |
| `harness-design.markdown` | 通用 harness 设计稿:六支柱去场景化 + 阶段 0→5 学习路线 + 激光适配预留预案 |
| `qwen3.8-27b-params.markdown` | Qwen3.8-27B 参数速查(引擎参数/请求参数/客户端种子/运行前提) |

## 阶段路线(详见 harness-design.markdown §9)

0. 单 Agent + 1 工具循环跑通(抄 `aux_assistant/llm.py` 的 `run_tools`)
1. 身份层(角色文件 + 红线加载)
2. 协调者 + 2 专家最小骨架
3. Gate(门禁)+ Recovery(状态机/故障分级)
4. Context(phase 切分 + CP 摘要)
5. Evolution + 端到端验收

每个 demo 跑通记录 + 对设计的修正,记在对应阶段的 README/日志里 —— 学习本身,不是产出。
