# safety-model — 模型侧（PreTrained Model）

GP1 审批门架构里橙色 **PreTrained Model** 的实现：把 `Qwen2.5-1.5B-Instruct` 后训练成
"工具调用风险打分器"，对门卫发来的单次工具调用输出 `{"risk": <0..1>, "reason": "<一句话>"}`，
由门卫按阈值做 approve / deny / escalate。接口冻结见 `../接口/接口信息.md`，
任务澄清见 `../GP1_设计文档/模型部分_任务澄清与规划.md`。

## 当前状态（2026-10-14）

- ✅ 基座已下载到本机 HF 缓存（约 3.09GB，bf16）；
- ✅ 训练/校验/评测基础设施就绪，`fixtures/tiny.jsonl` 冒烟通过；
- ⏳ 等数据同学交付 `data/train.jsonl` / `data/val.jsonl`（字段 spec 见澄清文档 §4）；
- ✅ 本机 CUDA torch 就绪：`2.11.0+cu128`（RTX 5060 Laptop，sm_120 实测可用）；
- ✅ 30 步 GPU 吞吐基准（2026-10-14 实测）：稳态 **~1.95 s/optimizer-step**（micro2×accum8，seq≤241），30 步 + 2 次 eval 共 109s，峰值显存 7.75/8.15GB ⇒ 整轮估算 0.5~1h，**本机可承担正式训练**（细节见 FAQ）；
- ⏳ 学校 GPU 用途：加速 + 多组消融（配置无需改，micro8×accum4 已注明）。

## 目录结构

| 文件 | 作用 |
|---|---|
| `prompts.py` | 冻结 system prompt（逐字符=接口文档）+ 目标文本模板 + risk 解析 |
| `safety_data.py` | 规范 schema、校验、可复现按组切分、统计 |
| `inference_utils.py` | 模型加载（base/LoRA）、冻结序列化 prompt、生成 |
| `metrics.py` | AUROC / AP / P-R-F1 / ECE（纯 numpy，无新依赖）|
| `train_sft.py` | LoRA 训练主脚本（warmup/cosine、早停、best/last 断点续训）|
| `eval_offline.py` | 离线评测 + 阈值扫描 CSV（给门卫挑阈值）|
| `probe_zeroshot.py` | 零样本探针：基座对冻结 prompt 的 JSON 服从度 |
| `validate_data.py` | 数据验收：schema、重复 id、切分泄漏、token 长度审计 |
| `configs/sft_lora.yaml` | 全部超参（唯一入口）|
| `fixtures/tiny.jsonl` | 12 条手写样本（6 安全 / 5 攻击 / 1 overlap），冒烟用 |
| `serve.py` | OpenAI 兼容 HTTP 服务（`--backend stub\|model` 双后端，标准库零依赖）|
| `smoke_serve.py` | 服务端到端自检：自动起停 + 契约检查（`--with-model` 连真权重）|

## 训练决策（v0，全部集中在 `configs/sft_lora.yaml`）

**切分策略**
- 单位：`(suite, user_task)` 组；同组调用永不跨 split；
- 分桶：`sha256(split_seed | 组键) % 100` → 80/10/10（train/val/test）。
  确定性分桶 = 任何人重算结果一致、多来源数据可直接合并；
- `validate_data.py` 会检查跨文件组泄漏（同组出现在两个 split = 报错）。

**warmup 与调度**
- 线性 warmup 前 3% 步数、下限 20 步；之后 cosine 衰减至峰值 lr 的 10%；
- LoRA 峰值 lr `2e-4`、weight decay `0.01`（bias 除外）、grad clip `1.0`。

**早停**
- 主指标 val **AUROC**（生成解析出的 risk 的排序质量）；无法计算时退化为 `-val_loss`；
- patience=3 次评估无提升（min_delta=0.002）即停，并恢复历史最优（`best/`）；
- 每次评估记录 `val_loss / AUROC / 解析失败数` → `metrics.jsonl`。

**batch 与长度**
- 本机 8GB：micro_batch=2 × grad_accum=8 = 有效 16；
- 学校 GPU 建议 micro=8 × accum=4 = 有效 32（已注明在 config）；
- `max_seq_len=1024`（fixture 实测 prompt+answer：p50=187 / p95=240；真实数据接入后复测）；
- OOM 时开 `grad_checkpoint: true`。

**训练目标格式**
- 目标文本 = `{"risk":0.95,"reason":"..."}`；软标签 0.05 / 0.95；
- reason 由模板生成（`prompts.target_reason`），可被数据中的 `label.reason` / `label.risk` 覆盖。

**断点续训**
- `best/`、`last/` 各存 LoRA adapter + tokenizer + optimizer/scheduler state；
- 被抢占后 `--resume` 从 `last/` 恢复。

## 快速开始（本机 Python 3.10）

```powershell
cd safety-model

# 1) 数据验收（含 token 长度审计）
py -3.10 validate_data.py fixtures/tiny.jsonl --tokenizer Qwen/Qwen2.5-1.5B-Instruct

# 2) 冒烟训练（CPU、2 步、真模型）
py -3.10 train_sft.py --config configs/sft_lora.yaml --train fixtures/tiny.jsonl --val fixtures/tiny.jsonl --smoke

# 3) 离线评测 + 阈值扫描（runs/eval_offline/<ts>/thresholds.csv）
py -3.10 eval_offline.py --data fixtures/tiny.jsonl

# 4) 零样本探针（不训练，看基座对冻结 prompt 的服从度）
py -3.10 probe_zeroshot.py --data fixtures/tiny.jsonl --limit 4

# 5) GPU 吞吐基准（30 步 + 2 次 eval，约 2 分钟）
py -3.10 train_sft.py --config configs/sft_lora.yaml --train fixtures/tiny.jsonl --val fixtures/tiny.jsonl --device cuda --max-steps 30 --eval-steps 15 --output-dir runs/bench-fixture

# 6) 正式训练（等 data/）
py -3.10 train_sft.py --config configs/sft_lora.yaml
```

## 服务进程（独立进程，M1 交付）

`PreTrained Model` 框的 OpenAI 兼容 HTTP 服务，契约逐字符对齐 `../接口/接口信息.md`：

```powershell
py -3.10 serve.py                    # stub 桩（默认 127.0.0.1:8100）
py -3.10 serve.py --backend model    # 基座/合并 LoRA（首个请求惰性加载）
py -3.10 serve.py --backend model --adapter runs/qwen25-1.5b-risk-lora-v0/best --preload
py -3.10 smoke_serve.py --with-model # 端到端自检（自动起停，约 20s）
```

| 端点 | 说明 |
|---|---|
| `POST /v1/chat/completions` | 冻结契约：接受并忽略 `response_format`；`choices[0].message.content` = `{"risk":...,"reason":...}` |
| `GET /healthz`（别名 `/health`）| 存活 + 当前后端 / 模型 / prompt 版本 |
| `GET /v1/models` | OpenAI 风格模型列表 |

- 门卫联调直接指向 `http://127.0.0.1:8100/v1/chat/completions`（端口可 `--port` 改）；
- `--backend stub`：规则桩（元数据表 + 参数黑名单）——**结果不得写进报告**；
- 每次请求记录 `risk / reason / latency_ms / prompt_tokens / completion_tokens / prompt_version / system_sha256`，
  追加到 `runs/serve_requests.jsonl`（报告 policy 成本列），stdout 同步打一行；
- 实测（2026-10-14，本机 RTX 5060）：stub <1ms；真权重首个请求 19s（含 3GB 加载），预热 **~1.2s/次**。

## 预训练语料与任务匹配性

- **任务型态**：短英文 JSON 输入（实测 p50≈190 token）+ 紧凑 JSON 输出（≈25 token），
  本质是"判断这次工具调用是否越权"的浅推理分类；
- **基座能力**：Qwen2.5 预训练覆盖网页/代码/数学（约 18T tokens、29 种语言），
  Instruct 版做过 SFT+DPO，JSON 与工具调用类结构化文本在其能力范围内，32K 上下文
  远大于我们的需求；因此"语料相符"——需要的是**任务适配**而不是领域补课；
- **需要后训练补齐的部分**：AgentDojo 语义（收款人 IBAN 归属、攻击者目标）、
  我们的输出契约、风险标定 —— 全部由 SFT + val 阈值校准覆盖；
- **已知短板与对策**：1.5B 的 JSON 服从度弱于大模型 → SFT 针对性训练 + 贪心解码 +
  解析容错（`prompts.parse_risk`）；risk 的**排序质量**（AUROC）比绝对值重要，
  三分类阈值一律在 val 上校准；证据看 `probe_zeroshot.py` 的输出（`runs/zeroshot_probe.jsonl`）。
- **零样本探针结论（2026-10-14）**：基座对冻结 prompt 的格式服从 4/4 全部可解析；但风险语义方向错误（安全的只读调用也给 0.95，reason 与数字自相矛盾）。⇒ 结论：**格式达标、语义必须靠后训练**；SFT 是主要工作，阈值校准兜底。

## 选型 FAQ（2026-10-14 讨论记录）

**Q: 预训练语料有英文吗？** 有。Qwen2.5 预训练以中英为主（29 种语言 + 代码/数学，约 18T
tokens）；AgentDojo 的英文任务和工具 JSON 都在覆盖范围内，不存在"中文模型看不懂英文任务"的问题。

**Q: 为什么门卫风险模型选 1.5B（"笨但合适"）？** 门卫不执行动作、只打分——**越准越好**，
"抗注入敏感度"对它是优点而不是缺点。1.5B 的定位是"便宜 + 本地 + 格式达标"：探针已证明
格式 100% 达标、语义靠 SFT 补齐；瓶颈在数据与校准，不在参数。升级路径：1.5B 为主；若 val
AUROC 不达标，再试本机已有的 Qwen2.5-3B（同族、改动成本低）；不建议上更大模型——成本/
延迟直接进报告的 policy 成本列。
注意区分：**agent（dsh 侧）"不能太聪明"是实验设计问题**（要能完成任务、又不能自带强抗注入
否则攻击永远打不进去）；选型前应在候选 agent 上先量化 bare utility 与 ASR，选定后所有
policy 实验冻结同一个 agent。

**Q: 本机（RTX 5060 8GB）能训练吗？** 能。2026-10-14 实测（torch 2.11.0+cu128、bf16、
micro=2×accum=8、fixture seq≤241）：30 步 + 2 次 eval 共 109s，稳态 **~1.95 s/step**，峰值显存 7.75/8.15GB
（出现在 eval 生成阶段）。外推真实数据（3~5k 样本、有效 batch 16、5 epoch ≈ 940~1565 步）：
训练约 30~50 min，含 eval/checkpoint 整轮约 **0.5~1 小时**——本机可承担正式训练与多轮迭代；
学校 GPU 用于加速 + `micro=8×accum=4` 消融。显存余量小的注意：真实序列变长时开 `grad_checkpoint: true`。
推理：GPU 约 20~80ms/次，CPU 约 2~10s/次。

**Q: 为什么不用 FFT（全参数微调）？** 8GB 显存放不下：1.5B 的 bf16 权重 3GB + 梯度 3GB +
AdamW fp32 动量 12GB ≈ 18GB 起（未计激活）。LoRA 只训练 18.5M 参数（~0.7%），优化器状态
极小；几千条窄任务数据下 LoRA 与 FFT 的差距通常可忽略，且 adapter 仅 74MB（易回滚、可
多实验并行）。若容量不够：先加大 r，或换 3B，而不是先上 FFT。

**Q: 为什么 SFT 最合适？** ① 监督标签现成（labels 口径 authorized）；② 训练目标与接口
同构（同一 prompt + 序列化 + 输出格式），train/serve 零偏移；③ 成本最低、失败可解释
（解析率=格式问题，AUROC=语义问题，分开诊断）；④ 备选均更差或更贵——zero-shot 探针已证
语义不行；分类头丢 reason 且偏离冻结接口；DPO/RLHF 需要偏好数据与更大算力，留作二轮
（难例偏好优化）即可。

**Q: 训练数据要不要负样本？** 必须要。负样本（authorized=true 的安全调用）定义决策边界；
缺了它会退化成"全高风险"（零样本探针就是这个失败模式：安全的只读调用也给 0.95），TCR
会被误伤毁掉。重点在**难负样本**：同工具、同参数位、不同取值（收款人=账单方 vs 攻击者）。
配额建议：安全:危险 ≈ 1:1~2:1；危险侧参数扰动 ≥30%；覆盖常规只读 + 任务必需写操作；
真实运行里"既非攻击、也非真值命中"的普通调用也要有（对齐部署分布）。

## Windows 下载注意（已踩坑）

- 本机无 symlink 权限：设 `HF_HUB_DISABLE_SYMLINKS=1` 后重跑 `snapshot_download`（降级为复制模式）；
- 基座缓存路径：
  `C:\Users\Fredd\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
  （学校 GPU 机器可整目录拷贝，或同法重新下载）。

## 冻结物（改任何一项 = 已训练权重作废）

1. `prompts.SYSTEM_PROMPT`（= 接口 `messages[0]`，逐字符一致）；
2. `safety_data.serialize_input`（紧凑 JSON；字段序 task → tool_name → tool_args → tool_metadata → trace）；
3. `tool_metadata` 取值来源（`../GP1_设计文档/工具清单.md` 的表）。

## TODO

- [ ] 数据格式到手后：加适配器把数据同学的字段映射到 canonical schema（一个小函数，validate/train 复用）；
- [ ] 数据交付后全量验收：`validate_data.py --tokenizer ... data/train.jsonl data/val.jsonl`；
- [ ] 用真实数据复测 token 长度分布，必要时调 `max_seq_len` / trace 截断；
- [ ] 学校 GPU：正式训练 + 校准，把阈值建议交给门卫；
- [x] 服务进程（OpenAI 兼容 `/v1/chat/completions`）：`serve.py` stub/model 双后端完成（2026-10-14）；训练完换 `--adapter` 即上线；
- [ ] 难例挖掘第二轮（用首版模型在 train 集上的错误样本）。

