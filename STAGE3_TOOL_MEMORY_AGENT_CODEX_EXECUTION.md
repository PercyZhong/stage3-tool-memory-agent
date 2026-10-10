# 阶段三执行文档：带工具与记忆的智能体

版本：2026-10-10。交给 Windows VS Code 中的 Codex 执行；正式推理与实验只在远程 Linux 服务器运行。

## 0. 阶段目标与边界

实现一个小型、单智能体、可审计的 ReAct 式工具循环和 MemGPT 启发的分层记忆。在离线的虚构校园信息环境里完成任务。**交付：任务样例、完整执行轨迹、按记录重放的结果，以及比较报告**。这是方法教学实验，不声称复现两篇论文的原始结果，也不声称能预测真实校园。

论文：ReAct <https://arxiv.org/abs/2210.03629>；MemGPT <https://arxiv.org/abs/2310.08560>。从 ReAct 借鉴“选择动作→工具观察→更新状态→继续/结束”，从 MemGPT 借鉴“工作记忆、历史记录、长期存储、按需检索”。日志中记录可见的决策摘要/结构化动作，不要求记录供应商不可见的内部推理。

本阶段新仓库建议命名 `stage3-tool-memory-agent`，与阶段一、二仓库完全隔离。**不把阶段二的 v1/v2 adapter 当作本阶段的默认智能体模型**；若另做迁移试验，作为扩展单列，避免混淆阶段三目标。

## 1. 已知条件及执行环境

- Windows：在本地 VS Code 让 Codex 新建仓库、编写代码和文档，运行无需模型的单元测试，然后 `git push`。
- Linux：普通用户权限，Python 3.11.16、PyTorch 2.5.1+cu121、4×RTX 4090（共享设备，运行时须重新检查空闲状态），约 3 TB 可用共享空间。只在个人目录建立虚拟环境/缓存/输出，不安装系统软件，不改全局配置。
- Linux 无法直连 huggingface.co；模型权重**是否已经本地缓存未知**。不得把联网下载作为实验的隐含前提。先做本地模型路径预检；缺失时报告缺失路径并停止真实模型实验，由用户按实验室允许的方式准备模型后续跑。不得以 mock 结果冒充模型实验。
- Windows 的 `gh` 登录状态和 GitHub 权限在执行时检查。只有本地认证可用时创建 GitHub 仓库并推送；不得索取或提交令牌。已有同名仓库时检查远端归属与内容，避免覆盖；不强推。

## 2. 核心研究问题和任务

R1：智能体能否利用受限工具获取正确且可引用的外部事实？R2：跨会话偏好或事实不在当前输入中时，记忆检索能否改善回答？R3：输入事实发生更新时，能否避免沿用过期记忆？R4：是否能用保存的轨迹在无模型、无网络条件下检验相同事件、状态变更和最终输出？

使用**明确标记为合成**的校园环境，例如课程时刻表、教室、社团活动、报名规则。所有事实来自版本化的 JSON/JSONL fixture，不引用真实学生隐私，不调用真实校园服务。至少 12 个任务样例，分为三类，每类至少 4 个：

1. **多步工具任务**：必须联合查询两项校园记录，例如“周三晚有空且符合摄影社报名条件的活动有哪些？”
2. **跨会话记忆任务**：第一会话提供用户偏好（如无障碍场地/周三不参加晚间活动），第二会话省略偏好并提出新问题，必须通过记忆找回；阶段内可设置多条连续交互，但评测计数以任务为单位。
3. **冲突与更新任务**：先保存旧安排，再将权威 fixture 更新至新版本，第二会话要求查询最新安排；智能体应给出新版并注明其证据，不能让旧记忆盖过工具的最新记录。

每个任务有稳定 `task_id`、`session_id`、`user_id`（虚构）、`fixture_version`、输入、允许工具、所需证据 ID、机器可检查的目标字段和 `expected_behavior`。标注答案由 fixture 人工预先编写，不要调用受评模型生成真值。按用户 ID 隔离记忆；每个实验臂从同一初始状态和相同任务顺序启动，臂之间不得共享写入后的记忆。

## 3. 功能设计

### 3.1 ReAct 式执行器

单步输出结构化 `tool_call`（名称与 JSON 参数）、`memory_call` 或 `final`；解析、校验 schema 并限制工具白名单。循环为：组装当前 prompt → 模型响应 → 解析动作 → 执行 → 将**真实观察/错误**追加到上下文 → 下一步；最多 8 次动作，达到上限记录明确失败状态。不得将模型声称已调用工具当作实际调用。处理无效 JSON、未知工具、超时、重复调用；错误返回模型供一次修正，并保留完整事件。不执行任意 shell 命令、任意 Python 或网络请求。

只提供只读校园工具，如 `lookup_schedule`, `find_events`, `lookup_policy`，输入限定类型和长度，输出包括 `record_id`、`fixture_version`、`source`、内容；所有工具执行完全本地、确定性。工具的输出作为数据处理，禁止其文本覆盖上层指令。最终回答至少输出可核对的结构化字段、证据 ID 和面向用户的文本；无法取得依据时允许明确“不知道”。

### 3.2 记忆管理器

- 当前上下文/工作记忆：当次任务要点、近期观察，使用明确 token/字符预算；记录装配后的完整 prompt 和其截断策略。
- 长期记忆：用户偏好、过去会话中需保留的事实，SQLite 或 JSONL（优先 SQLite，标准库即可）；每条带 `memory_id`、命名空间、内容、创建/更新时间、来源事件、状态/失效标记。
- 历史记录：对话与工具事件持续保存；检索通过显式 `memory_search` 动作实现，并写入命中 ID、排序与返回内容。提供 `memory_write` 和 `memory_update/invalidate`，旧记忆不直接覆盖新事实；当权威工具与旧记忆冲突时以 fixture 版本化工具记录为准。
- 摘要与裁剪：超预算时先保留系统规则、当前问题和近期工具结果，清理较早历史；可写摘要，但必须注明摘要来源和原始事件 ID。不要仅仅将整个历史塞回模型上下文来伪装“记忆检索”。

### 3.3 模型适配器与离线保障

支持 `--backend scripted`（固定动作脚本，用于工程与回放测试）和 `--backend local_hf`（真实本地模型，用于正式实验）。真实模型默认建议 `Qwen/Qwen2.5-0.5B-Instruct` **对应的已授权本地目录**，只作为初始小模型选择，不能假设已经下载；允许 CLI 指向其他能运行的本地指令模型。加载必须 `local_files_only=True`，禁止运行时静默联网。小模型可能难以稳定输出动作 JSON，记录失败，不以硬编码任务 ID 的脚本替代真实模型结果。

锁定模型目录或 snapshot/revision、tokenizer 与 chat template、推理参数（temperature=0、最大新 token、上下文长度）、seed、库版本、CUDA/GPU、代码 commit、fixture hash。`temperature=0` 仍不能保证各 GPU/内核完全逐字一致；验收的严格重放基于保存的响应与观察。

## 4. 必须实现的 CLI 与输出

以下名称为建议的稳定用户接口，可以内部重构，但 README 和实际命令必须一致：

```text
python -m stage3_agent.preflight --model-dir /absolute/local/model
python -m stage3_agent.run --backend scripted --suite data/tasks.jsonl --arm full --out runs/smoke
python -m stage3_agent.run --backend local_hf --model-dir /absolute/local/model --suite data/tasks.jsonl --arm full --seed 42 --out runs/full_s42
python -m stage3_agent.replay --run runs/full_s42 --mode recorded --out runs/replay_full_s42
python -m stage3_agent.verify --run runs/full_s42 --replay runs/replay_full_s42
python -m stage3_agent.evaluate --runs runs/full_s42 runs/no_memory_s42 runs/no_tools_s42 --out reports/s42
```

每次运行产生 `config.json`、`manifest.json`、`tasks.jsonl`（或其副本/hash）、`events.jsonl`、`responses.jsonl`、`memory_before.json`、`memory_after.json`、`metrics.json`、`examples.md` 和错误报告（即使为空也保留）。大模型权重不进仓库、不进 ZIP。

`events.jsonl` 每个事件包含：`run_id, task_id, session_id, seq, event_type, timestamp_or_logical_index, input, output, status, source_ids, pre_state_hash, post_state_hash`。涵盖模型可见请求和原始响应、解析结果、工具参数和原始观察、记忆读写/检索/失效、最终答案、错误与终止原因。敏感环境变量与密钥一律不写日志。采用规范 JSON 序列化（UTF-8、键排序、固定分隔符），散列原始 fixture 与关键工件；事件序号连续，哈希可重新计算。墙钟时间只作元数据，状态比对采用规范化内容和逻辑序号。

**recorded replay**：读取已保存的模型响应和工具观察，按原顺序驱动同一个解析器与状态机；不加载模型、不访问网络、不运行真实工具，逐步校验动作参数、事件序号、状态散列、记忆快照与最终答案。任何源文件缺失/篡改、消费不完的响应、事件不匹配都非零退出并输出具体差异。可以另提供 `live rerun`，但不能把它的逐字一致当作必须验收项。最好提供 `--from-task` 或按事件号定位故障。

## 5. 实验设计与指标

固定至少 12 个任务和同一 fixture，正式实验至少运行两个预先声明的 seed（建议 42、123），三个臂：

| 实验臂 | 工具 | 跨会话记忆 | 用途 |
| --- | --- | --- | --- |
| `full` | 开启 | 开启 | 目标系统 |
| `no_memory` | 开启 | 禁用持久记忆检索与跨会话写入 | 检查记忆作用 |
| `no_tools` | 禁用校园查询工具 | 开启 | 检查外部工具作用 |

每臂相同任务、顺序、模型、推理配置、seed 和初始 fixture；“无工具”臂允许承认不知道，**不能要求它编造事实**。跨会话任务的用户预置偏好只能经第一会话输入建立，不能在第二会话或系统提示泄漏。fixture 更新时间线对所有臂相同。实现数据校验器禁止评测目标/标准答案进入受评模型 prompt。

逐任务记录：`success`（目标字段与证据核验）、`evidence_valid`（证据 ID 存在且对应版本）、`tool_calls`、`memory_reads/writes`、`stale_fact_error`、`invalid_action`、`step_limit`、`latency`。按任务类型×臂×seed 报告分子/分母与失败样例；若只完成部分 run，显示 completed/attempted 和缺失原因，不选择性忽略失败。成功判定是机器核对 + 抽查的记录，不以模型自评替代。为每类至少保留一条成功和一条失败/边界轨迹；若没有失败，说明无失败。

不得预设 full 必胜：样本小、模型小，结果可能无改善。结果解释只针对合成任务和固定配置。

## 6. 仓库目录与 Git 规则

```text
stage3-tool-memory-agent/
  README.md
  pyproject.toml                  # Linux Python 3.11，依赖版本记录或锁文件
  configs/                        # 模型与实验默认配置
  data/fixtures/                  # 小型合成校园数据，版本化
  data/tasks.jsonl                # 样例任务与人工设定目标
  src/stage3_agent/               # runner, tools, memory, backend, trace, replay, evaluation
  tests/                          # 离线快速测试
  scripts/                        # Linux 启动脚本，检查不改系统配置
  docs/EXPERIMENT_PROTOCOL.md
  docs/LINUX_RUNBOOK.md
  reports/.gitkeep
  .gitignore
```

`.gitignore` 排除 `.env`、密钥、模型、缓存、SQLite 运行库、大型 `runs/`、临时日志与压缩包。只提交代码、合成数据、协议、少量去敏样例轨迹及小型报告；完整运行工件在 Linux 用户目录保存，并以 SHA256 manifest 归档，供后续发给我审核。README 写明 commit、模型路径如何设定，以及 Windows→GitHub→Linux 的流程。**GitHub 仓库只在 Windows 本地创建与推送，Linux 只拉取与运行，不要求 Linux 端 `gh auth`。**

## 7. Codex 在 Windows 的实施顺序

1. 在本地创建 `stage3-tool-memory-agent`，查看当前目录 Git 状态，避免误修改阶段二仓库；创建上述文件。
2. 实现合成 fixture 与标注任务、工具和持久记忆、模型适配器、轨迹、recorded replay、校验器、指标汇总及 README。Windows 上只运行 `scripted` 的单元/冒烟测试，并故意篡改复制轨迹验证 replay 返回非零。
3. 运行 `git status` 和测试；确认没有权重、账号、token、用户隐私或大型输出被暂存。
4. 若 `gh auth status` 可用，`gh repo create PercyZhong/stage3-tool-memory-agent --private --source . --remote origin --push`。若现有仓库已存在则核对后连接正确 remote；若认证不足，保留本地 commit，并清晰给出用户应完成的登录动作，不上传凭据。仓库 visibility 可按用户既有仓库策略调整，Codex 需记录最终 URL；不得覆盖其他仓库。
5. 给出 Linux 所需的准确命令，实际参数与代码一致；不声称在 Windows 完成了 Linux GPU 实验。

## 8. Linux 运行手册必须包含的步骤

仓库 URL 创建后由 Codex 替换占位符；下列命令仅为范式，**须以实际 CLI 和依赖安装方案为准**：

```bash
cd "$HOME"
git clone https://github.com/PercyZhong/stage3-tool-memory-agent.git
cd stage3-tool-memory-agent
git pull --ff-only origin main
python3.11 -m venv "$HOME/.venvs/stage3-agent"
source "$HOME/.venvs/stage3-agent/bin/activate"
python -m pip install -e .
python -m pytest -q
python -m stage3_agent.run --backend scripted --suite data/tasks.jsonl --arm full --out runs/smoke
python -m stage3_agent.preflight --model-dir /absolute/local/model
```

共享服务器运行前通过 `nvidia-smi` 确认空闲卡；设置 `CUDA_VISIBLE_DEVICES=<单张可用GPU编号>`，不占满所有卡。模型目录变量用 `STAGE3_MODEL_DIR` 等任务专用名称；当预检通过时才启动两种 seed × 三臂的正式运行，逐一完成后 replay + verify + evaluate；命令写入 `docs/LINUX_RUNBOOK.md`。可以在同一个用户目录下复用本地缓存；若 Linux 无 Python 3.11 的 `python3.11` 命令但现有用户环境 Python 3.11 可用，按检查结果使用现有环境并记录实际路径。安装依赖若被服务器网络限制，记录缺少的 wheel/包并停止，不请求管理员修改系统。归档 `runs/`、报告、manifest、环境记录与 ZIP SHA256 到个人目录；不要把权重打包。

## 9. 验收门槛与必须返回的信息

**工程门槛**：Windows 无模型测试通过；Linux 依赖与本地模型预检通过；无权限提升/真实网络工具；记录的异常也形成事件。

**实验门槛**：至少 12 个有效任务、两种 seed、三实验臂完成；任务 ID 和 fixture hash 对齐；每个 run 有独立记忆命名空间和完整配置；有按类型汇总的成功率、证据正确性和失败情况。性能高低不作为“是否完成工程阶段”的硬门槛，如实呈报即可。

**可重放门槛**：对每个正式 run 在独立输出目录无模型重放成功，逐任务最终输出与记忆状态规范化内容一致，`verify` 非零错误数为零；对缺失/篡改事件的负面测试必须失败。至少三类任务各提供可阅读的完整轨迹及逐步说明。若出现不一致，报告差异且不得宣称完成。

交付时向用户返回：GitHub URL、commit SHA、Linux 操作命令、模型真实来源/路径的非敏感描述、环境与依赖版本、运行矩阵和每臂任务完成数、各类指标表、重放核验结果、样例轨迹文件位置、ZIP 与 SHA256、失败和限制。**不要编造 Linux 实验结果；Windows Codex 先交付代码与操作说明，待用户在 Linux 运行后再汇总最终效果。**
