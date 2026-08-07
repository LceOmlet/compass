# DCI 融合 Skill 生成：独立 Grill 文档

状态：Grilling 重新打开；此前未经确认的 JSON 输出合同已撤销
创建日期：2026-08-03
范围：COMPASS proposal-side 的 skill 生成
与既有材料的关系：本文件是独立分析稿，不修改、续写或覆盖原有 grill、`CONTEXT.md` 或 ADR。

## 0. 本轮要解决什么

目标不是把 DCI 当成另一种检索器接到 prompt 前面，而是回答四个问题：

1. 如何从历史 rollout 中发现跨任务、跨 skill 的相同失败机制；
2. 如何从 skill 库和 rollout 库中找到可能修复这些失败的可复用 skill 组件；
3. 除了“找同类失败”和“找组件”，还应让 DCI 执行哪些交互项目；
4. 如何为不同交互项目分配有限预算，而不改写现有父 skill 选择、前沿、准入和谱系隔离语义。

本轮只建立共享定义、边界和决策树。未经逐项确认，不实现算法、不启动实验。

## 1. 已核实事实

### 1.1 原始 DCI 是什么

Direct Corpus Interaction（DCI）首先是一种**高分辨率语料交互接口**：agent 直接对原始 corpus 执行 `find`、`rg`/`grep`、局部读取、管道组合和轻量脚本，根据当前证据反复收窄、交叉核验，而不是先由固定 retriever 返回一个不可继续操作的 top-k 列表。

原始方法天然支持的交互形态包括：语料探索、宽关键词检索、迭代收窄、定点读取、文档内深搜和跨文档比较。[原始 DCI 论文](https://arxiv.org/pdf/2605.05242)，[官方 DCI-Agent-Lite](https://github.com/DCI-Agent/DCI-Agent-Lite)，[官方最小系统提示](https://raw.githubusercontent.com/DCI-Agent/DCI-Agent-Lite/main/prompts/system_prompt.txt)

原始 DCI **不自带**以下能力：

- 跨 rollout 的失败聚类；
- skill 组件抽取器；
- 任务重要性采样；
- Bayesian posterior 或 Bayesian partition；
- 对组件因果效用的估计。

这些若被采用，都属于本项目拟议的 **DCI–skill fusion layer**，不能写成 DCI 原方法已有功能。

DCI 论文中的 `coverage` 和 `localization` 依赖 gold document/evidence，是离线分析指标，不能直接作为在线 acquisition score；否则会发生 oracle leakage。

### 1.2 已知 DCI 失败模式

一手材料已经明确暴露出下列风险：

- 检索式过宽，结果洪泛，搜索不收敛，最后依据表面线索猜测；
- 找对中间实体，却在跨文档或跨 hop 绑定时归因错误；
- corpus 变大后，调用数、延迟和上下文负担快速上升；
- 工作记忆保留过多导致 drift，压缩过猛又丢失中间结构；
- exact surface form 对拼写、别名、变音符敏感；
- 缺少语义或相关度执行先验时，关键证据可能被文件顺序埋没。

DR-DCI 用 agent-callable retrieval 动态扩张持久 workspace；RARG 用 relevance 作为执行先验。二者可为交互调度提供参考，但都不是 Bayesian inference 或统计意义的 importance sampling。[DR-DCI](https://arxiv.org/pdf/2606.14885)，[RARG](https://arxiv.org/pdf/2607.24223)，[GrepSeek](https://arxiv.org/pdf/2605.29307)

### 1.3 当前 COMPASS 已经有什么

当前权威状态已经覆盖：

- accepted skill 文本：`GEPAState.program_candidates`；
- 父子谱系：`parent_program_for_candidate`；
- 出生时 proposal IDs：`program_birth_propose_ids`；
- 每个 skill 的 clean instance scores、实例前沿和前沿 owner；
- raw `F/E` 与默认高分辨率选择分数；
- reversible lineage mask、tie-inclusive top-5 和按选择分数随机抽父节点；
- run trace、evaluation cache，以及 adapter 自有状态。

现有官方 seam 也已经足够承载新的 proposer：`GEPAAdapter.propose_new_texts` 与可选的 `propose_new_texts_batch` 本来就是 proposal owner。DCI 不应在 GEPA 外面再造候选池、准入器、前沿或 selector。

相关代码事实：

- `bridge/b20_compass_reflection.py` 中的 `SparseObservationDspyAdapter` 已经通过官方 adapter proposer seam 生成新文本；
- `upstreams/gepa/src/gepa/proposer/reflective_mutation/reflective_mutation.py` 负责调用 adapter/custom proposer，然后把普通候选交回既有 acceptance/selection 流程；
- `bridge/b19_reversible_parent_selection.py` 已经独立拥有父 skill 的高分辨率/F/E 选择、可逆遮蔽与 top-5；
- `upstreams/gepa/src/gepa/core/state.py` 已经拥有 accepted pool、谱系、出生 proposal IDs、clean scores 和 frontier。

### 1.4 当前状态不等于“完整 rollout 库”

这是本设计目前最重要的事实边界。

`SparseObservationDspyAdapter._observation_facts` 对每个 `(skill, instance)` 只保留与当前最高 reward 原子绑定的一条 observation。它适合 admission reference ownership，但会自然删去低分失败、重复尝试和同一对上的其他 proposal/admission 轨迹。

`evaluation_cache` 主要保存 candidate–instance 的 output/score，不保存完整轨迹。当前轮的 `reflective_dataset` 来自当轮 `B_propose`，但不是全部历史 proposal rollout 的无损、可搜索档案。

因此：

- 现有状态足以做“最佳 observation 对照”；
- 现有状态**不足以**估计失败频率或可靠发现重复失败模式；
- 若用户所说的 rollout 库指“所有历史 rollout”，则需要一个 append-only 的训练期逻辑 rollout corpus，覆盖全部历史 `B_propose` 和已经结束轮次的 `B_admit`；
- 这个 corpus 是 proposer 的历史记忆，不应成为新的 GEPA state、frontier 或 admission 账本。

## 2. 建议的层次边界

当前已确认的架构如下；是否开始实现仍等待整体验收。

```text
官方 sampler 给出与 batch size 无关的 canonical instance 顺序
                |
                v
对一个 instance 执行一个线性化语义事件
  -> 在最新已提交状态上按既有 selector 选择 parent skill
  -> 只用该 instance 执行 B_propose，得到一个 bottleneck seed
                |
                v
单次 DCI subproblem discovery
  -> 搜索整个冻结 rollout corpus；不是当前 batch
  -> 每个 seed 都创建一个新的、不可变的 subproblem
  -> 不比较、归类、合并或重写已有 subproblem
                |
                v
DCI 全库证据集合
  -> 只在 task-level clean frontier evidence 上形成 instance hit
  -> 每个 subproblem、每个 instance 最多一票
  -> F_z = 同类失败 instance IDs；S_z = 同子问题成功对照 IDs
  -> raw score 直接为 |F_z|，不计算 failure rate
                |
                v
一个 instance 对应一个 skill proposal event
  -> seed rollout + subproblem definition
  -> 从 F_z 与 S_z 内按 instance ID 均匀取得 proposer evidence
  -> 通过官方 proposer seam 提出一个普通 skill nominee
  -> A_z = (F_z union S_z) minus existing prospective frontier-ineligible IDs
  -> 在全部 A_z 上执行 admission，不被 batch 截断
                |
                v
既有 acceptance / frontier / state commit（串行）
  -> 更新 skill library
  -> 更新 rollout corpus 和 subproblem library

batch / execution batch
  -> batch 之间串行；每个 batch 冻结一份 skill/frontier/corpus snapshot
  -> batch 内 W 个 seed-instance evolution events 并行
  -> 不截断 DCI corpus 或 A_z；新 subproblem 记录与 GEPA commit 按 canonical 顺序写入
```

必须保持的下游不变量：

- DCI 不参与选择 parent skill；
- DCI 不写 `F/E`、高分辨率 credit、lineage-active、selection-active 或 top-5；
- DCI 不修改实例 frontier、reference ownership、admission 或 acceptance；
- DCI 只通过官方 proposer seam 返回普通 `new_texts`；
- proposer 可以读取 `F_z`/`S_z` 中已有的历史 frontier evidence；nominee 的新 evaluation outcome 仍只能在 nominee 固定后产生；
- 一轮 admission 完成后，其轨迹与结果可以进入下一轮冻结的历史 corpus；
- held-out reporting validation/test 永远不进入 corpus；
- 同一 batch 的 instance events 使用共同冻结的 skill/frontier/corpus snapshot，互相看不到同批新 child；新 subproblem ID/记录和最终状态提交按 canonical instance 顺序写入，网络完成时序不参与算法。

### 2.1 与现有实现的直接连接

这个设计不需要在 COMPASS 外另造一套优化器：

- 官方 `EpochShuffledBatchSampler` 可提供稳定 instance 顺序，但其 `minibatch_size` 当前会直接改变 reflective data，不能冒充纯并发窗口；
- skill library 继续以 `program_candidates`、谱系和 clean frontier 为权威；
- task-level frontier identity 已由 `program_at_pareto_front_valset` 按 instance key 管理；并列 owners 是同一 instance 的多个证据，不是多次 hit；
- proposer 继续通过现有 `propose_new_texts` owner 返回普通 skill 文本；
- 固定 candidate 对多个 instance 的并发评测复用 adapter 的 `evaluate`/`batch_evaluate` 路径，结果按输入位置和 instance ID 对齐；
- admission 仍走现有 reference binding、acceptance、Stage-4 evaluation reuse 和顺序 state commit；
- engine 的顺序 add/commit 继续拥有 skill pool/frontier 更新，不在 bridge 复制 optimizer loop；
- subproblem library 是 proposer/adapter 的持久历史，不复制 GEPA 的 candidate pool、frontier、mask 或 acceptance。

尚未实现的真实增量只有：完整历史 rollout 归档、instance-level subproblem catalog、DCI full-corpus scope resolver，以及让 admission owner 接收动态的全部 `A_z`。后续实现前要先检查官方 owner 是否已有足够的 structured seam；不得另写 frontier 去重、candidate gate、evaluator、acceptance 或 commit loop。

当前 batch 是并发宽度与预算调度单位，而不是 `B_propose` 数据集合、subproblem 定义域、DCI 搜索域或 admission 截断器。batch 内任务共享 pre-batch snapshot，因此增大 batch 可以使 skill tree 更宽、更扁；但它不能缩小任何事件的全库 DCI 范围或完整 `A_z`，也不能让网络完成顺序决定 ID 分配与提交顺序。

### 2.2 已知训练实例上的 post-nomination evaluation

已确认的证据生命周期是：

```text
M_k = batch k 开始前的已提交 skill/frontier/rollout history
for each seed t in batch k, concurrently:
    M_k + seed_t -> full-corpus DCI -> F_z, S_z, definition_z
    seed_t + uniform evidence(F_z, S_z) + definition_z -> nominee_t
    nominee_t -> evaluate on every legally eligible instance in A_z
canonical subproblem append + canonical GEPA commit -> M_(k+1)
```

这里的 admission task IDs 及其历史 frontier evidence 已经参与 DCI，因此它们不是 fresh/held-out instances；fresh 的只是 nominee 在这些实例上的新 rollout/outcome。这个 admission 是训练期全子问题经验改进检查，不证明跨 task generalization。真正的 held-out reporting set 仍永远不进入 proposal、DCI、admission、parent allocation 或 frontier。

## 3. 待采用的核心术语

以下是工作定义，须在 grill 中确认后才能进入 `CONTEXT.md`。

### 3.1 Failure signature（失败签名）

不是“题目看起来相似”，也不是简单的低分标签。它是一个带 provenance 的、可跨任务比较的失败机制描述，至少区分：

- 任务要求或约束；
- rollout 试图完成的子目标/动作；
- 最早出现的决定性偏差；
- 被违反的约束或 evaluator 反馈；
- 最终 observable outcome；
- 支持它的 rollout IDs 与反例 IDs。

同一 rollout 可以属于多个失败假设；不强迫互斥硬聚类。由 LLM 归纳出的签名是**假设**，不是 ground truth。

### 3.2 Skill component（skill 组件）

不是从某个答案复制的 task-specific fact。它是 skill 文本中最小的、可复用的行为规则、检查步骤、工具策略、分解方式或终止条件，并带有：

- 来源 skill、文本 span/hash 和谱系；
- 支持它的成功/失败对照；
- 适用失败签名；
- 已知反例、冲突组件和禁用条件；
- 被组合进哪些 nominee 的记录。

“可能有用”只表示 proposal hypothesis；最终仍由既有 admission 机制决定候选是否进入 pool。

### 3.3 Interaction project（交互项目）

一次 interaction 不是“随便再搜几条”，而是一个可审计的假设消歧任务：

```text
项目类型 + 目标失败签名/组件 + 冻结 corpus snapshot
+ 查询/命令轨迹 + 返回记录 + 正证据 + 反证据
+ 未解决约束 + 成本 + 下一动作
```

### 3.4 Bottleneck event（rollout 堵点）

一个堵点不是整个 batch 的摘要，也不是“这题得分低”这一标签。它是单条逻辑 rollout 中带定位证据的事件：agent 正在尝试某个必要中间目标，却在某一步开始违反必要条件、停滞或走入此后未恢复的错误分支。最小记录为：

```text
task_id + rollout_id + event span/step
+ attempted intermediate goal + violated/unsatisfied condition
+ observable consequence + later recovery/non-recovery evidence
```

DCI 从当前 canonical instance event 新产生的堵点出发，但必须回到整个冻结历史 corpus 搜索同类、反例和成功对照。execution window 中还有多少其他 instance 与本次定义无关。堵点是待核验的 operational hypothesis，不自动等于因果瓶颈。

### 3.5 Subproblem（子问题）与重叠发现集合

subproblem `z` 是从跨 rollout 堵点聚合得到的能力区域，至少包含：

```text
definition_z
+ membership predicate M_z(task, rollout evidence)
+ crossing predicate C_z(complete rollout, step)
+ positive/negative exemplars + provenance
+ corpus snapshot on which the definition was induced
```

当前 subproblem library 是每次独立 seed DCI 产生的、允许相互重叠的发现集合，不宣称形成唯一近似区划。每个新 seed frontier rollout 触发一次 DCI，并无条件创建一个新的 subproblem identity；已有定义、文本相似和 instance 重叠都不触发自动归类或合并。每个 subproblem 的定义、成员与元数据在创建时冻结。split/merge、定义重写和 batch 内聚类均不进入当前设计。

DCI 对全部历史 rollout 做证据搜索，但排序 hit 只在当前 clean frontier snapshot 上按 unique instance ID 计数。成功/失败对照和其他历史 rollout 可进入 proposer evidence，不额外增加 raw failure count。

## 4. 候选交互项目

### P1. 相同失败检索

从当前 `B_propose` 的失败出发，搜索具有相同 observable breakdown 的历史 rollout，并同时寻找近邻与反例，防止只按任务主题收集 evidence。该搜索形成本次新 subproblem 的成员集合，不负责与历史 subproblem 归并。

### P2. 成功–失败对照

优先寻找：同一 instance 不同 skill、同一 skill 不同 instance、同一失败签名下成功与失败的配对。目标是定位“哪一段行为发生了变化”，而不只是收集更多相似失败。

### P3. 组件来源检索

在 accepted skill、历史 proposal 文本和成功对照中定位可能修复目标失败的最小组件，并保留原始 span 与来源谱系。

### P4. 组件冲突与负迁移检索

在组合前主动搜索反例：某组件是否在另一类任务上诱发过度约束、错误工具选择、过早终止或更长轨迹。没有反例搜索的组件不能被标成“已验证有效”。

### P5. 组合兼容性检索

检查多个组件是否重复、矛盾、存在顺序依赖或共同占用上下文预算；输出最小组合，而不是把检索到的所有规则堆进 skill。

### P6. 新失败/覆盖缺口探索

为未被现有 failure signatures 解释的 rollout 保留探索预算，防止高频失败长期垄断交互。

### P7. 归因核验

专门检查“找到正确中间实体但最终绑定错对象”的 DCI 式跨 hop 错误；要求每个结论可回指具体 rollout span，而不是只给摘要。

### P8. 查询质量修复

检测 broad-query drift、词形脆弱、别名漏检、过度 truncation/compaction；必要时改变检索表达式或 workspace，而不是把“没搜到”解释成“语料里没有”。

## 5. 实例级 DCI、proposal、admission 与 execution window

### 5.1 不得混在一起的三个 owner

1. **父 skill 选择**：已有高分辨率/F/E、可逆遮蔽、top-5 和比例采样负责；本设计不碰。
2. **DCI subproblem/evidence scope**：每个 seed instance 启动一次逻辑 DCI，搜索全 corpus，创建一个新 subproblem，并形成 `F_z`、`S_z` 和 proposer evidence；它不判断与任何历史 subproblem 是否相同。
3. **candidate admission/state commit**：既有 admission/reference/acceptance/frontier owner 评测全部 `A_z` 并顺序提交；DCI 不复制这些状态转换。

execution window 只限制上述 owner 内部可交换的实例请求并发量，不成为第四个算法 owner。

### 5.2 直接计数的 DCI 集合

对 canonical seed instance `p_t`，先冻结当前已提交的 rollout corpus、clean instance frontier 和 subproblem catalog。DCI 搜索整个冻结 corpus，但 `raw_count_z` 只在每个 instance 的 task-level clean frontier evidence 上计票：

```text
F_z(t) = {unique instance_id:
          该 instance 的 clean frontier evidence
          被本次 DCI 判为 subproblem z 的同类失败}

S_z(t) = {unique instance_id:
          该 instance 的 clean frontier evidence
          被本次 DCI 判为 z 的成功对照}

raw_count_z(t) = |F_z(t)|
```

实验直接报告 `raw_count_z`；它是 subproblem coverage 的诊断统计量，不参与在线子问题选择。每个 seed 的 DCI 返回哪个 subproblem，就立即处理该次结果一次，不建立 subproblem top-k、优先队列或额外调度器。不计算失败率，也不做“失败率 × 数量”的乘法。DCI 可以读取一个 instance 的多个并列 frontier-owner rollout 作为证据，但计数 key 始终是 `(subproblem_id, instance_id)`，因此每个 subproblem 的 raw count 不超过合法 instance 总数。现有 `program_at_pareto_front_valset` 已按 instance key 保存 clean frontier owners，应复用该身份结构；不能另造 span/rollout 级计数器。

timeout、TPM/隧道、parser 故障、缺失 frontier observation 或 DCI 无法判断都不进入 `F_z`。它们沿用各 owner 已有的失败/缺失语义，不能为 DCI 新增一个 gate。

### 5.3 单实例 subproblem discovery

每个 seed instance 只执行一次 discovery：

```text
当前 frontier rollout -> DCI 搜索冻结 rollout/skill corpus
                       -> 创建一个新 subproblem
                       -> 冻结 definition、成员与元数据
```

这一步以 instance 为单位。batch 内的 `W` 次逻辑 DCI 读取同一份冻结 rollout/skill corpus，并可并行执行；它们彼此不做 pairwise 比较。DCI 返回后，`W` 个新 subproblem 按 canonical instance 顺序分配稳定 ID 并追加到 catalog。这样一个含 `W` 个 instances 的 batch 恰好有 `W` 条 seed rollouts、`W` 次逻辑 DCI 和最多 `W` 个完整的新 subproblems，而不是 `O(W^2)` 次比较。

不同 DCI discoveries 即使自由文本相近或成员重叠，也保持不同 identity。重叠 instance 可以分别为各自 subproblem 贡献一票，但在同一个 subproblem 内仍按 unique instance 去重；不存在跨 subproblem 的共享计数 owner。

### 5.4 proposer evidence 与固定预算 DCI admission

对确定的 `z`：

```text
proposer evidence
    = seed rollout
    + subproblem definition
    + 从 F_z 中按 unique instance 均匀取得的失败 evidence
    + 从 S_z 中按 unique instance 均匀取得的成功 evidence

A_z = (F_z union S_z) minus X_t
B_admit = 从 A_z 无放回抽至 3 个；不足部分从全局合法 admission pool 无放回补齐
```

其中 `X_t` 直接复用现有 `state.get_prospective_frontier_ineligible_ids(parent, birth_propose_ids)`：它至少包含当前 seed propose instance，并保持已存在的递归 proposal-lineage exclusion。不要在 DCI 层重写这套排除关系。

batch 中每个 instance 产生的 seed rollout 都以概率 1 进入它自己的 proposer evidence，无论它是否成为该 instance 的最高-reward observation。seed 不占 `k_failure` 或 `k_success`，也不能因 max-bound observation 更新而被删掉。它是否另外进入 raw frontier hit 统计仍由既有 clean-frontier owner 决定，不能因为是 seed 而重复计数。

内层 proposal minibatch 的其余 evidence 槽位分别独立抽样。当 success/failure 两侧都有可用 instance 时，每个槽位先以 `0.5/0.5` 选择 complete-success 或 unresolved/failure，再在所选集合的 unique instance IDs 内均匀抽一个尚未用于本 proposal 的 instance。因此 minibatch size 为 3 时，seed 固定占一个位置，另外两个随机槽位中的成功样本数服从 `Binomial(2, 0.5)`：两个成功、成功失败各一、两个失败的概率分别是 `0.25/0.5/0.25`。期望上 `k_success=k_failure=1`，但不固定为各一个。

若一侧没有任何可用 instance，则不再抛类别硬币，剩余 evidence 槽位全部从另一侧均匀无放回取得。特别地，没有 success 时全部使用 failure。若两侧都没有额外可用 instance，则只跳过当前 seed event 的 skill proposal 与 admission，并累计 `proposal_skipped_no_dci_evidence`；同一外层 batch 的其他 seed events 不受影响。这个分支是 proposer 输入不足，不是新 admission gate，也不能导致整批回滚。

预算只统计实际新执行的 instance rollouts。所以上述跳过发生前已经生成的 fresh seed rollout 仍计数；历史 corpus 读取、对已存 rollout 的 DCI 检索、cache hit、proposal 文本生成事件本身，以及根本没有执行的 admission instances 都不计 rollout budget。若 candidate 进入 admission，则每条实际新执行的 admission rollout 分别计数，不能按计划集合大小或整批成败补账。

官方 proposer 用这些 evidence 提出一个 skill candidate。candidate 随后只在固定大小为 `3` 的 `B_admit` 上 admission。若合法 `A_z` 至少有 3 个成员，则从中均匀、无放回抽 3 个；若不足 3 个，则保留全部合法成员，并从全局合法 admission pool 均匀、无放回补齐。补齐实例不成为 `A_z` 成员。固定 candidate 在 `B_admit` 内的实例 rollout 可以并行，结果按 instance ID 稳定归并，再交给既有 reference、acceptance 和 commit owner。

由于 `A_z` 的历史 frontier evidence 已参与 proposal，这个 admission 是训练期 empirical admission，不是 fresh-instance generalization test。它仍是良定义的 candidate-vs-history-fixed-reference 比较；真正 generalization 只由永不进入 DCI 的 held-out reporting set 衡量。

### 5.5 batch 的合法语义：batch 间串行、batch 内并行

语义状态写为：

```text
X_k = (skill pool/lineage, clean frontier, rollout corpus,
       subproblem catalog, RNG/budget state)
```

batch `k` 开始时冻结 pre-batch skill/frontier/corpus snapshot。该 batch 的 `W` 个 canonical seed instances 从这份 snapshot 形成各自的 parent rollout、逻辑 DCI、proposal evidence、nominee 和完整 `A_z` admission；这些实例级工作可以并行，同批 nominee 互相不可见，因此可以形成兄弟分支。batch `k+1` 只能在 batch `k` 全部结束并提交后开始。

batch 内有两个必须按 canonical instance 顺序写入的状态点：

- 每次完整 DCI discovery 的新 subproblem ID 与冻结记录；相近或重叠发现也分别追加，不做冲突归并；
- admission 结果的 GEPA state commit：reference/acceptance 使用各事件已经冻结和绑定的证据，但 skill/frontier/corpus 写入不按网络完成顺序抢占。

并发可以覆盖 seed rollouts、全库 DCI 交互、skill proposal 调用，以及固定 nominee 在全部 `A_z` 上的逐实例 evaluation。结果必须按 batch position、instance ID 和 proposal ID 稳定归并。batch size 可以通过共享 snapshot 改变树的宽深权衡，这是明确的算法/资源参数；它不得改变单个事件的 DCI 搜索总体、`F_z/S_z` 计票规则、完整 admission scope 或 canonical commit order。

当前 GEPA 的 `IndependentSampling(n)` 已具备“同一 pre-commit state 上形成多个 proposal tasks”的批语义，但 `reflection_minibatch_size` 仍把多个 instances 合成一个 reflective dataset。DCI 路径需要保持一 instance 一 seed、一 DCI、一 proposal event，同时复用官方批量 proposal/evaluation 和后续 acceptance/commit owner。

### 5.6 计算量边界

每个 batch 的 instance 数、seed rollout 数和逻辑 DCI 数完全相同；总的底层计算量不能同时固定：

- full-corpus DCI 的朴素扫描成本随 rollout corpus 增长；
- 全 `A_z` admission 的 rollout 数是 `|(F_z union S_z) minus X_t|`，随 subproblem coverage 变化；
- 一个 proposal 的成功/失败 evidence 读取量也受上下文预算影响。

可以复用 DCI/上游索引、cache、adapter 并发评测和 stable corpus version 来加速，但这些优化必须保持同一 hit set 和 `A_z`。若用 batch cap 截断 admission，算法就不再是“在全部 DCI tasks 上 admit”。

### 5.7 importance sampling 与 Bayesian 当前不进入算法

当前 raw score 是同类失败 instance 的直接经验计数，不估计总体 failure rate；因此不需要 importance correction 或 Bayesian shrinkage。若未来只返回 DCI top hits、非均匀丢弃 corpus instance，或声称估计未观察总体，才需要重新讨论；本轮不引入。

## 6. 必须防止的伪收益

- **max-only censorship**：只看到最高分 observation，误以为某 skill 没有失败；
- **survivor bias**：只搜 accepted skill，错过 rejected proposal 中有用或有害的组件；
- **lineage echo chamber**：多个后代继承同一段文本，却被当作多份独立支持；
- **duplicate retry bias**：同一逻辑 rollout 的网络重试被当成多个失败；
- **infra/semantic conflation**：timeout、隧道、TPM 或 parser 故障被归为能力失败；
- **topic clustering**：把表面主题相近误当成同一失败机制；
- **adaptive cherry-picking**：只展示支持预期组件的搜索结果；
- **component confounding**：skill 同时改了多个组件，却把收益归到其中一个；
- **future/current-admit leakage**：并行任务或当前 `B_admit` 结果在 nominee 固定前进入 proposer；已结束轮次的 admission 不是该类泄漏；
- **test leakage**：测试实例或其派生证据进入 corpus；
- **query-order artifacts**：文件顺序、截断或 compaction 决定“证据不存在”；
- **batch-local universe**：只在当前 execution window 的有限任务里做 DCI 或 admission；
- **snapshot siblings**：多个 proposal 从同一旧 skill state 并发产生，batch 越大树越扁；
- **silent semantic merge**：把不同 DCI discoveries 因文本相似或 instance 重叠而静默合并，伪造不存在的同一性；当前设计明确允许重复或重叠 subproblems；
- **completion-order commit**：谁先返回就先写 skill/frontier/catalog，使网络时序改变算法；
- **frontier-owner overcount**：同一 instance 的并列 frontier owners 或多个 span 被算成多个 DCI hits；
- **hidden admission truncation**：用 batch cap 截断 `A_z`，却仍称为全 DCI admission；
- **fixed-cost illusion**：把固定 seed 数误写成固定 metric-call 成本，掩盖 full-corpus DCI 与全 `A_z` admission 的可变成本；
- **skill bloat**：组件只增加文本长度，没有改变可观测决策。

## 7. 不新增审计框架

本轮不实现独立 provenance ledger、审计事件总线或第二套 GEPA state。复现所需信息只沿既有 owner 保存：DCI-Agent-Lite/Pi 保留官方运行产物；GEPA 保留 candidate、lineage、frontier、acceptance 与 metric-call state；COMPASS adapter 仅保存其语义必需的 append-only logical rollout corpus、subproblem catalog、frontier-fact 绑定和无 evidence 跳过计数。任何额外记录都必须先由真实复现缺口证明，不能以“可能以后要审计”为由提前加入。

## 8. 决策树（一次只推进一个节点）

当前节点：D6b 表示语义已闭合，等待整体验收后进入实现。D1--D7 的算法语义与 pinned DCI-Agent-Lite/Pi runtime owner 继续有效；严格 JSON、自定义提交工具、文本 parser、existing/new 分类和自动归并均已否决。DCI 检索结果的自由文本就是每次新 subproblem 的不可变 definition；selection workspace 给出其成员，seed 是隐式成员；固定成员、instance 数量和其他创建元数据一并冻结保留。timeout/运行失败完全复用 GEPA 已有 proposal 失败逻辑，不新增 DCI failure state、计数器或 gate。

- D1：rollout corpus 的内容边界与保留语义；
- D2：batch 中每个 instance 对应一条 seed rollout 和一次逻辑 DCI；底层全库搜索与全量 admission 成本可变；
- D3：每个 seed 的 DCI 独立创建一个新 subproblem；batch 内 DCI 并行，新记录按 canonical 顺序追加，不做归类或归并；
- D4：成功/失败 evidence 的均匀采样数量与空 strata 语义；
- D5：全 `A_z` admission 继续复用既有 reference binding、acceptance 与顺序 commit owner，不新增 gate；
- D6：已确认复用 pinned DCI-Agent-Lite/Pi；
- D7：已确认只补动态 admission scope 的最小 upstream seam，不新增 instance-event scheduler；
- D8：离线评测与论文消融如何同在线进化隔离。

## 9. D1：已确认——渐进式训练期 rollout corpus

结论：DCI 使用 append-only 的全部训练期历史逻辑轨迹，而不是复用当前 max-bound observation facts。

确认边界：

- 保存所有历史 `B_propose` 的成功、任务失败、格式失败，以及真正超时前已经安全获得的可用轨迹片段；
- 一轮关闭后，保存该轮 `B_admit` 的 nominee/reference 轨迹、reward 差异和 admission outcome，供后续轮次挖掘；
- 网络重试/TPM keepalive 以 logical rollout group 去重，不把基础设施事件当任务失败；
- accepted skill 仍以官方 `program_candidates` 为权威；
- rejected proposal 文本另作只读 proposal archive，以便找反例和避免 survivor bias，但绝不加入官方 candidate pool；
- DCI 可以读取全部已提交历史 frontier/admission evidence，并据此确定当前 `A_z`；不能读取尚未生成的 nominee rollout/outcome；
- 训练期 admission pool 与最终 held-out reporting validation/test 是不同集合；后者及其任何派生信息永不进入 DCI 或优化；
- 每个 instance event 开始时冻结 snapshot，当前 nominee 的 admission 结果只能在该 nominee 固定后产生，并在本 event 串行提交后进入后续历史。

理由：重复失败发现需要保留低分与被覆盖的历史轨迹；复用 max-bound facts 会系统性删掉这些信息。历史 admission 又包含 candidate/reference 对照与准入结果，是后续组件挖掘的重要材料。最新定义不把 `A_z` 当 fresh instances；只要求 nominee 的新 rollout/outcome 在 nominee 固定前不存在，因此不会用自己的评测结果生成自己。

## 10. D2：已确认——batch 内一实例对应一 seed、一 DCI

当前定义下，batch 之间串行，batch 内 seed-instance evolution events 并行，并共享一份 pre-batch skill/frontier/corpus snapshot。DCI 对每个 seed 搜全 corpus；raw score 直接是同类失败 unique instance 数；proposer 从成功/失败集合均匀取证据；candidate 在全部合法 DCI success/failure instances（复用现有 proposal-lineage exclusion）上 admission。batch 不切分这些集合。

因此 batch 含 `W` 个 instances 时，固定为 `W` 条 seed rollouts、`W` 次逻辑 DCI 和最多 `W` 个相互不可见的 proposal events。新 subproblem ID/记录追加与最终 state commit 按 canonical instance 顺序串行；其余实例工作和每个 candidate 的完整 `A_z` evaluation 可以并行。总 rollout/metric calls 会随 corpus 与 `A_z` 大小变化。

## 11. D3：已确认——每个 seed DCI 都新建 subproblem

每次逻辑 DCI 搜索全 corpus 后，编排器都提交一个新的 subproblem identity、自由文本 definition、固定成员集合和冻结元数据。batch 内这些 discoveries 读取相同冻结 rollout/skill corpus 并行完成；新记录按 canonical instance 顺序追加。历史 subproblem catalog 可以保持在模型可见的 workspace 中作为可选上下文，但 prompt 不强制读取，也不要求模型判断 existing/new、identity 或 merge。当前路径不做 batch 内 pairwise 聚类，也不自动合并相似或重叠 discoveries。

## 12. D4：已确认——0.5/0.5 类别抽样与无样本回退

已确认：

- 外层 batch 只是互不作为彼此数据切片的 seed-instance 并发批次；有 `W` 个 instances 就有 `W` 条 seed rollouts 和 `W` 次逻辑 DCI；
- 内层 minibatch 是单个 seed 的一次 skill-evolution 输入；
- 每条 seed rollout 与 subproblem definition 始终加入自己的 proposal，不受最高-reward 过滤；
- 若 instance 的 frontier reward 低于 evaluator 明示的最高可得 reward，它仍未完全成功，仍有堵点；只有达到明示满分才属于 complete success。没有明示 reward ceiling 时不能靠猜测判定“仍可提高”；
- seed 固定进入；两侧都有可用 instance 时，其余每个 evidence 槽位独立以 `0.5` 选择 success、以 `0.5` 选择 failure，再从所选集合内按 unique instance 均匀无放回抽样；
- 一侧为空时，剩余槽位全部使用另一侧；两侧都没有额外可用 instance 时，只跳过当前 seed 的 proposal/admission，并累计 `proposal_skipped_no_dci_evidence`，不影响同 batch 其他 seed。

因此 minibatch size 为 3 时：

```text
1 个固定 seed + 2 个独立的 Bernoulli(0.5) evidence 槽位
```

两个随机槽位都选 success、各选一种、都选 failure 的概率分别为 `0.25/0.5/0.25`；所以 success/failure 的期望数量相同，但不强制每个 minibatch 各一个。外层 batch 中所有 seed events 使用同一个抽样规则，batch 分组不改变概率。

`0.5/0.5` 只适用于两侧均可选的正常分支；单侧为空后的确定性回退必须单独记录，不能伪报成一次平衡抽样。它仍不保证具体 instance 的边际概率跨不同大小的 `F_z`、`S_z` 相同。seed 入选概率为 1，也不属于类别抽签。

预算语义已确认：只计算 fresh rollouts。发生 `proposal_skipped_no_dci_evidence` 时，已执行的 fresh seed rollout 计数；未执行的 proposal/admission rollout 不计；DCI 对历史 corpus 的检索和 cache reuse 不计。

## 13. D5：已确认——不重开 admission 设计

固定大小 `B_admit` 的执行继续交给已有 GEPA/COMPASS owner：复用现有 prospective proposal-lineage exclusions、history-fixed reference binding、已有 acceptance policy、Stage-4 result reuse 和顺序 state commit。DCI hook 只在官方 post-proposal admission seam 内从 exact subproblem member IDs 选取并按规则补齐，不新增 admission gate、第二套 reference、frontier 或 commit loop。

## 14. D6：已确认——复用 pinned DCI-Agent-Lite/Pi

### 14.1 本地事实

当前 pinned COMPASS/GEPA workspace 中没有 DCI runtime、subproblem search engine 或 corpus-interaction agent；只有本 grill 文档和术语定义。因此下一步确实应该寻找并审计上游方法，而不是继续在本地发明一个名字叫 DCI 的检索循环。

### 14.2 官方候选审计

| 候选 | 官方已提供 | 与当前合同的关系 | 当前结论 |
|---|---|---|---|
| DCI-Agent-Lite | 对原始 corpus 使用 `rg`、`find`、read/bash 的零索引交互；MIT；程序运行保留 `events.jsonl`、state、完整 conversation 与 final output | 唯一直接保持 raw-corpus interaction 的公开基线；但实现是 Python runner 调用 pinned Pi/Node agent，默认 owner 是“回答问题”，没有直接返回 COMPASS 的成员集合 | 第一候选；先做只读兼容性 probe，复用其 runtime/trajectory owner；自由文本保持官方输出，成员通过已确认的 selection workspace 表达 |
| DR-DCI | retriever 动态把文档扩展到持久 workspace，以提升大 corpus 的效率与稳定性 | retriever 若未访问某些 rollouts，就可能改变全 corpus hit set 和 raw count | 不作为第一版语义 owner；仅可在未来证明不改变 exact hit set 后作为扩展 |
| RARG | 用 relevance 排文档遍历顺序、初始化入口并 rerank grep matches | 若只改变完整遍历顺序可作为优化；若截断或隐藏低相关 hit，会改变计数 | 暂作后续 execution-order 优化候选，不承担 membership 语义 |
| GrepSeek | 训练一个紧凑 DCI search agent | 引入额外训练目标、checkpoint 和策略，不是当前缺失的最薄接口 | 不作为首版集成；可作为未来 proposer/search-agent 研究变量 |

一手来源：[DCI-Agent-Lite 官方仓库](https://github.com/DCI-Agent/DCI-Agent-Lite)、[DCI 原论文](https://arxiv.org/abs/2605.05242)、[DR-DCI](https://arxiv.org/abs/2606.14885)、[RARG](https://arxiv.org/abs/2607.24223)、[GrepSeek](https://arxiv.org/abs/2605.29307)。

### 14.3 integrate-upstream-first 边界

第一步不是复制 DCI-Agent-Lite 的 agent loop，而是固定其官方仓库与 Pi 依赖版本，验证它能否在只读 rollout corpus 上：

1. 接收一个 seed rollout、冻结 rollout/skill corpus 和稳定 `DataId`；
2. 使用官方 terminal interaction runtime 搜索整个 corpus；
3. 保留官方自由文本结果，并通过隔离 selection workspace 明确本次新 subproblem 的 success/failure `DataId`；
4. 保留 exact tool trajectory 和 instance provenance；
5. 允许多个 seed runs 隔离输出目录并发，而不共享可变 agent session。

若官方 seam 能表达这些结果，COMPASS 只做 representation adapter；若不能，应先形成缺口证据，再决定最小 owner patch，不能直接重写一个本地 DCI。

已确认：采用 **DCI-Agent-Lite 作为首个官方 runtime/trajectory owner**；Pi 使用该仓库当前依赖分支的精确提交。DR-DCI、RARG、GrepSeek 和 Pi-Serini 只保留为论文对照或未来研究候选，不进入首版运行依赖，也不为它们提前建立兼容层。

### 14.4 只读兼容性核对结果

核对的精确版本：

- DCI-Agent-Lite：`271f37e71f053bf0c99c05ce6d2fb53b841d922e`；
- Pi `codex/context-management-ablation`：`a6be5eb4cce278de31ac05792af3dfc0883215dc`。

官方 owner 已经覆盖：

- Pi RPC 子进程的启动、事件读取、abort/stop 和 stderr 处理；
- `--cwd`、工具集合、system prompt、question 与独立 `--output-dir` 输入；
- `events.jsonl`、`state.json`、`conversation_full.json`、`final.txt` 等逐事件 artifact；
- 默认 `--no-session` 的独立运行，以及非空输出目录拒绝误覆盖。

因此 COMPASS 不实现第二套 agent loop、RPC reader、artifact recorder、resume、工具调度、重试或并发 session owner。

官方输出的表示事实是：`final.txt` 是自由文本，官方 DCI-Agent-Lite 没有要求模型生成 COMPASS `(existing/new subproblem, F_z, S_z, evidence, components)` JSON。此前在 system-prompt/final-output seam 强制“恰好一个 JSON object”、再由 bridge 严格解码的方案未经 grilling，现已撤销。首版不得要求 DCI 模型生成 JSON，也不得用自定义提交工具、代码围栏抽取、修复 parser 或另一种固定格式悄悄恢复同一要求。自由文本与官方 tool trajectory 如何映射到 subproblem、evidence scope 和 proposer 输入，重新进入 D6b grilling。

官方 `--cwd` 只设置进程工作目录，不构成文件系统边界；Pi 的 read path 接受绝对路径和 `..`，bash 也运行本地 shell。由于 held-out/secrets 不可见是已确认的数据合同，首版运行必须让 DCI 进程实际上只能看到冻结的允许 corpus，而不能仅依赖 prompt 或 cwd。应优先使用现有进程隔离能力；若运行环境没有该能力，才通过 Pi 官方 tool interception/operations seam 做最小只读路径约束。它只保护输入边界，不承担 DCI 语义、proposal、admission 或 acceptance，也不扩展成通用安全框架。

DCI-Agent-Lite 的 `setup.sh` 只 checkout 浮动 Pi 分支，因此 COMPASS 必须用既有 upstream lock 记录上述两个 commit；不另写依赖管理器。

## 15. 2026-08-03 前沿文献复核与接入判决

本节补充 D6 的一手文献和官方代码审计。若与 14.2 的初步判断冲突，以本节为准。

### 15.1 总判决

DCI **适合作为 COMPASS 提出侧的高分辨率语料交互器**：它擅长从已有线索出发，在原始 rollout/skill corpus 中反复定位、比较和核验具体证据。上一版“DCI 不能直接承担 subproblem、失败计数或 admission”的说法过强，应撤回。当前设计中，DCI 可以且应当承担 **每次独立 subproblem discovery、同类 instance 发现以及 admission scope 的形成**；它不承担跨 discoveries 的 identity 判断或自动归并，也不替代既有父 skill selector、fresh candidate/reference evaluation、acceptance 和 GEPA state commit。

目前证据支持的强结论是：交互式、可组合、细粒度的 corpus 操作能显著改善证据定位与核验。证据不支持“无索引的全库 grep 在所有规模和任务上都优于调好的 retrieval”。这个更强说法同时被语料扩展成本、词面盲区、harness 差异和强 BM25 基线削弱。

首版应采用：

1. DCI 从一个 seed bottleneck 创建一个新的 subproblem 假设，并在冻结 clean-frontier snapshot 上形成带证据的 instance-level `F_z`/`S_z`；
2. COMPASS bridge 只校验稳定 `DataId`、按 instance 去重并计算 `|F_z|`，不再设置第二个语义分类器；
3. `A_z=(F_z∪S_z)−X_t` 就是 DCI 负责形成的 admission scope；既有 COMPASS/GEPA owner 继续负责 scope 内的 fresh evaluation、reference binding、acceptance、frontier 与 commit；
4. 加速只能改变搜索执行顺序，不能改变可见 corpus、最终 hit set 或计数；
5. 用同模型、同 harness、同预算的深 BM25/retrieval 作为必需对照，避免把 harness 收益误写成 DCI 收益。

### 15.2 论文与官方实现审计

| 工作 | 最可信的一手证据 | 主要局限 | 对 COMPASS 的判决 |
|---|---|---|---|
| [DCI](https://arxiv.org/pdf/2605.05242) / [DCI-Agent-Lite](https://github.com/DCI-Agent/DCI-Agent-Lite) | 同一 Claude Sonnet 4.6 在 BrowseComp-Plus 上，Qwen3 embedding retrieval 为 69.0%，DCI 为 80.0%，估算成本下降 29.4%；收益经常来自在已找到文档内继续定位与核验，而不只是 gold-document recall | 2026 预印本；部分 QA 只抽 50 条；模型、judge 与成本估计会漂移；raw full-corpus 规模扩展差；官方 Lite 是 Python runner 调 pinned Pi/Node，不是稳定嵌入式 library | **首个 runtime/trajectory owner 候选**；先做 pinned-commit probe，只复用 agent/tool/artifact seam，不复制 agent loop |
| [Is Grep All You Need?](https://arxiv.org/pdf/2605.15184) | 固定 corpus/model 后，harness 和工具交付方式本身可带来与检索方法相当的分差；某些 file/provider 组合会消除或反转 lexical 优势 | LongMemEval-S 仅 116 条且领域单一 | DCI 对比必须固定 harness、上下文管理和工具可见性，不能把整套 orchestration 的差异归给 DCI |
| [Pi-Serini](https://arxiv.org/pdf/2605.10848) / [官方实现](https://github.com/justram/pi-serini) | 调优 BM25、加深检索并配合强 agent loop，在 BrowseComp-Plus 全集达到 83.1%，surfaced-evidence recall 94.7%；depth/configuration 可支配结果 | 有界 retrieval，不是 full-corpus DCI | **必需强对照与 provenance/runtime 参考**；不替代 exact DCI membership owner |
| [GrepSeek](https://arxiv.org/pdf/2605.29307) / [官方实现](https://github.com/alirezasalemi7/grepseek) | Tutor/Planner 生成可核验轨迹，SFT+GRPO 训练 9B DCI agent；sharded-parallel search engine 报告最高 7.6x 加速并测试 byte equivalence | 若干 QA 集回退；作者明确指出 paraphrase、变音符和 lexical mismatch；训练策略引入新变量 | 不引入 SFT/GRPO；仅把 `parallel_search` 当**可能的等价执行优化**，须在本项目布局通过 byte-equivalence 和依赖 probe |
| [DR-DCI](https://arxiv.org/pdf/2606.14885) | agent-callable retrieval 动态扩张持久 workspace；BrowseComp-Plus 全集上 raw DCI 62.90%/3139.10s，DR-DCI 71.20%/146.16s；workspace-preserving reset 可到 73.25% | workspace 会隐藏未拉取文档；随意 reset 会回退；未找到官方代码 | 证明 raw DCI 的规模瓶颈；首版**不采用 workspace 作为 membership universe**，未来只能作 discovery 加速并补完整性物化 |
| [RISE](https://arxiv.org/pdf/2606.06880) / [官方实现](https://github.com/texttron/RISE) | BM25 top-K 构造局部可交互 workspace；100K 语料上以约四分之一成本匹配 pure DCI，1M 时比 raw DCI 更稳定 | top-K 改变可见语料；官方仓库很新；部分结论基于 100-query 子集 | 可做 scaling baseline；不能作为首版 exact `F_z`/`S_z` owner |
| [RARG](https://arxiv.org/pdf/2607.24223) / [官方实现](https://github.com/LeqsNaN/RARG) | relevance 用作 `rg` 遍历、段落入口和局部 match reranking 的执行先验；深度型 QA 上减少工具调用 | BRIGHT 上局部 rerank 降低 breadth-first recall；依赖 embedding；官方实现是 Pi agent 的 Python重写 | 只借用“相关度改变遍历顺序而非资格集合”；不复制 harness，不允许 hard cutoff 参与 hit count |

### 15.2.1 与 COMPASS 上千条数据的量级比较

必须区分 **corpus 记录数** 与 **DCI query/seed 数**：

| 工作 | corpus 规模 | 实际 query/evaluation 规模 |
|---|---:|---:|
| 原始 DCI BrowseComp-Plus | 100,195 篇文档，平均 5,179 words；扩展实验到 200K/400K | 主结果完整 830 问；扩展与多数 ablation 为固定 100 问 |
| 原始 DCI Wikipedia QA | 21,015,324 个约 100-word passages | Bamboogle 使用完整集，其余多数 QA 数据集随机 50 问 |
| 原始 DCI BRIGHT/BEIR | 5,183–121,249 篇 | 四个 BRIGHT 域共 423 问；ArguAna/SciFact 各抽 50 问 |
| DR-DCI | 100K→10M，并另测约 20M Wiki-18 | BrowseComp 主结果 830 问；规模实验固定 100 问 |
| RARG | 100K→1M | BrowseComp 固定 100 问，另测四个 BRIGHT 域 |
| GrepSeek | 21M 行 Wikipedia | 169,615 条 RL 训练样本，最终评测 51,713 问 |

因此，若 COMPASS 当前是**上千个 instance、累计几千到数万条 rollout 记录**，它在 corpus I/O 维度远小于上述 DCI 工作，并不算多。`rg` 扫描本身不应成为首要风险，首版没有证据需要 DR-DCI/RISE 式 hard retrieval scope。

真正可能很大的有两个量：

1. 每个 seed 都启动一次多轮 DCI agent 时，总 LLM/tool 轮数约随 seed 数增长；一千个 seed 已与完整 BrowseComp 的 830 次 agent runs 同量级；
2. 全量 admission 的 fresh rollout 成本为 `Σ_t |A_z(t)|`，最坏可远大于 corpus 搜索成本，并可能在一个高覆盖 subproblem 上反复评测相同 candidate–instance 组合。

所以不应因“上千条 corpus”预先截断 DCI；应先测 corpus bytes、每次 search latency、每个 DCI 的 turns，以及 `|A_z|` 分布。只有测到 I/O/搜索是实际瓶颈，才引入保持 hit-set 等价的分片执行；admission 成本继续由既有 budget/cache/owner 语义处理，不能拿 search top-k 偷换。

这些工作都发表于 2026，尚缺同行评审后的稳定版本和独立复现。方法质量可概括为：**接口动机与同模型受控证据较强，跨规模普适性与独立复现成熟度中等；作为 COMPASS 的 subproblem/evidence/admission-scope owner 适配度较高，但没有证据支持它替代整个 optimizer、fresh evaluator、acceptance 和 state commit**。

### 15.3 正确的 COMPASS 接口

原始 DCI 论文和官方 runner 只定义搜索轨迹、局部证据和自由文本最终回答；它们没有现成实现 COMPASS 的 canonical subproblem、unique instance 计数和 admission scope。这个事实说明表示边界仍需 grilling，**既不证明 DCI 不能承担这些工作，也不授权向 DCI 模型增加 JSON 或其他固定格式要求**。

所以 DCI 命中的文本 span 只能是 provenance，不能是一票：

```text
seed bottleneck
  -> DCI 在冻结 corpus/frontier snapshot 上交互搜索
  -> 官方自由文本 final answer + 官方 tool trajectory
  -> free-text immutable definition + explicit selection workspace members
  -> official proposer/admission/reference/acceptance/frontier/commit owners
```

这里没有第二个 COMPASS 分类器重新判断 DCI 结果。计数 `|F_z|` 是对 DCI 返回的合法 unique `DataId` 做确定性去重；同一文件的多个 span、同一 instance 的多个 rollout/frontier owners 不能重复加票。

### 15.4 上游优先的最小接入面

对 DCI-Agent-Lite/Pi 的只读兼容性 probe 必须验证：

1. 每个 seed 有隔离的只读 corpus、工作目录和 artifact 目录，不共享可变 session；
2. 输入携带稳定 `DataId`、seed rollout 和已有 subproblem definitions；
3. 保留自由文本 final answer，并验证官方 tool trajectory 是否已足够承载候选 `DataId`、evidence span、执行命令和不确定项；不得预设模型生成 JSON 或固定字段；
4. conversation、tool events、final output 和失败状态可回溯；官方 Lite 已提供 `events.jsonl`、`state.json`、`conversation_full.json`、`final.txt` 等 artifact seam；
5. runtime 与 Pi 依赖固定到 commit，不能依赖浮动分支；
6. raw rollout 文本视为不可信输入，只允许只读搜索/读取，不能访问密钥、项目写路径或 held-out data。

若官方 seam 足够，只写 representation adapter。若不足，先记录可复现缺口，再选最小 owner patch；不在 bridge 重写 DCI agent loop。RARG 是 Pi harness 的 Python 重实现，不能整套引入；GrepSeek executor 也只有在可独立复用且 byte-exact 时才进入。

### 15.5 排序中性的加速合同

这里的“无偏”沿用本项目的目标排序中性，不是统计无偏。对固定 seed、corpus snapshot 和 subproblem catalog：

```text
加速前后的 unique hit set、canonical DataId classification、F_z、S_z
以及后续 RNG 输入必须完全相同。
```

满足合同的 shard parallelism、cache 或 traversal reordering 只改变墙钟时间。hard top-K、局部 workspace、提前停止或只留高相关 match 会改变 hit set、raw failure count、DCI evidence scope 和 skill tree，不能称为中性优化。RARG/RISE/DR-DCI 只有在最终完整性物化或有等价性证明时才可进入 exact 路径。

DCI agent 的成员检索仍可能有方差和系统错误。因此应固定 prompt/model/snapshot，并在离线人工标注子集上测 instance-level precision/recall；不能因执行顺序中性，就把 DCI 成员发现称为统计无偏。

### 15.6 最小实验矩阵

- 相同 bottleneck 的 instance-level precision/recall 与人工审计分歧；
- duplicate subproblem creation rate；
- evidence/component provenance 正确率，以及组件是否真正进入 nominee；
- 每个 subproblem 的 unique `|F_z|`、`|S_z|`，而非 span/tool-call 数；
- DCI tool calls、tokens、wall time、truncation/compaction 与失败类型；
- 改 execution window、shard 数和文件布局后，hit set、`F_z`/`S_z`、nominee 与 commit 是否不变；
- 同模型、同 harness、同语料、同预算下：raw DCI、调优深 BM25（Pi-Serini 风格）、relevance-order-only、无 DCI proposer；
- 最终有效 skill 数量、subproblem 分布、admission outcome 与 held-out 效果。

DCI 论文依赖 gold evidence 的 coverage/localization 只能离线诊断，不能进入在线 evidence sampling 或 admission。

### 15.7 当前明确不做

- 不把 grep 命中数直接当失败数；
- DCI 可以决定 `F_z`/`S_z` 和 admission scope `A_z`；但不把它变成新的 corpus-only acceptance gate，也不替代 scope 内已有 fresh evaluation/reference/acceptance；
- 不用 hard top-K/workspace 替代全语料 membership universe；
- 不复制 Pi、RARG 或 GEPA 已有 agent/evaluator/optimizer loop；
- 第一版不引入 GrepSeek SFT/GRPO、DR-DCI reset 或 Bayesian/importance correction；
- 不把 held-out reporting validation/test 放入 rollout corpus；
- 不依据 gold document/evidence 在线决策。

**D6 已确认的范围仅限**：首版只接 pinned DCI-Agent-Lite/Pi 的官方 runtime、tool 和 artifact seams。严格 JSON final-output 适配不属于已确认范围，已经撤销。其他检索/加速方法不进入首版实现；需要比较时作为独立实验方法处理，不提前增加运行路径或伪需求。

## 17. D6b：重新 grilling——官方自由文本结果的语义边界

**当前有效结论**：D6b-11 已确认每次独立 seed DCI 都创建一个新的不可变 subproblem。因而 D6b-4 至 D6b-9 中关于 existing/new 归属、多 revision 更新和未来归类的讨论均已被取代，只作为被放弃路径的决策记录保留；首版没有这些运行分支。

已确认的否定约束：

- DCI 原方法不要求模型生成 JSON；COMPASS 不得把这一要求施加给 DCI 模型；
- 不用 `submit_subproblem` 自定义工具、Markdown/正则抽取、修复 parser、第二个格式化模型调用或另一种固定文本模板替代 JSON；
- 当前 `bridge/prompts/dci_subproblem_json.txt` 与 `decode_dci_final()` 是建立在无效设计决定上的诊断实现，不能作为正式算法继续运行；
- pinned DCI-Agent-Lite/Pi 的 runtime、read/bash tool trajectory 和 artifact ownership 继续复用。

已确认的表示边界：官方自由文本 final answer 原样成为本次新 subproblem 的不可变 definition 并进入 proposer；selection workspace 的稳定 DataIds 是成员、proposal sampling pool 和 admission scope 的程序性依据；普通 tool hits 仍只是检索轨迹。不得再从“下游需要状态”跳到“模型必须按某种格式生成状态”。

### D6b-1（已确认）：final answer 是 definition，不是状态命令

具体冲突：一次 DCI 可能读取实例 A 来支持“同类失败”，又读取实例 B 作为反例。官方自由文本可以解释这一区别，但官方 artifact 没有定义可由程序直接消费的 membership 字段。把所有 tool hits 都算入 `F_z/S_z` 会混入探索噪声；解析自由文本、要求引用格式或改用自定义提交工具，又会恢复已经否决的模型格式合同。

官方自由文本 final answer 原样成为本次新 subproblem 的不可变 definition，并作为只读自然语言证据交给现有 proposer。它不被解析为 `F_z/S_z` 或 admission scope；这些程序状态只来自 DCI selection workspace 的稳定 DataIds 与既有 reward/frontier metadata。

### D6b-2（已确认）：DCI selection workspace

JSON 不是抽取 subproblem rollout 集合的唯一办法。一个更贴近直接语料交互的候选是给每次 seed DCI 一个隔离、可写的 **DCI selection workspace**：

- 冻结 rollout corpus 保持只读，每个逻辑 rollout 以稳定 `DataId` 身份呈现；
- DCI 可以自由检索和读取成员候选、成功对照与反例；
- 只有 DCI 显式放入 selection workspace 的 rollout 身份才构成该 seed 的候选 subproblem rollout 集合；
- 普通读取、`rg` 返回或反例核验不自动成为成员；
- bridge 只验证 workspace 中的身份属于冻结合法 corpus、按 unique instance 去重；不解析 final answer；
- `F_z/S_z` 可依据被选择 rollout 已有的 reward/frontier 元数据确定，而不是让模型生成这两个 JSON 列表；
- 官方自由文本 final answer 继续作为 subproblem/skill proposal 的自然语言证据。

Pi 的 read/bash 与官方事件轨迹可承载这一操作，但“workspace membership 是 subproblem membership”是 COMPASS 的新集成语义，不是官方 DCI-Agent-Lite 已有实现。它仍要求模型完成一次显式选择动作，只是不要求固定自然语言格式、JSON、引用模板或第二个格式化调用。

已确认：selection workspace 是该次 DCI 明确选择的 rollout 成员集合。普通 `rg` 命中、读取过的候选、反例和探索路径不自动成为成员。程序仅验证 workspace 中的稳定身份、按 unique instance 去重，并依据已有 reward/frontier 元数据形成 proposal 的 success/failure sampling pools；完整 admission scope 由同一成员集合减去既有 prospective frontier-ineligible IDs 得到。官方自由文本 final answer 原样进入 proposer，不承担集合序列化。

### D6b-3（由 D4 推出）：seed 是隐式成员

建议：当前 seed rollout 由系统以概率一视为该次 subproblem expansion 的隐式成员，不要求 DCI 再把它放入 selection workspace。selection workspace 只选择额外的同 subproblem rollout。这样保持已经确认的“每条 seed rollout 必定进入自己的 proposer evidence”，并避免模型遗漏 seed 导致集合语义不稳定；seed 是否进入 raw frontier count 继续由既有 clean-frontier owner 决定，admission 仍通过既有 proposal-lineage exclusion 排除 seed。

这不是新增决定：D4 已确认 seed 固定进入自己的 proposal，且每次 expansion 由该 seed 触发。因此 seed 是系统隐式成员；DCI 只显式选择额外 rollout。

### D6b-4（已被 D6b-11 取代）：existing/new 子问题归属

仅有 rollout 成员集合还不能确定它属于哪个已有 subproblem，或是否需要创建新 subproblem。这里的“归属”只表示本次 DCI 结果记在哪个子问题名下，不是新的软件模块或状态 owner。不能用 membership overlap 阈值自动猜测，否则会引入未经定义的相似度 gate；也不能解析自由文本。

在每次隔离 selection workspace 中预置互斥目标：一个 `new` 目标，以及当前冻结 catalog 的每个 existing canonical subproblem 目标。DCI 只把本次选中的 DataIds 放入其中一个目标：选择 existing 目标表示归入该子问题，并追加 definition revision；选择 `new` 表示创建新 subproblem，其官方自由文本结果原样成为首个 definition revision。程序只读取被选择的目标身份与成员文件，不解析 final answer，也不增加相似度 gate。

用户确认的表示关系：DCI 检索结果的官方自由文本本身就是 subproblem definition；selection workspace 是该 definition 对应的 rollout 成员集合。程序不从 definition 中抽取成员 ID。因而一次 DCI 的语义产物是：

```text
natural-language retrieval result = subproblem definition
explicit selection workspace       = subproblem rollout members
```

结论：selection workspace 可以承载一个已经确定的 existing/new 归属，但它本身不能产生该判断。不能凭文本相似度、membership overlap 或未经验证的 LLM 分类任务自动猜测；在归属判据未解决前，这个目录表示不构成实现授权。

### D6b-5（已被 D6b-11 简化）：已有 definition 不覆盖

当 DCI 判断 seed 属于已有 subproblem 时，本次检索仍会产生新的自由文本结果。建议 canonical definition 不被单次结果覆盖：已有 definition 保持身份稳定，本次自由文本作为带 provenance 的补充 evidence report；只有选择 `new` 时，自由文本才创建新的 canonical definition。否则同一 subproblem 会随随机生成不断改名和概念漂移，并改变后续检索、采样与 admission scope。

已确认：每次自由文本检索结果都追加为该 canonical subproblem 的 definition revision，不覆盖任何旧 revision。每个 revision 同时保留与它配对的 selection workspace 成员、unique instance 数量及其语义所需元数据。概念迁移通过后续 revisions 表达，而不是原地改写历史。

### D6b-6（已被 D6b-11 简化）：revision 元数据在提交时冻结

每个 revision 的成员身份、unique instance count、success/failure counts、seed identity 和 corpus/frontier snapshot identity 都在 canonical commit 时冻结；未来 corpus/frontier 改变不回写旧 revision。新的 DCI expansion 产生新的 revision 和新快照。因此旧 revision 始终能解释当时的 rollout sampling、raw count 与 admission scope。

结论：**旧 revision 不回算；新证据写入新 revision。**

### D6b-7（已被 D6b-11 取代）：没有形成唯一子问题归属时只跳过当前 seed

若一次 DCI 结束或真正超时后，selection workspace 没有选择任何子问题，或同时写入多个子问题，程序没有合法依据从自由文本、命中重叠或相似度中猜测归属。只让该 seed event 跳过 proposal/admission，保留 fresh seed rollout 与官方 DCI artifacts，并累计一次明确的 `dci_attribution_incomplete` skip；它不算任务失败，同 batch 其他 seed events 继续。

### D6b-8（已被 D6b-11 取代）：只读取最新 definition，但保留原始 instance 证据

旧 definition revisions 永久保留，但未来 DCI 做子问题归类时只读取最新一条 definition 文本，不把全部历史 definition 塞入 prompt，也不自动生成覆盖式摘要。同时，DCI 必须能查看该子问题的原始 instance 证据，以避免连续 revisions 把语义逐步带离最初问题。

### D6b-9（保留为创建记录语义）：首个且唯一 definition 的 instances 构成固定原始锚点

“原始子问题 instances”是创建该 subproblem 时首个且唯一 definition 配对的 frozen unique instance 集合。它永久保留，不随后续 corpus/frontier 变化而移动。首版不会向这个 subproblem 追加后续 DCI revisions；新的 DCI discovery 总是创建新的 subproblem。

固定锚点不需要把所有长 rollout 拼进 prompt。它以稳定 DataId、基础元数据和只读原始 rollout 文件存在于 catalog workspace；DCI 初始只得到 seed、最新 definition 与锚点文件入口，再复用官方 `rg`/`find`/`sed`/`read` 交互按需查找和分段读取。只有实际读取的片段进入模型上下文，官方 runtime context management 继续负责长工具轨迹；首版不增加 rollout summarizer、embedding index 或第二个压缩模型。

### D6b-10（已否决）：不得把 existing/new 变成新的模型判断任务

只让模型“选 existing 或 new”并不充分可靠：默认 existing 会造成错误合并，默认 new 会造成重复拆分，固定锚点只能抑制概念漂移，不能把 LLM 判断变成事实。

曾建议使用三路模型判断：

1. `existing(z)`：seed 与 `z` 的固定锚点具有相同的最早未解决转移、失败机制和所需通用修复；题面相似本身不算证据；
2. `new`：在搜索 catalog 后，seed 与所有有实际证据的 existing 候选在上述至少一个事实维度上存在具体不相容，而不是仅仅“相似度不够高”；
3. `abstain`：证据不足以忠实支持前两者时，不默认选任一侧，复用 D6b-7，只跳过当前 seed 的 proposal/admission。

该建议仍然假定模型能够忠实识别“同一中间步骤、失败机制和通用修复”，并把本方法尚未定义的语义判定包装成一个额外任务。用户已明确拒绝这一假设。因此本节只保留为被否决的设计记录，不能进入 prompt、代码、gate 或实验配置。

### D6b-11（已确认）：每次 DCI discovery 都新建 subproblem

本地与 pinned upstream 均没有已验证的 subproblem identity owner。不存在可直接复用的结构化标签、确定性等价关系或官方分类接口。此时最小且事实忠实的首版是：每个独立 seed DCI discovery 都创建一个新的不可变 subproblem identity（首个 revision）；只有同一 logical DCI event 的安全 resume/replay 保持该 identity。不同 DCI discoveries 即使成员重叠或自由文本相近，也不自动合并；重叠只作为事实保留。

代价是会产生重复 subproblems、降低预算效率；优点是不会凭模型猜测或人为阈值伪造“它们相同”。若未来要自动归并，必须另有可验证的身份合同或单独研究出的归并方法，再重新 grilling。

已确认的 prompt 边界：`subproblems.json` 保持模型可见，但只是可选上下文。prompt 不强制模型读取它，不告诉模型“必须新建一个 subproblem”，也不要求任何 existing/new、identity 或 merge 判断。一次调用后的新 identity 与不可变记录继续由编排器自动创建。

### D6b-12（已撤销）：不另定义 DCI timeout/失败语义

timeout、运行中断或 proposal operation 失败完全复用 GEPA 已有失败逻辑。DCI 层不增加 incomplete discovery 状态、专用计数器、恢复分支或 gate；官方 DCI artifacts 的保留仍由 pinned runner owner 负责。

### D6b-13（已确认）：在 prompt 中说明 corpus 与 instance 计数单位

DCI 能通过工作目录和显式文件名定位语料，并不意味着它已经理解 COMPASS 的记录结构或成员计数单位。每次 seed interaction 的动态 prompt 应加入一小段 corpus guide，直接说明：`seed.json` 是当前 fresh seed；`rollouts/` 只包含此前已提交的训练期 proposal-parent/admission-candidate rollouts；held-out reporting data 不在其中；每个 `rollouts/<DataId-token>.json` 是一个 unique task instance 的文档，其中按稳定 rollout 顺序完整保留该实例的全部历史 logical rollouts。

该段同时报告当前冻结快照的真实统计：rollout record 数、unique historical `DataId` 数、proposal-parent/admission-candidate 各自数量，以及合法 clean-frontier instance 数。统计由编排器从已物化的精确记录计算，不交给模型估计。

这里的 subproblem instance 是一个合法 unique `DataId`：其 clean-frontier evidence 面对与 seed 相同的具体要求或可观察堵点，可以仍在该处失败，也可以作为已经跨过该堵点的 success contrast。完整题面与主题不必相同；同一 `DataId` 的多条 rollout、多个 frontier owner、文本 span 或搜索命中仍只计一个 instance。该定义只澄清现有 selection workspace、`F_z/S_z` 和 admission scope 的单位，不增加分类器、审计 gate、相似度阈值或第二套状态。

### D6b-14（已确认）：合法 DataId 只保留一个精确来源

动态 prompt 不再内联展开完整的合法 DataId 列表，只报告合法 marker token 的数量，并明确要求从 `allowed_data_ids.json` 读取精确 token。该文件继续由现有 corpus 物化 owner 生成，现有 `id_by_token` 边界继续拒绝未知 marker；不新增解析器、验证器、fallback 或状态。这样不改变 selection 语义，只移除与同一 owner 文件重复且会干扰弱模型的长串 opaque IDs。

### D6b-15（已确认）：采用官方 document-corpus 等价布局

首版采用 pinned DCI-Agent-Lite 的 BRIGHT/BrowseComp benchmark 所使用的 identity-to-document 物理布局，而不是新增 retriever 或 manifest。对 COMPASS，文档身份是稳定 `DataId`：每个 `rollouts/<DataId-token>.json` 完整保留旧版 corpus 已有的序列化 logical-rollout payload，并保持该实例内的稳定 rollout 顺序；当前 seed 仍单独放在 `seed.json`，不提前混入历史 corpus。DCI 先以官方建议的多组定向 `rg -l` 搜完整 `rollouts/`，再读取候选 instance 文档并补充搜索角度。

这是相对于旧版 DCI corpus snapshot 无新增损失的 representation adapter：它继续使用既有 DSPy JSON 序列化边界，所有已经序列化的历史 rollout、frontier 标记、candidate、example、output、trajectory 与 provenance 字段仍可访问；不额外承诺任意 Python `Any` 对象的字节级往返。合法 marker 集合和下游 `F_z/S_z`、proposal sampling、admission、frontier、selector 均不改变。首版不增加 generated summary、semantic manifest、embedding、ranking、top-k、过滤、提前停止或第二压缩模型，也不在本决定中启用有损 runtime context level。

历史 subproblem catalog 仍只是可选上下文；某个 DataId 已属于历史 subproblem 不会使它失去本次 selection 资格。不同独立 discoveries 可以保留重叠成员，模型不承担 existing/new、归并或排他分配任务。

## 16. D7：已确认——只补 post-proposal admission owner seam

实际调用链已经给出一个可复现缺口：普通 GEPA admission 在 reflective proposal 生成之前由 admission sampler 固定 IDs；而 DCI 的合法 subproblem admission pool 只有 DCI 结束后才存在。现有 `frontier_ineligible_ids` 只能过滤 sampler 已选集合，不能让 sampler 在这个动态 pool 内选择预算受限的 admission IDs。

最小修复位于现有 `ReflectiveMutationProposer` owner：

- admission hook 若没有实现新 seam，原 sampler、`prepare`、Stage 4 和 commit 路径完全不变；
- hook 若实现 `prepare_after_proposals`，proposer 在一个 reflection batch 成功形成 nominees 后，按 canonical task 顺序一次传入这些 nominees；
- hook 只返回与 requests 对齐的既有 `AdmissionPlan`，或用 `None` 跳过对应 nominee；
- GEPA 继续验证 IDs 属于官方 admission loader、batch 与 loader 对象严格对齐，并复用已有 recursive proposal-lineage exclusion；
- 之后仍由已有 Stage 4 evaluator、reference/acceptance/frontier/commit owners 处理，hook 不执行这些状态转换。

这不是新的 scheduler，也不是第二套 admission。它只是让产生动态 scope 的 proposal operation 把 exact IDs 交回 admission owner。batch 并发继续由已有 `IndependentSampling`、adapter batch proposal/evaluation 和 GEPA canonical commit 顺序承担。

rollout corpus 的另一个已复现表示缺口也按同一原则处理：`_observation_facts` 只保留最大值绑定事实，不能代表 append-only corpus；cache 又不保存 trajectory。实现只允许在 GEPA 已经同时持有稳定 `DataId` 与完成的 `EvaluationBatch` 的位置，把对齐结果转发给 adapter 的 append-only sink。它不重新执行 rollout，不从 callback/object identity 猜 ID，不改变 cache、预算、失败、acceptance 或 frontier 语义。

因此首版明确不增加：第二个并发循环、候选池、准入 gate、reference selector、acceptance policy、retry/timeout owner、修复 parser、通用安全/审计框架，或为了未来可能需要而预留的多后端抽象。

### D7-1（已确认）：1:1 是 proposal/admission evidence 实例数之比

补充确认：admission 数量固定为 3 个不同实例。先从当前 subproblem 的 eligible 成员中无放回随机抽取；若不足 3 个，则从全局合法 admission pool 无放回随机补齐到 3。全局补齐仍排除本轮 proposal instance 以及其递归祖先的 proposal 集合；不得重复同一实例，不建立 quota/debt，也不把补齐实例伪装成 subproblem 成员。该规则只决定本次 admission 实例集合，不改变 subproblem membership、frontier、reference、F/E 或后续 mask 语义。

先前“把整个 subproblem 送入 admission”的决定撤销。`proposal_evidence_size=3` 时，一个 proposal 使用 `1` 个 fresh seed instance 加 `2` 个复用的 subproblem-history rollout instances；对应 admission 的目标大小为 `3` 个 distinct eligible subproblem instances。因此这里的 `1:1` 是 `3:3` 的 evidence cardinality ratio，不是 fresh metric-call ratio；history evidence 不重新执行，但仍计入 proposal 侧的 evidence 数量。

DCI 仍先给出完整合法 member pool，并继续复用 GEPA 已有的当前与递归 proposal-lineage 排除。只有从剩余 pool 选出的预算内 IDs 才进入既有 reference、Stage 4 candidate evaluation、StrictImprovement、frontier、F/E 和 commit 路径；未抽中的 subproblem members 不获得伪造 observation 或 F/E。若该 subproblem 的 eligible distinct members 少于 `3`，则从全局合法 admission pool 均匀、无放回补齐到 `3`；补齐实例仍服从同一递归 proposal-lineage 排除，不成为 subproblem member，也不改变其元数据。

### D7-2（已确认）：正式 DCI 进化前先做一个完整冷启动 epoch

初始 skill 先在每个 proposal-eligible 训练实例上各执行一次。冷启动必须完整覆盖这一轮训练实例，整轮完成之前不运行 DCI、不生成 skill candidate，也不执行 admission；整轮结束后才允许正式 DCI-COMPASS 进化。冷启动 rollout 是正常的真实 rollout：进入 append-only rollout corpus，并按官方 owner 计入预算以及既有 clean frontier/F/E。冷启动只增加初始化证据，不建立 quota/debt，不改变后续 batch 的算法语义。
