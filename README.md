# COMPASS / IFBench skill evolution

这是从共享服务器
`/mnt/geogpt-doc-new/deepresearch/gepa-multi-skill/reflection-bridge`
保存的代码快照。快照时间为 2026-07-28（Asia/Shanghai）。

仓库直接保存全部自定义源码、测试和实验配置：

- `bridge/`：FlashTrace、terminal likelihood、dependency gate、实例级 frontier、
  admission，以及可逆祖先遮蔽和 tie-inclusive top-n proposer 选择。
- `experiments/`：AIME/IFBench 入口和全部现存 JSON 配置，包括生产配置
  `11_ifbench_siliconflow_sparse_single_gpu_v16.json`。
- `upstreams/`：语义 owner 的官方源码，以 submodule 固定到实际运行所用提交。
- `patches/`：远端官方工作树相对固定提交的精确补丁。补丁保留 owner-layer
  修复和 GEPA admission/state 扩展，没有在 bridge 中复制官方实现。
- `upstreams.lock.json`：来源、提交和补丁的机器可读清单。

缓存、模型权重、API 密钥和生成型 run 产物不提交到 Git。v16 的完整停止点
（包括约 18 MB 的 `gepa_state.bin`）已另存于本机 `.local-runs/`，并由
`docs/v16-local-run-manifest.json` 记录校验和。

## 取得完整工作树

Windows：

```powershell
git clone git@github.com:LceOmlet/compass.git
cd compass
powershell -ExecutionPolicy Bypass -File scripts/prepare-upstreams.ps1
```

Linux：

```bash
git clone git@github.com:LceOmlet/compass.git
cd compass
bash scripts/prepare-upstreams.sh
```

脚本会初始化精确提交的 submodule，并幂等地应用 `patches/`。GEPA artifact
仓库中的大型 LFS 实验压缩包不属于运行代码，初始化时会跳过其下载。

## v16 基线语义

- `n_candidates = 3`
- `parent_selection.top_n = 5`
- `epsilon_dep = 0.8`
- `credit_hops = dependency_hops = 1`
- 先做精确 `q=F/E` 的可逆传递后代遮蔽，再做全局 tie-inclusive top-5，
  最后只在 selection-active 集合中按 `p ∝ q` 随机采样。
- quota/debt 和立即强制探索均未实现，也不应恢复。

本地运行前需要把 v16 JSON 中的绝对 checkpoint/run 路径改为当前机器路径，
设置 `SILICONFLOW_API_KEY` 与 `DSPY_CACHEDIR`，并提供与锁定源码兼容的 Python/
CUDA 依赖。入口默认要求 run 目录此前不存在；只有显式传入
`--resume-existing` 时才会接受已存在且含官方 `gepa_state.bin` 的目录。

## 批内并发与 teacher forcing

配置 `11_ifbench_siliconflow_sparse_single_gpu_v17_batched_tf.json` 保持
`B_propose/B_admit` 各 3 个实例、`n_candidates=3`，并显式设置：

- `official_gepa.num_threads=32`：DSPy 只在当前批的实例内并发；每批实际最多
  3 个任务，不会并发 optimizer iteration 或跨批执行。
- `teacher_forcing_enabled=true` 与 `teacher_forcing_batch_size=3`：同一旧
  rollout 的 3 个候选通过左填充、显式 `attention_mask/position_ids` 在一次
  Qwen3 forward 中评分；batch 行与 candidate index 保持一一对应。
- 真实 BF16 kernel 可因 batch 形状改变 likelihood 数值和候选排名；终态选择
  使用该次 batch=3 forward 的实际排名，不要求复现 batch=1 排名。
- batch=3 若 OOM 会直接失败，不会降为 batch=2、batch=1 或顺序执行。
- 只有 TF 排序和 dependency gate 后的最终候选会进入 `B_admit`。

正式续跑配置
`11_ifbench_siliconflow_sparse_single_gpu_v20_resume_v16_batched_tf.json`
从另存的 v16 官方 checkpoint 继续，沿用上述 batch=3 语义，并要求入口显式
传入 `--resume-existing`。原始 v16 停止点不被覆盖。

配置 `11_ifbench_siliconflow_sparse_single_gpu_v18_no_tf.json` 是关闭 TF 的
对应入口：官方 proposer 固定只生成 1 个候选，跳过 TF forward，但仍执行
同一个 old-rollout dependency gate；`B_propose/B_admit` 的三实例 DSPy
并发保持不变。历史配置若只有 `n_candidates` 字段，会继续按原有的顺序 TF
（batch size 1）解释。

生产启动前应在实际任务模型/设备上执行：

```bash
python scripts/validate-token-replay-batch.py \
  --checkpoint /path/to/Qwen3-8B
```

该检查同时报告 batch=1 与 batch=3 的 token likelihood、最大数值差和候选
排名，拒绝非有限值，但不把两种执行形状的数值或排名差异判为错位。候选行、
mask、position 和 credited-token 坐标由接口测试严格验证。v17/v18 目前仅
保存为配置，仓库操作不会自动启动实验。

快照状态和代码/产物边界见 `docs/snapshot-20260728.md`。
