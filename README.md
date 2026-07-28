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

## v16 关键配置

- `n_candidates = 3`
- `parent_selection.top_n = 5`
- `epsilon_dep = 0.8`
- `credit_hops = dependency_hops = 1`
- 先做精确 `q=F/E` 的可逆传递后代遮蔽，再做全局 tie-inclusive top-5，
  最后只在 selection-active 集合中按 `p ∝ q` 随机采样。
- quota/debt 和立即强制探索均未实现，也不应恢复。

本地运行前需要把 v16 JSON 中的绝对 checkpoint/run 路径改为当前机器路径，
设置 `SILICONFLOW_API_KEY` 与 `DSPY_CACHEDIR`，并提供与锁定源码兼容的 Python/
CUDA 依赖。入口会要求 run 目录此前不存在，因此不要把已有停止点当成新 run
目录直接覆盖。

快照状态和代码/产物边界见 `docs/snapshot-20260728.md`。
