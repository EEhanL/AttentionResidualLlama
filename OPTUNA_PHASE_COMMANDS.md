# Optuna 三阶段运行指令

## 1) Phase A（粗筛）

```bash
python optuna_multiphase_pipeline.py \
  --phase A \
  --phase-a-trials 24 \
  --phase-a-epochs 2
```

## 2) Phase B（复筛，自动读取 A 的 top-k）

```bash
python optuna_multiphase_pipeline.py \
  --phase B \
  --phase-b-topk 8 \
  --phase-b-trials 8 \
  --phase-b-epochs 4
```

## 3) Phase C（决赛，自动读取 B 的 top-n）

```bash
python optuna_multiphase_pipeline.py \
  --phase C \
  --phase-c-topn 3 \
  --phase-c-trials 3 \
  --phase-c-epochs 8
```

## 可选：一键顺序跑完

> 第一次不建议直接用，建议先单独跑 A，确认稳定后再跑 B/C。

```bash
python optuna_multiphase_pipeline.py --phase all
```

## 常用可选参数

```bash
# 指定 GPU
--cuda-visible-devices 0

# 指定 sqlite 存储位置
--storage sqlite:////root/autodl-tmp/AttentionResidualLlama/runs/optuna/optuna_multiphase.db

# 如果想同时搜索 adamw / muon
--search-optimizer
```

---

## 结果汇总脚本（A/B/C 当前最佳）

脚本文件：`optuna_report.py`

作用：
- 从 Optuna sqlite 中读取 A/B/C 三个 study 的 best trial
- 打印一页表（phase、best score、trial id、trial 目录）
- 同时打印每个 phase 的 best params，便于你直接复现实验

### 使用方法

```bash
python optuna_report.py
```

如果你改过 study 名字或 sqlite 路径，可显式指定：

```bash
python optuna_report.py \
  --storage sqlite:////root/autodl-tmp/AttentionResidualLlama/runs/optuna/optuna_multiphase.db \
  --study-a multilingual_phase_a \
  --study-b multilingual_phase_b \
  --study-c multilingual_phase_c \
  --run-root /root/autodl-tmp/AttentionResidualLlama/runs/optuna
```

### 什么时候运行

- 跑完 Phase A 后：看粗筛当前最优参数与分数
- 跑完 Phase B 后：看复筛最优参数与 `avg_zeroshot`
- 跑完 Phase C 后：查看最终最优候选



这版 optuna_multiphase_pipeline.py 默认会调这些参数：
learning_rate
范围：1e-4 ~ 6e-4（log 采样）
weight_decay
范围：0.01 ~ 0.2
dropout
范围：0.0 ~ 0.15
optimizer（可选）
默认不搜索，固定为 muon（--optimizer-default muon）
只有你加了 --search-optimizer 才会在 adamw / muon 间搜索
muon_learning_rate
仅当 optimizer=muon 时调
范围：5e-5 ~ 3e-4（log 采样）
若 optimizer 不是 muon，则用默认 1e-4