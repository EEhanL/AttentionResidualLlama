#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import optuna

RESULT_JSON_KEYS = {
    "zeroshot_eng": ("results", "zeroshot_eng", "acc,none"),
    "zeroshot_nld": ("results", "zeroshot_nld", "acc,none"),
    "zeroshot_zho": ("results", "zeroshot_zho", "acc,none"),
    "blimp_eng": ("results", "blimp_babylm_filtered", "acc,none"),
    "blimp_nld": ("results", "blimp_nl", "acc,none"),
    "blimp_zho": ("results", "zhoblimp", "acc,none"),
}


def run_cmd(cmd: list[str], cwd: Path, log_path: Path, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
        p = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        return p.wait()


def parse_best_val_loss(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    m = re.findall(r"best_val_loss=([0-9]+(?:\.[0-9]+)?)", txt)
    if m:
        return float(m[-1])
    m2 = re.findall(r"best_val_loss:([0-9]+(?:\.[0-9]+)?)", txt)
    return float(m2[-1]) if m2 else None


def finalize_metrics(out: dict[str, float]) -> dict[str, float]:
    if all(k in out for k in ("zeroshot_eng", "zeroshot_nld", "zeroshot_zho")):
        out["avg_zeroshot"] = (out["zeroshot_eng"] + out["zeroshot_nld"] + out["zeroshot_zho"]) / 3.0
    bl = [out[k] for k in ("blimp_eng", "blimp_nld", "blimp_zho") if k in out]
    if bl:
        out["avg_blimp"] = sum(bl) / len(bl)
    return out


def parse_json_metrics(path: Path) -> dict[str, float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for name, keys in RESULT_JSON_KEYS.items():
        cur: Any = d
        ok = True
        for k in keys:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and isinstance(cur, (int, float)):
            out[name] = float(cur)
    return finalize_metrics(out)


def read_eval_metrics(eval_root: Path, revision: str, hf_model_dir: Path) -> dict[str, float]:
    slug = str(hf_model_dir).replace("/", "__")
    results_dir = eval_root.parent / "results" / revision / slug
    if not results_dir.exists():
        return {}
    files = sorted(results_dir.glob("results_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    merged: dict[str, float] = {}
    for p in files[:12]:
        one = parse_json_metrics(p)
        for k, v in one.items():
            if k not in ("avg_zeroshot", "avg_blimp"):
                merged[k] = v
        if all(k in merged for k in ("zeroshot_eng", "zeroshot_nld", "zeroshot_zho")):
            break
    return finalize_metrics(merged)


def suggest_params(trial: optuna.Trial, cfg: argparse.Namespace) -> dict[str, Any]:
    optimizer = trial.suggest_categorical("optimizer", ["adamw", "muon"] if cfg.search_optimizer else [cfg.optimizer_default])
    p = {
        "optimizer": optimizer,
        "learning_rate": trial.suggest_float("learning_rate", cfg.lr_min, cfg.lr_max, log=True),
        "weight_decay": trial.suggest_float("weight_decay", cfg.wd_min, cfg.wd_max),
        "dropout": trial.suggest_float("dropout", cfg.dropout_min, cfg.dropout_max),
        "muon_learning_rate": cfg.muon_lr_default,
    }
    if optimizer == "muon":
        p["muon_learning_rate"] = trial.suggest_float("muon_learning_rate", cfg.muon_lr_min, cfg.muon_lr_max, log=True)
    return p


def pretrain(cfg: argparse.Namespace, trial_dir: Path, p: dict[str, Any], max_epoch: int) -> tuple[Path, float | None]:
    out_dir = trial_dir / "train_out"
    env = os.environ.copy()
    if cfg.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    cmd = [
        sys.executable, "pretrain.py", "--data-bin", cfg.data_bin, "--vocab-size", str(cfg.vocab_size),
        "--out-dir", str(out_dir), "--max-epoch", str(max_epoch), "--learning-rate", str(p["learning_rate"]),
        "--weight-decay", str(p["weight_decay"]), "--dropout", str(p["dropout"]), "--optimizer", p["optimizer"],
        "--muon-learning-rate", str(p["muon_learning_rate"]),
    ]
    if cfg.cuda_visible_devices:
        cmd += ["--cuda-visible-devices", cfg.cuda_visible_devices]
    rc = run_cmd(cmd, Path(cfg.project_root), trial_dir / "logs" / "pretrain.log", env=env)
    if rc != 0:
        raise RuntimeError(f"pretrain failed rc={rc}")
    best = out_dir / "pretrain" / "best.pth"
    if not best.exists():
        raise RuntimeError("best.pth missing")
    return best, parse_best_val_loss(out_dir / "pretrain" / "log.log")


def convert(cfg: argparse.Namespace, trial_dir: Path, best_pth: Path) -> Path:
    hf_dir = trial_dir / "hf_model"
    cmd = [
        sys.executable, "hf_remote_code/convert_pth_to_hf.py", "--pth", str(best_pth), "--out_dir", str(hf_dir),
        "--tokenizer_type", cfg.tokenizer_type, "--dim", str(cfg.dim), "--n_layers", str(cfg.n_layers), "--n_heads", str(cfg.n_heads),
        "--n_kv_heads", str(cfg.n_kv_heads), "--vocab_size", str(cfg.vocab_size), "--multiple_of", str(cfg.multiple_of),
        "--max_seq_len", str(cfg.max_seq_len), "--dropout", str(cfg.dropout_default),
    ]
    rc = run_cmd(cmd, Path(cfg.project_root), trial_dir / "logs" / "convert.log")
    if rc != 0:
        raise RuntimeError(f"convert failed rc={rc}")
    return hf_dir


def evaluate(cfg: argparse.Namespace, trial_dir: Path, hf_dir: Path) -> dict[str, float]:
    cmd = ["bash", "scripts/zeroshot_model.sh", "--model_name", str(hf_dir), "--langs", cfg.langs, "--revision", cfg.revision]
    rc = run_cmd(cmd, Path(cfg.eval_root), trial_dir / "logs" / "eval.log")
    if rc != 0:
        raise RuntimeError(f"eval failed rc={rc}")
    return read_eval_metrics(Path(cfg.eval_root), cfg.revision, hf_dir)


def save_meta(trial_dir: Path, obj: dict[str, Any]) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "meta.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def top_params(study: optuna.Study, k: int) -> list[dict[str, Any]]:
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    rev = study.direction.name == "MAXIMIZE"
    done.sort(key=lambda t: t.value, reverse=rev)
    return [t.params for t in done[:k]]


def run_phase_a(cfg: argparse.Namespace) -> None:
    run_root = Path(cfg.run_root) / "phase_a"
    study = optuna.create_study(study_name=cfg.study_a, storage=cfg.storage, load_if_exists=True, direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.seed))

    def objective(trial: optuna.Trial) -> float:
        p = suggest_params(trial, cfg)
        td = run_root / f"trial_{trial.number:04d}"
        try:
            best, val = pretrain(cfg, td, p, cfg.phase_a_epochs)
            save_meta(td, {"phase": "A", "trial": trial.number, "params": p, "best_pth": str(best), "best_val_loss": val, "status": "ok"})
            return float(val) if val is not None else 1e9
        except Exception as e:
            save_meta(td, {"phase": "A", "trial": trial.number, "params": p, "status": "failed", "error": str(e)})
            return 1e9

    study.optimize(objective, n_trials=cfg.phase_a_trials)


def run_phase_b_or_c(cfg: argparse.Namespace, phase: str) -> None:
    prev = cfg.study_a if phase == "B" else cfg.study_b
    curr = cfg.study_b if phase == "B" else cfg.study_c
    topn = cfg.phase_b_topk if phase == "B" else cfg.phase_c_topn
    epochs = cfg.phase_b_epochs if phase == "B" else cfg.phase_c_epochs
    n_trials = cfg.phase_b_trials if phase == "B" else cfg.phase_c_trials

    prev_study = optuna.load_study(study_name=prev, storage=cfg.storage)
    seeds = top_params(prev_study, topn)
    if not seeds:
        raise RuntimeError(f"Phase {phase} requires completed previous phase")

    run_root = Path(cfg.run_root) / f"phase_{phase.lower()}"
    study = optuna.create_study(study_name=curr, storage=cfg.storage, load_if_exists=True, direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.seed + (1 if phase == 'B' else 2)))
    if len(study.trials) == 0:
        for p in seeds:
            study.enqueue_trial(p)

    def objective(trial: optuna.Trial) -> float:
        p = suggest_params(trial, cfg)
        td = run_root / f"trial_{trial.number:04d}"
        try:
            best, val = pretrain(cfg, td, p, epochs)
            hf = convert(cfg, td, best)
            m = evaluate(cfg, td, hf)
            score = float(m.get("avg_zeroshot", -1.0))
            save_meta(td, {"phase": phase, "trial": trial.number, "params": p, "best_pth": str(best), "best_val_loss": val, "metrics": m, "objective": score, "status": "ok"})
            return score
        except Exception as e:
            save_meta(td, {"phase": phase, "trial": trial.number, "params": p, "status": "failed", "error": str(e)})
            return -1.0

    study.optimize(objective, n_trials=max(n_trials, len(seeds)))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="3-phase Optuna pipeline")
    ap.add_argument("--phase", choices=["A", "B", "C", "all"], default="A")
    ap.add_argument("--project-root", default="/root/autodl-tmp/AttentionResidualLlama")
    ap.add_argument("--eval-root", default="/root/autodl-tmp/babylm-eval/multilingual")
    ap.add_argument("--run-root", default="/root/autodl-tmp/AttentionResidualLlama/runs/optuna")
    ap.add_argument("--storage", default="sqlite:////root/autodl-tmp/AttentionResidualLlama/runs/optuna/optuna_multiphase.db")
    ap.add_argument("--study-a", default="multilingual_phase_a")
    ap.add_argument("--study-b", default="multilingual_phase_b")
    ap.add_argument("--study-c", default="multilingual_phase_c")

    ap.add_argument("--data-bin", default="./data/merged_multilingual_regexbbpe_zh1_en1_nl1_100m.bin")
    ap.add_argument("--tokenizer-type", choices=["chatglm", "regex_bbpe"], default="regex_bbpe")
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--langs", default="eng nld zho")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--cuda-visible-devices", default="0")

    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-kv-heads", type=int, default=8)
    ap.add_argument("--multiple-of", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--dropout-default", type=float, default=0.05)

    ap.add_argument("--phase-a-trials", type=int, default=24)
    ap.add_argument("--phase-b-trials", type=int, default=8)
    ap.add_argument("--phase-c-trials", type=int, default=3)
    ap.add_argument("--phase-a-epochs", type=int, default=2)
    ap.add_argument("--phase-b-epochs", type=int, default=4)
    ap.add_argument("--phase-c-epochs", type=int, default=8)
    ap.add_argument("--phase-b-topk", type=int, default=8)
    ap.add_argument("--phase-c-topn", type=int, default=3)

    ap.add_argument("--optimizer-default", choices=["adamw", "muon"], default="muon")
    ap.add_argument("--search-optimizer", action="store_true")
    ap.add_argument("--lr-min", type=float, default=1e-4)
    ap.add_argument("--lr-max", type=float, default=6e-4)
    ap.add_argument("--wd-min", type=float, default=0.01)
    ap.add_argument("--wd-max", type=float, default=0.2)
    ap.add_argument("--dropout-min", type=float, default=0.0)
    ap.add_argument("--dropout-max", type=float, default=0.15)
    ap.add_argument("--muon-lr-min", type=float, default=5e-5)
    ap.add_argument("--muon-lr-max", type=float, default=3e-4)
    ap.add_argument("--muon-lr-default", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    cfg = parse_args()
    Path(cfg.run_root).mkdir(parents=True, exist_ok=True)
    if cfg.phase in ("A", "all"):
        run_phase_a(cfg)
    if cfg.phase in ("B", "all"):
        run_phase_b_or_c(cfg, "B")
    if cfg.phase in ("C", "all"):
        run_phase_b_or_c(cfg, "C")
    print("\n[OPTUNA PIPELINE DONE]")
    print(f"run_root={cfg.run_root}")
    print(f"storage={cfg.storage}")


if __name__ == "__main__":
    main()
