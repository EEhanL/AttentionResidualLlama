#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna


def fmt_params(params: dict) -> str:
    if not params:
        return "-"
    ordered = sorted(params.items(), key=lambda x: x[0])
    return ", ".join(f"{k}={v}" for k, v in ordered)


def read_meta_score(run_root: Path, phase: str, trial_number: int) -> str:
    meta_path = run_root / f"phase_{phase}" / f"trial_{trial_number:04d}" / "meta.json"
    if not meta_path.exists():
        return "-"
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return "-"

    if phase == "a":
        v = data.get("best_val_loss")
        return f"{v:.6f}" if isinstance(v, (int, float)) else "-"

    metrics = data.get("metrics", {})
    v = metrics.get("avg_zeroshot")
    return f"{v:.6f}" if isinstance(v, (int, float)) else "-"


def summarize_one(storage: str, study_name: str, phase: str, run_root: Path) -> dict:
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception as e:
        return {
            "phase": phase.upper(),
            "study": study_name,
            "status": f"not_found ({e})",
            "direction": "-",
            "best_value": "-",
            "best_from_meta": "-",
            "trial": "-",
            "params": "-",
            "trial_dir": "-",
        }

    if study.best_trial is None:
        return {
            "phase": phase.upper(),
            "study": study_name,
            "status": "empty",
            "direction": study.direction.name,
            "best_value": "-",
            "best_from_meta": "-",
            "trial": "-",
            "params": "-",
            "trial_dir": "-",
        }

    t = study.best_trial
    trial_dir = run_root / f"phase_{phase}" / f"trial_{t.number:04d}"
    return {
        "phase": phase.upper(),
        "study": study_name,
        "status": "ok",
        "direction": study.direction.name,
        "best_value": f"{t.value:.6f}" if isinstance(t.value, (int, float)) else str(t.value),
        "best_from_meta": read_meta_score(run_root, phase, t.number),
        "trial": str(t.number),
        "params": fmt_params(t.params),
        "trial_dir": str(trial_dir),
    }


def print_table(rows: list[dict]) -> None:
    headers = ["Phase", "Study", "Direction", "Best(Optuna)", "Best(meta)", "Trial", "Status", "TrialDir"]
    data_rows = []
    for r in rows:
        data_rows.append([
            r["phase"],
            r["study"],
            r["direction"],
            r["best_value"],
            r["best_from_meta"],
            r["trial"],
            r["status"],
            r["trial_dir"],
        ])

    widths = [len(h) for h in headers]
    for row in data_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells: list[str]) -> str:
        return " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells)))

    sep = "-+-".join("-" * w for w in widths)
    print(line(headers))
    print(sep)
    for row in data_rows:
        print(line(row))

    print("\nBest params per phase:")
    for r in rows:
        print(f"[{r['phase']}] {r['params']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Report best trials for Optuna phases A/B/C")
    ap.add_argument("--storage", default="sqlite:////root/autodl-tmp/AttentionResidualLlama/runs/optuna/optuna_multiphase.db")
    ap.add_argument("--study-a", default="multilingual_phase_a")
    ap.add_argument("--study-b", default="multilingual_phase_b")
    ap.add_argument("--study-c", default="multilingual_phase_c")
    ap.add_argument("--run-root", default="/root/autodl-tmp/AttentionResidualLlama/runs/optuna")
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    rows = [
        summarize_one(args.storage, args.study_a, "a", run_root),
        summarize_one(args.storage, args.study_b, "b", run_root),
        summarize_one(args.storage, args.study_c, "c", run_root),
    ]
    print_table(rows)


if __name__ == "__main__":
    main()
