#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict


TARGET_KEYS = {
    "zeroshot_eng": "zeroshot_eng",
    "zeroshot_nld": "zeroshot_nld",
    "zeroshot_zho": "zeroshot_zho",
    "blimp_babylm_filtered": "blimp_eng",
    "blimp_nl": "blimp_nld",
    "zhoblimp": "blimp_zho",
}

RESULT_JSON_KEYS = {
    "zeroshot_eng": ("results", "zeroshot_eng", "acc,none"),
    "zeroshot_nld": ("results", "zeroshot_nld", "acc,none"),
    "zeroshot_zho": ("results", "zeroshot_zho", "acc,none"),
    "blimp_eng": ("results", "blimp_babylm_filtered", "acc,none"),
    "blimp_nld": ("results", "blimp_nl", "acc,none"),
    "blimp_zho": ("results", "zhoblimp", "acc,none"),
}


def run_cmd(cmd: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        return proc.wait()


def parse_eval_table(eval_text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for line in eval_text.splitlines():
        if "|" not in line:
            continue
        if set(line.strip()) <= {"|", "-", " ", "+"}:
            continue

        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 6:
            continue

        task = cols[1] if cols[0] == "" else cols[0]
        value_col = cols[5] if cols[0] == "" and len(cols) > 5 else cols[4] if len(cols) > 4 else ""
        if task not in TARGET_KEYS:
            continue

        m = re.search(r"-?\d+(?:\.\d+)?", value_col)
        if m:
            out[TARGET_KEYS[task]] = float(m.group(0))

    return finalize_metrics(out)


def finalize_metrics(out: Dict[str, float]) -> Dict[str, float]:
    if all(k in out for k in ("zeroshot_eng", "zeroshot_nld", "zeroshot_zho")):
        out["avg_zeroshot"] = (out["zeroshot_eng"] + out["zeroshot_nld"] + out["zeroshot_zho"]) / 3.0

    blimps = [out[k] for k in ("blimp_eng", "blimp_nld", "blimp_zho") if k in out]
    if blimps:
        out["avg_blimp"] = sum(blimps) / len(blimps)
    return out


def parse_metrics_from_result_json(result_json_path: Path) -> Dict[str, float]:
    data = json.loads(result_json_path.read_text(encoding="utf-8"))
    out: Dict[str, float] = {}
    for metric_name, key_path in RESULT_JSON_KEYS.items():
        cur = data
        ok = True
        for k in key_path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and isinstance(cur, (int, float)):
            out[metric_name] = float(cur)
    return finalize_metrics(out)


def locate_eval_result_files(eval_root: Path, hf_model_dir: Path, revision: str) -> list[Path]:
    model_slug = str(hf_model_dir).replace("/", "__")
    results_root = eval_root.parent / "results" / revision / model_slug
    if not results_root.exists():
        return []
    files = sorted(results_root.glob("results_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def parse_eval_results(eval_root: Path, hf_model_dir: Path, revision: str, eval_log_path: Path) -> Dict[str, float]:
    # 1) Preferred: parse newest json result files produced by lm-eval
    result_files = locate_eval_result_files(eval_root, hf_model_dir, revision)
    merged: Dict[str, float] = {}
    for p in result_files[:10]:
        part = parse_metrics_from_result_json(p)
        merged.update({k: v for k, v in part.items() if k not in ("avg_zeroshot", "avg_blimp")})
        if all(k in merged for k in ("zeroshot_eng", "zeroshot_nld", "zeroshot_zho")):
            break
    merged = finalize_metrics(merged)
    if any(k.startswith("zeroshot_") for k in merged):
        return merged

    # 2) Fallback: parse terminal table
    eval_text = read_text_or_empty(eval_log_path)
    return parse_eval_table(eval_text)


def read_text_or_empty(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def fail_and_exit(run_root: Path, status: dict, status_name: str, msg: str) -> None:
    status["status"] = status_name
    (run_root / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(msg)
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke run: pretrain -> HF convert -> official multilingual eval")
    ap.add_argument("--project-root", default="/root/autodl-tmp/AttentionResidualLlama")
    ap.add_argument("--eval-root", default="/root/autodl-tmp/babylm-eval/multilingual")

    ap.add_argument("--run-name", default=None, help="Optional run name. Default: smoke_YYYYmmdd_HHMMSS")
    ap.add_argument("--resume-run-dir", default=None, help="Existing run dir under runs/smoke to reuse artifacts.")
    ap.add_argument("--skip-pretrain", action="store_true", help="Skip pretrain and reuse existing checkpoint/HF dir.")
    ap.add_argument("--skip-convert", action="store_true", help="Skip HF conversion and evaluate an existing HF dir.")
    ap.add_argument("--hf-model-dir", default=None, help="Existing HF model dir to evaluate directly.")

    ap.add_argument("--data-bin", default="./data/merged_multilingual_regexbbpe_zh1_en1_nl1_100m.bin")
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--tokenizer-type", choices=["chatglm", "regex_bbpe"], default="regex_bbpe")

    ap.add_argument("--max-epoch", type=int, default=1)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--muon-learning-rate", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--optimizer", choices=["adamw", "muon"], default="muon")
    ap.add_argument("--cuda-visible-devices", default="0")

    ap.add_argument("--langs", default="eng nld zho")
    ap.add_argument("--revision", default="main")

    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-kv-heads", type=int, default=8)
    ap.add_argument("--multiple-of", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=512)

    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    eval_root = Path(args.eval_root).resolve()

    if args.resume_run_dir:
        run_root = Path(args.resume_run_dir).resolve()
        run_name = run_root.name
    else:
        run_name = args.run_name or f"smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_root = project_root / "runs" / "smoke" / run_name

    train_out = run_root / "train_out"
    default_hf_out = run_root / "hf_model"
    hf_out = Path(args.hf_model_dir).resolve() if args.hf_model_dir else default_hf_out
    logs_dir = run_root / "logs"
    run_root.mkdir(parents=True, exist_ok=True)

    status = {
        "run_name": run_name,
        "project_root": str(project_root),
        "eval_root": str(eval_root),
        "train_out": str(train_out),
        "hf_out": str(hf_out),
        "started_at": datetime.now().isoformat(),
        "skip_pretrain": args.skip_pretrain,
        "skip_convert": args.skip_convert,
    }

    best_pth = train_out / "pretrain" / "best.pth"

    if not args.skip_pretrain:
        train_cmd = [
            sys.executable,
            "pretrain.py",
            "--data-bin",
            args.data_bin,
            "--vocab-size",
            str(args.vocab_size),
            "--out-dir",
            str(train_out),
            "--max-epoch",
            str(args.max_epoch),
            "--learning-rate",
            str(args.learning_rate),
            "--muon-learning-rate",
            str(args.muon_learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--dropout",
            str(args.dropout),
            "--optimizer",
            args.optimizer,
            "--cuda-visible-devices",
            args.cuda_visible_devices,
        ]
        rc = run_cmd(train_cmd, project_root, logs_dir / "pretrain.log")
        status["pretrain_rc"] = rc
        if rc != 0:
            fail_and_exit(run_root, status, "failed_pretrain", f"[FAIL] pretrain failed with rc={rc}")

    if not best_pth.exists() and not args.skip_convert:
        fail_and_exit(run_root, status, "failed_no_best_ckpt", "[FAIL] best.pth not found for conversion")

    if not args.skip_convert:
        convert_cmd = [
            sys.executable,
            "hf_remote_code/convert_pth_to_hf.py",
            "--pth",
            str(best_pth),
            "--out_dir",
            str(hf_out),
            "--tokenizer_type",
            args.tokenizer_type,
            "--dim",
            str(args.dim),
            "--n_layers",
            str(args.n_layers),
            "--n_heads",
            str(args.n_heads),
            "--n_kv_heads",
            str(args.n_kv_heads),
            "--vocab_size",
            str(args.vocab_size),
            "--multiple_of",
            str(args.multiple_of),
            "--max_seq_len",
            str(args.max_seq_len),
            "--dropout",
            str(args.dropout),
        ]
        rc = run_cmd(convert_cmd, project_root, logs_dir / "convert.log")
        status["convert_rc"] = rc
        if rc != 0:
            fail_and_exit(run_root, status, "failed_convert", f"[FAIL] convert failed with rc={rc}")

    if not hf_out.exists():
        fail_and_exit(run_root, status, "failed_no_hf_dir", f"[FAIL] HF model dir not found: {hf_out}")

    eval_cmd = [
        "bash",
        "scripts/zeroshot_model.sh",
        "--model_name",
        str(hf_out),
        "--langs",
        args.langs,
        "--revision",
        args.revision,
    ]
    eval_log_path = logs_dir / "eval.log"
    rc = run_cmd(eval_cmd, eval_root, eval_log_path)
    status["eval_rc"] = rc
    if rc != 0:
        fail_and_exit(run_root, status, "failed_eval", f"[FAIL] eval failed with rc={rc}")

    metrics = parse_eval_results(eval_root, hf_out, args.revision, eval_log_path)

    required = ["zeroshot_eng", "zeroshot_nld", "zeroshot_zho"]
    wanted_langs = args.langs.split()
    wanted_keys = [f"zeroshot_{x}" for x in wanted_langs if f"zeroshot_{x}" in required]
    if not wanted_keys:
        wanted_keys = required
    missing = [k for k in wanted_keys if k not in metrics]

    result = {
        "run_name": run_name,
        "best_ckpt": str(best_pth) if best_pth.exists() else None,
        "hf_model_dir": str(hf_out),
        "metrics": metrics,
        "missing_required_metrics": missing,
        "status": "ok" if not missing else "warning_missing_metrics",
        "finished_at": datetime.now().isoformat(),
    }

    (run_root / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    status["status"] = result["status"]
    status["finished_at"] = result["finished_at"]
    (run_root / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    print("\n[SMOKE RUN DONE]")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
