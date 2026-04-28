#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm


DEFAULT_SOURCES = {
    "zh": "./data/babylm/zho",
    "en": "./data/babylm/eng_strict",
    "nl": "./data/babylm/nld",
}

DEFAULT_RATIOS = {
    "zh": 1,
    "en": 1,
    "nl": 1,
}


class DatasetCursor:
    def __init__(self, dataset, seed: int):
        self.dataset = dataset
        self.size = len(dataset)
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(self.size)
        self.ptr = 0
        self.epochs = 0

    def next_text(self) -> str:
        if self.ptr >= self.size:
            self.order = self.rng.permutation(self.size)
            self.ptr = 0
            self.epochs += 1
        text = self.dataset[int(self.order[self.ptr])]["text"]
        self.ptr += 1
        return text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a multilingual BabyLM pretraining bin with fixed token ratios."
    )
    parser.add_argument(
        "--output",
        default="./data/merged_multilingual_zh1_en1_nl1_100m.bin",
        help="Output .bin path.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=100_000_000,
        help="Total token budget after tokenization, including <eos>.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for document order.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="./chatglm_tokenizer/tokenizer.model",
        help="SentencePiece tokenizer model path.",
    )
    parser.add_argument("--zh-path", default=DEFAULT_SOURCES["zh"], help="Saved Chinese dataset path.")
    parser.add_argument("--en-path", default=DEFAULT_SOURCES["en"], help="Saved English dataset path.")
    parser.add_argument("--nl-path", default=DEFAULT_SOURCES["nl"], help="Saved Dutch dataset path.")
    parser.add_argument(
        "--buffer-tokens",
        type=int,
        default=1_000_000,
        help="Flush token buffer to disk after this many tokens.",
    )
    return parser.parse_args()


def choose_language(token_counts, target_counts):
    remaining = {lang: target_counts[lang] - token_counts[lang] for lang in target_counts}
    candidates = [lang for lang, left in remaining.items() if left > 0]
    if not candidates:
        return max(target_counts, key=lambda lang: target_counts[lang] - token_counts[lang])
    return max(candidates, key=lambda lang: remaining[lang])


def flush_tokens(output_path: Path, token_buffer):
    if not token_buffer:
        return
    arr = np.asarray(token_buffer, dtype=np.uint16)
    with open(output_path, "ab") as f:
        f.write(arr.tobytes())
    token_buffer.clear()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    sp = SentencePieceProcessor(model_file=args.tokenizer_path)
    eos_id = sp.eos_id()

    datasets_map = {
        "zh": load_from_disk(args.zh_path)["train"],
        "en": load_from_disk(args.en_path)["train"],
        "nl": load_from_disk(args.nl_path)["train"],
    }
    cursors = {
        lang: DatasetCursor(dataset, seed=args.seed + idx)
        for idx, (lang, dataset) in enumerate(datasets_map.items())
    }

    ratio_sum = sum(DEFAULT_RATIOS.values())
    target_counts = {
        lang: int(args.budget * ratio / ratio_sum)
        for lang, ratio in DEFAULT_RATIOS.items()
    }
    target_counts["nl"] += args.budget - sum(target_counts.values())

    token_counts = {lang: 0 for lang in DEFAULT_RATIOS}
    doc_counts = {lang: 0 for lang in DEFAULT_RATIOS}
    token_buffer = []
    total_tokens = 0

    pbar = tqdm(total=args.budget, desc="building multilingual bin", unit="tok")
    while total_tokens < args.budget:
        lang = choose_language(token_counts, target_counts)
        text = cursors[lang].next_text()
        text_ids = sp.encode(text)
        if not text_ids:
            continue
        text_ids.append(eos_id)

        remaining_total = args.budget - total_tokens
        remaining_lang = target_counts[lang] - token_counts[lang]
        allowed = min(len(text_ids), remaining_total, max(remaining_lang, 0))
        if allowed <= 0:
            continue

        token_buffer.extend(text_ids[:allowed])
        token_counts[lang] += allowed
        doc_counts[lang] += 1
        total_tokens += allowed
        pbar.update(allowed)

        if len(token_buffer) >= args.buffer_tokens:
            flush_tokens(output_path, token_buffer)

    flush_tokens(output_path, token_buffer)
    pbar.close()

    print(f"saved to: {output_path}")
    print(f"total tokens: {total_tokens}")
    for lang in ["zh", "en", "nl"]:
        ratio = token_counts[lang] / max(total_tokens, 1)
        print(
            f"{lang}: tokens={token_counts[lang]} docs={doc_counts[lang]} "
            f"share={ratio:.4f} dataset_passes={cursors[lang].epochs}"
        )


if __name__ == "__main__":
    main()
