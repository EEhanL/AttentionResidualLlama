#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
import re
import math

try:
    import jieba  # optional, used for better Chinese word counting/truncation
except Exception:  # pragma: no cover
    jieba = None


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

BYTE_PREMIUM = {
    # BabyLM multilingual track byte premium (English is baseline)
    "en": 1.0,
    "nl": 1.0516,
    "zh": 0.9894,
}

_WORD_RE_LATIN = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def count_words(text: str, lang: str) -> int:
    text = (text or "").strip()
    if not text:
        return 0

    if lang in ("en", "nl"):
        return len(_WORD_RE_LATIN.findall(text))

    if lang == "zh":
        if jieba is not None:
            return sum(1 for t in jieba.cut(text, cut_all=False) if t.strip())
        return len(_CJK_RE.findall(text))

    return len([t for t in re.split(r"\s+", text) if t])


def truncate_to_words(text: str, lang: str, max_words: int) -> str:
    text = (text or "").strip()
    if max_words <= 0 or not text:
        return ""

    if lang in ("en", "nl"):
        parts = [t for t in re.split(r"\s+", text) if t]
        if len(parts) <= max_words:
            return text
        return " ".join(parts[:max_words])

    if lang == "zh":
        if jieba is not None:
            toks = [t for t in jieba.cut(text, cut_all=False) if t.strip()]
            if len(toks) <= max_words:
                return text
            return "".join(toks[:max_words])
        cjk = _CJK_RE.findall(text)
        return "".join(cjk[:max_words])

    parts = [t for t in re.split(r"\s+", text) if t]
    return " ".join(parts[:max_words])


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
        description=(
            "Build a multilingual BabyLM pretraining bin under an adjusted-words budget "
            "(100M words with byte premium)."
        )
    )
    parser.add_argument(
        "--output",
        default="./data/merged_multilingual_zh1_en1_nl1_100m.bin",
        help="Output .bin path.",
    )
    parser.add_argument(
        "--budget-words",
        type=int,
        default=100_000_000,
        help="Total adjusted word budget (words * byte_premium).",
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


def choose_language(adj_counts, target_adj):
    remaining = {lang: target_adj[lang] - adj_counts[lang] for lang in target_adj}
    candidates = [lang for lang, left in remaining.items() if left > 0]
    if not candidates:
        return max(target_adj, key=lambda lang: target_adj[lang] - adj_counts[lang])
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
    target_adj_words = {
        lang: int(args.budget_words * ratio / ratio_sum)
        for lang, ratio in DEFAULT_RATIOS.items()
    }
    target_adj_words["nl"] += args.budget_words - sum(target_adj_words.values())

    adj_word_counts = {lang: 0.0 for lang in DEFAULT_RATIOS}
    raw_word_counts = {lang: 0 for lang in DEFAULT_RATIOS}
    token_counts = {lang: 0 for lang in DEFAULT_RATIOS}
    doc_counts = {lang: 0 for lang in DEFAULT_RATIOS}
    token_buffer = []
    total_tokens = 0
    total_adj_words = 0.0

    pbar = tqdm(total=args.budget_words, desc="building multilingual bin", unit="adj_words")
    while total_adj_words < args.budget_words:
        lang = choose_language(adj_word_counts, target_adj_words)
        text = cursors[lang].next_text()
        raw_words = count_words(text, lang)
        if raw_words <= 0:
            continue

        premium = BYTE_PREMIUM.get(lang, 1.0)
        remaining_total_adj = args.budget_words - total_adj_words
        remaining_lang_adj = target_adj_words[lang] - adj_word_counts[lang]
        remaining_adj = min(remaining_total_adj, max(remaining_lang_adj, 0.0))
        max_raw_words = int(math.floor(remaining_adj / premium)) if premium > 0 else 0
        if max_raw_words <= 0:
            continue

        if raw_words > max_raw_words:
            text = truncate_to_words(text, lang, max_raw_words)
            raw_words = count_words(text, lang)
            if raw_words <= 0:
                continue

        text_ids = sp.encode(text)
        if not text_ids:
            continue
        text_ids.append(eos_id)

        token_buffer.extend(text_ids)
        token_counts[lang] += len(text_ids)
        doc_counts[lang] += 1
        raw_word_counts[lang] += raw_words
        adj_added = raw_words * premium
        adj_word_counts[lang] += adj_added
        total_adj_words += adj_added
        total_tokens += len(text_ids)
        pbar.update(adj_added)

        if len(token_buffer) >= args.buffer_tokens:
            flush_tokens(output_path, token_buffer)

    flush_tokens(output_path, token_buffer)
    pbar.close()

    print(f"saved to: {output_path}")
    print(f"total tokens: {total_tokens}")
    print(f"total adjusted words: {total_adj_words:.2f} (budget={args.budget_words})")
    for lang in ["zh", "en", "nl"]:
        token_share = token_counts[lang] / max(total_tokens, 1)
        adj_share = adj_word_counts[lang] / max(total_adj_words, 1e-9)
        print(
            f"{lang}: tokens={token_counts[lang]} docs={doc_counts[lang]} "
            f"token_share={token_share:.4f} raw_words={raw_word_counts[lang]} "
            f"adj_words={adj_word_counts[lang]:.2f} adj_share={adj_share:.4f} "
            f"dataset_passes={cursors[lang].epochs}"
        )


if __name__ == "__main__":
    main()
