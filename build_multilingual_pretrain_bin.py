# #!/usr/bin/env python3
# import argparse
# from pathlib import Path

# import numpy as np
# from datasets import load_from_disk
# from sentencepiece import SentencePieceProcessor
# from tqdm import tqdm
# import re
# import math
# from transformers import AutoTokenizer

# try:
#     import jieba  # optional, used for better Chinese word counting/truncation
# except Exception:  # pragma: no cover
#     jieba = None


# DEFAULT_SOURCES = {
#     "zh": "./data/babylm/zho",
#     "en": "./data/babylm/eng_strict",
#     "nl": "./data/babylm/nld",
# }

# DEFAULT_RATIOS = {
#     "zh": 1,
#     "en": 1,
#     "nl": 1,
# }

# BYTE_PREMIUM = {
#     # BabyLM multilingual track byte premium (English is baseline)
#     "en": 1.0,
#     "nl": 1.0516,
#     "zh": 0.9894,
# }

# _WORD_RE_LATIN = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
# _CJK_RE = re.compile(r"[\u4e00-\u9fff]")


# def count_words(text: str, lang: str) -> int:
#     text = (text or "").strip()
#     if not text:
#         return 0

#     if lang in ("en", "nl"):
#         return len(_WORD_RE_LATIN.findall(text))

#     if lang == "zh":
#         if jieba is not None:
#             return sum(1 for t in jieba.cut(text, cut_all=False) if t.strip())
#         return len(_CJK_RE.findall(text))

#     return len([t for t in re.split(r"\s+", text) if t])


# def truncate_to_words(text: str, lang: str, max_words: int) -> str:
#     text = (text or "").strip()
#     if max_words <= 0 or not text:
#         return ""

#     if lang in ("en", "nl"):
#         parts = [t for t in re.split(r"\s+", text) if t]
#         if len(parts) <= max_words:
#             return text
#         return " ".join(parts[:max_words])

#     if lang == "zh":
#         if jieba is not None:
#             toks = [t for t in jieba.cut(text, cut_all=False) if t.strip()]
#             if len(toks) <= max_words:
#                 return text
#             return "".join(toks[:max_words])
#         cjk = _CJK_RE.findall(text)
#         return "".join(cjk[:max_words])

#     parts = [t for t in re.split(r"\s+", text) if t]
#     return " ".join(parts[:max_words])


# class DatasetCursor:
#     def __init__(self, dataset, seed: int):
#         self.dataset = dataset
#         self.size = len(dataset)
#         self.rng = np.random.default_rng(seed)
#         self.order = self.rng.permutation(self.size)
#         self.ptr = 0
#         self.epochs = 0

#     def next_text(self) -> str:
#         if self.ptr >= self.size:
#             self.order = self.rng.permutation(self.size)
#             self.ptr = 0
#             self.epochs += 1
#         text = self.dataset[int(self.order[self.ptr])]["text"]
#         self.ptr += 1
#         return text


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description=(
#             "Build a multilingual BabyLM pretraining bin under an adjusted-words budget "
#             "(100M words with byte premium)."
#         )
#     )
#     parser.add_argument(
#         "--output",
#         default="./data/merged_multilingual_zh1_en1_nl1_100m.bin",
#         help="Output .bin path.",
#     )
#     parser.add_argument(
#         "--budget-words",
#         type=int,
#         default=100_000_000,
#         help="Total adjusted word budget (words * byte_premium).",
#     )
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=42,
#         help="Random seed for document order.",
#     )
#     parser.add_argument(
#         "--tokenizer-type",
#         choices=["chatglm", "regex_bbpe"],
#         default="chatglm",
#         help="Tokenizer backend to use for bin building.",
#     )
#     parser.add_argument(
#         "--tokenizer-path",
#         default="./chatglm_tokenizer/tokenizer.model",
#         help=(
#             "Tokenizer path. For chatglm: tokenizer.model path. "
#             "For regex_bbpe: tokenizer folder path (or HF-loadable tokenizer path)."
#         ),
#     )
#     parser.add_argument("--zh-path", default=DEFAULT_SOURCES["zh"], help="Saved Chinese dataset path.")
#     parser.add_argument("--en-path", default=DEFAULT_SOURCES["en"], help="Saved English dataset path.")
#     parser.add_argument("--nl-path", default=DEFAULT_SOURCES["nl"], help="Saved Dutch dataset path.")
#     parser.add_argument(
#         "--buffer-tokens",
#         type=int,
#         default=1_000_000,
#         help="Flush token buffer to disk after this many tokens.",
#     )
#     return parser.parse_args()


# def choose_language(adj_counts, target_adj):
#     remaining = {lang: target_adj[lang] - adj_counts[lang] for lang in target_adj}
#     candidates = [lang for lang, left in remaining.items() if left > 0]
#     if not candidates:
#         return max(target_adj, key=lambda lang: target_adj[lang] - adj_counts[lang])
#     return max(candidates, key=lambda lang: remaining[lang])


# def flush_tokens(output_path: Path, token_buffer):
#     if not token_buffer:
#         return
#     arr = np.asarray(token_buffer, dtype=np.uint16)
#     with open(output_path, "ab") as f:
#         f.write(arr.tobytes())
#     token_buffer.clear()


# class TokenizerAdapter:
#     def __init__(self, tokenizer_type: str, tokenizer_path: str):
#         self.tokenizer_type = tokenizer_type
#         self.tokenizer_path = tokenizer_path
#         self._sp = None
#         self._hf = None

#         if tokenizer_type == "chatglm":
#             self._sp = SentencePieceProcessor(model_file=tokenizer_path)
#             self._eos_id = int(self._sp.eos_id())
#         elif tokenizer_type == "regex_bbpe":
#             self._hf = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=True)
#             if self._hf.eos_token_id is None:
#                 raise ValueError("regex_bbpe tokenizer has no eos_token_id; please set eos token in tokenizer config")
#             self._eos_id = int(self._hf.eos_token_id)
#         else:
#             raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type}")

#     @property
#     def eos_id(self) -> int:
#         return self._eos_id

#     def encode(self, text: str):
#         if self.tokenizer_type == "chatglm":
#             return self._sp.encode(text)
#         return self._hf.encode(text, add_special_tokens=False)


# def main():
#     args = parse_args()
#     output_path = Path(args.output)
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     if output_path.exists():
#         output_path.unlink()

#     tokenizer = TokenizerAdapter(args.tokenizer_type, args.tokenizer_path)
#     eos_id = tokenizer.eos_id

#     datasets_map = {
#         "zh": load_from_disk(args.zh_path)["train"],
#         "en": load_from_disk(args.en_path)["train"],
#         "nl": load_from_disk(args.nl_path)["train"],
#     }
#     cursors = {
#         lang: DatasetCursor(dataset, seed=args.seed + idx)
#         for idx, (lang, dataset) in enumerate(datasets_map.items())
#     }

#     ratio_sum = sum(DEFAULT_RATIOS.values())
#     target_adj_words = {
#         lang: int(args.budget_words * ratio / ratio_sum)
#         for lang, ratio in DEFAULT_RATIOS.items()
#     }
#     target_adj_words["nl"] += args.budget_words - sum(target_adj_words.values())

#     adj_word_counts = {lang: 0.0 for lang in DEFAULT_RATIOS}
#     raw_word_counts = {lang: 0 for lang in DEFAULT_RATIOS}
#     token_counts = {lang: 0 for lang in DEFAULT_RATIOS}
#     doc_counts = {lang: 0 for lang in DEFAULT_RATIOS}
#     token_buffer = []
#     total_tokens = 0
#     total_adj_words = 0.0

#     pbar = tqdm(total=args.budget_words, desc="building multilingual bin", unit="adj_words")
#     langs = list(DEFAULT_RATIOS.keys())
#     eps = 1e-6
#     min_premium = min(BYTE_PREMIUM.get(l, 1.0) for l in langs)

#     while total_adj_words < args.budget_words - eps:
#         remaining_total_adj = args.budget_words - total_adj_words
#         # If we can't add even 1 raw word in any language, stop to avoid infinite loop.
#         if remaining_total_adj < (min_premium - eps):
#             break

#         feasible_langs = []
#         for l in langs:
#             prem = BYTE_PREMIUM.get(l, 1.0)
#             if prem <= 0:
#                 continue
#             remaining_lang_adj = target_adj_words[l] - adj_word_counts[l]
#             remaining_adj = min(remaining_total_adj, max(remaining_lang_adj, 0.0))
#             if math.floor(remaining_adj / prem) >= 1:
#                 feasible_langs.append(l)

#         if not feasible_langs:
#             break

#         lang = choose_language(
#             {l: adj_word_counts[l] for l in feasible_langs},
#             {l: target_adj_words[l] for l in feasible_langs},
#         )
#         text = cursors[lang].next_text()
#         raw_words = count_words(text, lang)
#         if raw_words <= 0:
#             continue

#         premium = BYTE_PREMIUM.get(lang, 1.0)
#         remaining_total_adj = args.budget_words - total_adj_words
#         remaining_lang_adj = target_adj_words[lang] - adj_word_counts[lang]
#         remaining_adj = min(remaining_total_adj, max(remaining_lang_adj, 0.0))
#         max_raw_words = int(math.floor(remaining_adj / premium)) if premium > 0 else 0
#         if max_raw_words <= 0:
#             continue

#         if raw_words > max_raw_words:
#             text = truncate_to_words(text, lang, max_raw_words)
#             raw_words = count_words(text, lang)
#             if raw_words <= 0:
#                 continue

#         text_ids = tokenizer.encode(text)
#         if not text_ids:
#             continue
#         text_ids.append(eos_id)

#         token_buffer.extend(text_ids)
#         token_counts[lang] += len(text_ids)
#         doc_counts[lang] += 1
#         raw_word_counts[lang] += raw_words
#         adj_added = raw_words * premium
#         adj_word_counts[lang] += adj_added
#         total_adj_words += adj_added
#         total_tokens += len(text_ids)
#         pbar.update(adj_added)

#         if len(token_buffer) >= args.buffer_tokens:
#             flush_tokens(output_path, token_buffer)

#     flush_tokens(output_path, token_buffer)
#     pbar.close()

#     print(f"saved to: {output_path}")
#     print(f"total tokens: {total_tokens}")
#     print(f"total adjusted words: {total_adj_words:.2f} (budget={args.budget_words})")
#     for lang in ["zh", "en", "nl"]:
#         token_share = token_counts[lang] / max(total_tokens, 1)
#         adj_share = adj_word_counts[lang] / max(total_adj_words, 1e-9)
#         print(
#             f"{lang}: tokens={token_counts[lang]} docs={doc_counts[lang]} "
#             f"token_share={token_share:.4f} raw_words={raw_word_counts[lang]} "
#             f"adj_words={adj_word_counts[lang]:.2f} adj_share={adj_share:.4f} "
#             f"dataset_passes={cursors[lang].epochs}"
#         )


# if __name__ == "__main__":
#     main()








"""
Build a multilingual BabyLM pretraining bin file using a HuggingFace tokenizer.
Modified to support HF tokenizers (e.g. Regex-Guided BBPE) instead of SentencePiece.
"""
import argparse
from pathlib import Path
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm

DEFAULT_SOURCES = {
    "zh": "./data/babylm/zho",
    "en": "./data/babylm/eng_strict",
    "nl": "./data/babylm/nld",
}

# Equal split for fairness (BabyLM-aligned). Chinese byte premium ≈ 0.99, almost 1.
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
        default="./data/merged_multilingual_100m.bin",
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
        default="./tokenizer_regex_bbpe",
        help="HuggingFace tokenizer directory path (containing tokenizer.json).",
    )
    parser.add_argument("--zh-path", default=DEFAULT_SOURCES["zh"])
    parser.add_argument("--en-path", default=DEFAULT_SOURCES["en"])
    parser.add_argument("--nl-path", default=DEFAULT_SOURCES["nl"])
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
        output_path.unlink()  # Reset the bin file

    # ============ Load HuggingFace tokenizer (key change!) ============
    print(f"Loading HuggingFace tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print(f"  Vocab size: {len(tokenizer)}")

    # Get EOS token id (handle if not defined)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        # Fallback: use the </s> token id explicitly
        eos_id = tokenizer.convert_tokens_to_ids("</s>")
        print(f"  No eos_token_id, using </s> id: {eos_id}")
    print(f"  EOS token id: {eos_id}")

    # ============ Load datasets ============
    paths = {"zh": args.zh_path, "en": args.en_path, "nl": args.nl_path}
    datasets = {}
    for lang, path in paths.items():
        print(f"Loading {lang} dataset from {path}")
        ds = load_from_disk(path)
        # Handle DatasetDict by selecting 'train' split
        if hasattr(ds, 'keys'):  # It's a DatasetDict
            datasets[lang] = ds['train']
        else:
            datasets[lang] = ds
        print(f"  Size: {len(datasets[lang]):,} documents")

    cursors = {lang: DatasetCursor(datasets[lang], args.seed + i) for i, lang in enumerate(paths)}

    # ============ Compute target token counts per language ============
    total_ratio = sum(DEFAULT_RATIOS.values())
    target_counts = {
        lang: int(args.budget * DEFAULT_RATIOS[lang] / total_ratio) for lang in DEFAULT_RATIOS
    }
    print(f"Target token counts per language: {target_counts}")

    # ============ Tokenize and write ============
    token_counts = {lang: 0 for lang in DEFAULT_RATIOS}
    token_buffer = []
    total_tokens = 0
    pbar = tqdm(total=args.budget, desc="Tokenizing", unit="tok")

    while total_tokens < args.budget:
        lang = choose_language(token_counts, target_counts)
        text = cursors[lang].next_text()
        if not text:
            continue

        # ============ HF tokenizer encode (key change!) ============
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        text_ids.append(eos_id)

        # Sanity check: ensure token IDs fit in uint16
        if any(tid > 65535 or tid < 0 for tid in text_ids):
            raise ValueError(f"Token id out of uint16 range encountered: max={max(text_ids)}")

        # Don't exceed budget
        remaining = args.budget - total_tokens
        if len(text_ids) > remaining:
            text_ids = text_ids[:remaining]

        token_buffer.extend(text_ids)
        token_counts[lang] += len(text_ids)
        total_tokens += len(text_ids)
        pbar.update(len(text_ids))

        if len(token_buffer) >= args.buffer_tokens:
            flush_tokens(output_path, token_buffer)

    flush_tokens(output_path, token_buffer)
    pbar.close()

    print(f"\n=== Tokenization complete ===")
    print(f"Final token counts: {token_counts}")
    print(f"Total tokens written: {total_tokens:,}")
    print(f"Output: {output_path}")
    print(f"Output size: {output_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()