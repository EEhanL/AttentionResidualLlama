# BabyLlama with Attention Residual

A lightweight LLaMA-style Transformer enhanced with **Attention Residual** for the **BabyLM Challenge** multilingual track (Chinese/English/Dutch).

## Attention Residual

Instead of the standard residual connection (`x + Attention(x)`), we implement **cross-layer attention aggregation** inspired by [Kimi-K2](https://arxiv.org/abs/2506.09857):

- Each layer maintains a learnable **depth query** vector
- The input to each layer is computed by attending over all previous layer outputs
- This allows information to flow directly from any earlier layer to deeper layers

```
Standard Residual:     h_l = h_{l-1} + Attention(h_{l-1})

Attention Residual:    h_l = Aggregate(h_0, h_1, ..., h_{l-1}) + Attention(...)
                       where Aggregate uses softmax attention with depth_query
```

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Hidden Size | 512 |
| Layers | 8 |
| Attention Heads | 8 |
| KV Heads | 8 (MHA) |
| FFN Hidden | 2048 (SwiGLU) |
| Max Sequence Length | 512 |
| Vocab Size | 16,000 |
| Total Parameters | ~35M |

**Key Features:**
- **Attention Residual** (cross-layer aggregation)
- RMSNorm (Pre-LN)
- Rotary Position Embedding (RoPE)
- SwiGLU FFN
- Weight Tying (input/output embeddings)

## Tokenizer

**RegexBBPE** (Byte-Level BPE with regex pre-tokenization):
- Vocab size: 16,000
- Regex pattern handles CJK characters, Latin words, numbers, and punctuation separately

## Usage

### 1. Build Training Data

```bash
python build_multilingual_pretrain_bin.py \
  --zh-path ./data/babylm/zho \
  --en-path ./data/babylm/eng_strict \
  --nl-path ./data/babylm/nld \
  --tokenizer-type regex_bbpe \
  --tokenizer-path ./tokenizer_regex_bbpe \
  --budget-words 100000000 \
  --output ./data/train.bin
```

### 2. Pretrain

```bash
python pretrain.py \
  --data-bin ./data/train.bin \
  --vocab-size 16000 \
  --learning-rate 2e-4 \
  --dropout 0.05 \
  --max-epoch 5
```

### 3. Export to HuggingFace Format

```bash
python hf_remote_code/convert_pth_to_hf.py \
  --pth out/pretrain/best.pth \
  --out_dir out/hf_model \
  --tokenizer_type regex_bbpe \
  --dim 512 --n_layers 8 --n_heads 8 --n_kv_heads 8 \
  --vocab_size 16000 --max_seq_len 512 --dropout 0.05
```

## References

- [Kimi-K2 Technical Report](https://arxiv.org/abs/2506.09857) - Attention Residual mechanism

## License

MIT
