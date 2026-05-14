# 发现的问题及修复方案

## 🔴 问题 1：权重绑定（Weight Tying）在转换时丢失 [已修复]

### 问题描述
训练代码 `model.py` 中使用了权重绑定：
```python
self.tok_embeddings.weight = self.output.weight  # 共享权重
```

但在转换到 HuggingFace 格式时，`convert_pth_to_hf.py` 的 `_ensure_tied_embeddings()` 函数只是**复制**了权重，而不是真正的共享。更严重的是，HF 的 `modeling_babyllama_kimi.py` 中虽然定义了 `tie_weights()` 方法，但在 `Transformer.__init__` 中没有显式绑定权重。

### 影响
- **训练时**：embedding 和 output 共享权重，参数量更少，正则化效果更好
- **评估时**：如果权重没有正确绑定，相当于模型突然多了一倍的输出层参数，但这些参数是随机初始化的或者是复制的旧值，导致输出质量严重下降
- **BLiMP 等语法任务**：对输出分布的准确性要求极高，权重不一致会导致得分大幅下降

### 修复内容

#### 1. 修复 `hf_remote_code/modeling_babyllama_kimi.py`
在 `Transformer.__init__` 中添加权重绑定：
```python
self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

# Weight tying: share embeddings with output projection
self.tok_embeddings.weight = self.output.weight
```

在 `BabyLlamaKimiForCausalLM.__init__` 中确保调用 `tie_weights()`：
```python
self.model = Transformer(args)

# Ensure weight tying is applied
self.tie_weights()
self.post_init()
```

#### 2. 修复 `hf_remote_code/convert_pth_to_hf.py`
修改 `_ensure_tied_embeddings()` 函数，只保存一份权重：
```python
def _ensure_tied_embeddings(state: dict) -> dict:
    tok_k = "model.tok_embeddings.weight"
    out_k = "model.output.weight"
    
    # 如果两个都存在，删除 output.weight（HF 会自动绑定）
    if tok_k in state and out_k in state:
        if not torch.equal(state[tok_k], state[out_k]):
            print("WARNING: tok_embeddings.weight and output.weight are different!")
        del state[out_k]
    elif out_k in state and tok_k not in state:
        state[tok_k] = state[out_k]
        del state[out_k]
    
    return state
```

### 验证方法
运行验证脚本：
```bash
# 验证训练好的 .pth 文件
python verify_weight_tying.py out/pretrain/best.pth

# 验证转换后的 HF 模型
python verify_weight_tying.py out/hf_babylm_ckpt
```

---

## ⚠️ 问题 2：`build_multilingual_pretrain_bin.py` 中的 DatasetDict 访问错误 [已修复]

### 问题描述
`load_from_disk()` 返回的是 `DatasetDict`（包含 'train' 等 split），但代码直接用索引访问。

### 修复内容
在加载数据集后选择 'train' split：
```python
ds = load_from_disk(path)
if hasattr(ds, 'keys'):  # It's a DatasetDict
    datasets[lang] = ds['train']
else:
    datasets[lang] = ds
```

---

## 🔍 其他需要检查的潜在问题

### 1. Dropout 一致性
**检查点**：确保训练、转换、评估时的 dropout 参数一致

- 训练时：`--dropout 0.05`（默认）
- 转换时：`--dropout 0.05`（必须匹配）
- 评估时：模型会自动进入 `eval()` 模式，dropout 会被禁用

**验证**：
```bash
# 检查训练日志中的 dropout 值
grep "dropout" out/pretrain/log.log

# 检查转换后的 config.json
cat out/hf_babylm_ckpt/config.json | grep dropout
```

### 2. 词表大小一致性
**检查点**：确保 tokenizer、训练、转换时的 vocab_size 一致

- ChatGLM tokenizer: `vocab_size=64793`
- Regex BBPE tokenizer: `vocab_size=16000`

**验证**：
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("./tokenizer_regex_bbpe", trust_remote_code=True)
print(f"Tokenizer vocab size: {len(tok)}")
```

### 3. 特殊 token ID 一致性
**检查点**：确保 EOS/BOS/PAD token ID 在训练和评估时一致

默认值：
- `bos_token_id=1`
- `eos_token_id=2`
- `pad_token_id=None` 或 `0`

**验证**：
```python
from transformers import AutoTokenizer, AutoConfig
tok = AutoTokenizer.from_pretrained("out/hf_babylm_ckpt", trust_remote_code=True)
config = AutoConfig.from_pretrained("out/hf_babylm_ckpt", trust_remote_code=True)

print(f"Tokenizer EOS: {tok.eos_token_id}")
print(f"Config EOS: {config.eos_token_id}")
```

### 4. 数据预处理一致性
**检查点**：确保训练数据和评估数据的预处理方式一致

- 是否在每个样本末尾添加 `<eos>`？
- 序列长度截断方式是否一致？
- 是否使用了相同的 tokenizer？

### 5. 模型架构参数
**检查点**：确保所有架构参数在训练和转换时完全一致

```bash
# 训练时的默认参数（pretrain.py）
dim=512
n_layers=8
n_heads=8
n_kv_heads=8
vocab_size=64793  # 或 16000
multiple_of=32
max_seq_len=512
dropout=0.05

# 转换时必须完全匹配
python hf_remote_code/convert_pth_to_hf.py \
  --pth out/pretrain/best.pth \
  --out_dir out/hf_babylm_ckpt \
  --dim 512 --n_layers 8 --n_heads 8 --n_kv_heads 8 \
  --vocab_size 64793 --multiple_of 32 --max_seq_len 512 --dropout 0.05
```

---

## 📋 完整的检查清单

在重新训练和评估之前，请确认：

- [ ] 已应用所有代码修复
- [ ] 训练时的 dropout 参数已记录
- [ ] 转换时的所有参数与训练时完全一致
- [ ] 运行 `verify_weight_tying.py` 验证权重绑定
- [ ] 检查 tokenizer vocab_size 与模型配置一致
- [ ] 检查特殊 token ID 一致性
- [ ] 对比同学的配置，确认没有其他差异

---

## 🚀 建议的重新训练流程

### 1. 清理旧模型
```bash
rm -rf out/pretrain/*
rm -rf out/hf_babylm_ckpt*
```

### 2. 重新训练
```bash
python pretrain.py \
  --data-bin ./data/merged_multilingual_zh1_en1_nl1_100m.bin \
  --vocab-size 64793 \
  --dropout 0.05 \
  --max-epoch 5 \
  --learning-rate 2e-4 \
  --weight-decay 0.1
```

### 3. 转换为 HF 格式
```bash
python hf_remote_code/convert_pth_to_hf.py \
  --pth out/pretrain/best.pth \
  --out_dir out/hf_babylm_ckpt \
  --tokenizer_type chatglm \
  --dim 512 --n_layers 8 --n_heads 8 --n_kv_heads 8 \
  --vocab_size 64793 --multiple_of 32 --max_seq_len 512 --dropout 0.05
```

### 4. 验证权重绑定
```bash
python verify_weight_tying.py out/hf_babylm_ckpt
```

### 5. 运行评估
在官方 eval 项目中运行评估脚本。

---

## 💡 为什么权重绑定问题会导致 BLiMP 得分特别差？

BLiMP（Benchmark of Linguistic Minimal Pairs）测试的是模型对语法细节的理解，通过比较两个句子的困惑度来判断哪个更符合语法。

如果 `output.weight` 没有正确绑定：
1. 输出层使用了未经训练的权重（或复制的权重但没有共享梯度）
2. 模型的输出分布与训练时完全不同
3. 困惑度计算不准确
4. 语法判断失败

这就像训练了一个模型，但在评估时换了一个随机的输出层，结果自然会很差。
