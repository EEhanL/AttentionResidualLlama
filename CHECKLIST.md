# 性能差异排查清单

## ✅ 已验证正确的部分

1. **权重绑定** - 训练时正确，转换后也正确
2. **模型架构** - depth_query 参数都存在（8层）
3. **配置参数** - dropout=0.05, vocab_size=16000, 等都正确

## 🔍 需要进一步检查的方面

### 1. Tokenizer 差异（最可能的原因！）

**你使用的**：regex_bbpe (vocab_size=16000)
**同学可能使用的**：chatglm (vocab_size=64793)

**影响**：
- 词表大小差异巨大（16k vs 64k）
- 不同的 tokenizer 会导致完全不同的 token 序列
- 更大的词表通常在语言任务上表现更好
- **这很可能是性能差异的主要原因！**

### 2. 数据预处理差异

- 数据混合比例可能不同
- 是否在每个样本末尾加 <eos>
- 序列截断方式

### 3. 训练超参数差异

- learning rate
- weight decay
- 训练 epoch 数
- batch size
- warmup steps

### 4. 评估方式差异

- 官方 eval 脚本的版本
- 评估时的设置

## 🎯 建议的下一步

### 第一步：确认 tokenizer

**最重要！** 先问同学：
- 使用的是 chatglm tokenizer 还是 regex_bbpe？
- vocab_size 是多少？

### 第二步：对比配置

创建配置对比表：

| 配置项 | 你的设置 | 同学的设置 |
|--------|----------|------------|
| tokenizer | regex_bbpe | ? |
| vocab_size | 16000 | ? |
| dropout | 0.05 | ? |
| learning_rate | 2e-4 | ? |
| weight_decay | 0.1 | ? |
| max_epoch | 5 | ? |
| batch_size | 32 | ? |

### 第三步：如果需要，使用相同配置重新训练

如果确认同学用的是 chatglm tokenizer，建议重新训练。

## 💡 快速诊断命令

```bash
echo "=== 模型配置 ==="
cat out/hf_babylm_ckpt_regexbbpe_fixed/config.json | grep -E "vocab_size|dropout"

echo -e "\n=== Tokenizer 信息 ==="
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('out/hf_babylm_ckpt_regexbbpe_fixed', trust_remote_code=True)
print(f'Vocab size: {len(tok)}')
print(f'EOS token id: {tok.eos_token_id}')
"
```
