# BabyLM 三语预训练（zh/en/nl）- BabyLlama (Llama2 改造版)

本项目基于一个 Llama 风格的 Transformer 训练代码改造而来，面向 **BabyLM Challenge** 的约束进行适配。

- **三语数据构建**：中文/英文/荷兰语，按 token 比例混合生成 `.bin`（见 `build_multilingual_pretrain_bin.py`）
- **模型改造**：参考 kimi 团队的 attention residual，将模块间残差连接改为“跨层 attention 聚合”（见 `model.py`）
- **仅预训练**：不使用仓库中的微调（SFT）部分
- **导出 HF 格式**：将预训练产物 `.pth` 转为 Hugging Face 标准目录，便于 BabyLM 官方 eval 侧加载（见 `hf_remote_code/convert_pth_to_hf.py`）

---

## 环境依赖

建议使用 Python 3.10+。

```bash
pip install -U torch numpy tqdm datasets sentencepiece transformers
```

---

## 目录约定（默认）

### BabyLM 三语数据（HuggingFace datasets 的 `save_to_disk` 目录）

`build_multilingual_pretrain_bin.py` 默认读取：

- `./data/babylm/zho`（中文）
- `./data/babylm/eng_strict`（英文）
- `./data/babylm/nld`（荷兰语）

每个目录应能被 `datasets.load_from_disk(PATH)["train"]` 正常加载，并且样本字段包含 `text`。

### 分词器

- `./chatglm_tokenizer/tokenizer.model`

### 三语预训练 bin 输出（默认）

- `./data/merged_multilingual_zh4_en3_nl3_100m.bin`

---

## 1) 构建三语 `.bin`（zh/en/nl）

该脚本会对三种语言分别循环取样、按需截断、分词，并在每个样本末尾追加 `<eos>`，最终以 `uint16` token 序列写入 `.bin`。

### 计量规则（对齐 BabyLM multilingual track）

- **总上限**：100M **words**（注意不是 100M tokens）
- **byte premium（用于等效计量）**：
  - English：`1.0`（基准）
  - Dutch：`1.0516`
  - Chinese：`0.9894`
- **adjusted words 定义**：`adjusted_words = raw_words * byte_premium`
- **三语等比例**：这里的 `zh:en:nl = 1:1:1` 指 **adjusted words** 占比相等（等效语义信息量一致），而不是 raw words 或 tokens 相等。

### Word 计数与截断策略（保证不超过预算）

- **English / Dutch**：用正则抽取类 word 片段计数；需要截断时按空白切分并保留前 N 个词。
- **Chinese**：优先使用 `jieba` 分词计数与截断；若环境未安装 `jieba`，退化为按 CJK 汉字数近似计数/截断。

脚本在接近预算上限时，会先按 word 边界截断文本，再做 SentencePiece 分词写入 `.bin`，从而 **严格保证**总 adjusted words 不超过 `--budget-words`。

```bash
python build_multilingual_pretrain_bin.py \
  --zh-path ./data/babylm/zho \
  --en-path ./data/babylm/eng_strict \
  --nl-path ./data/babylm/nld \
  --tokenizer-path ./chatglm_tokenizer/tokenizer.model \
  --budget-words 100000000 \
  --output ./data/merged_multilingual_zh1_en1_nl1_100m.bin
```

运行结束会打印每种语言的 token 数、raw words、adjusted words 与占比（`adj_share` 用于检查三语是否接近 1:1:1）。

---

## 2) 运行预训练（仅 pretrain）

训练入口：`pretrain.py`

### 重要说明（多卡前必看）

`pretrain.py` 顶部当前写死了：

```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

这会导致 **torchrun 多卡只能看到 1 张卡**。如果你要多卡预训练，请先删除/注释掉这两行，或改成你期望的卡集合（例如 `"0,1,2,3"`）。

### 数据输入与输出（默认）

`pretrain.py` 内部默认读取：

- `./data/merged_multilingual_zh4_en3_nl3_100m.bin`

并将 checkpoint 输出到：

- `out/pretrain/iter_*.pth`
- `out/pretrain/epoch_*.pth`

### 单卡（最稳妥）

```bash
python pretrain.py
```

### 多卡 DDP（示例：4 卡）

```bash
torchrun --standalone --nproc_per_node=4 pretrain.py
```

---

## 3) `.pth` 导出为 Hugging Face 标准目录（用于 BabyLM 官方 eval）

导出入口：`hf_remote_code/convert_pth_to_hf.py`

它会在目标目录写入（可被 Transformers 加载）：

- `config.json`
- `pytorch_model.bin`
- `configuration_babyllama_kimi.py` / `modeling_babyllama_kimi.py`（remote code）
- tokenizer 相关文件（包含 `tokenizer.model` 与 `tokenization_chatglm.py` 等）

### 导出命令（以 `pretrain.py` 默认 92M 配置为例）

`pretrain.py` 默认超参是：`dim=512, n_layers=8, n_heads=8, max_seq_len=512, vocab_size=64793, multiple_of=32, dropout=0.0`。  
导出时必须与训练时一致。

```bash
python hf_remote_code/convert_pth_to_hf.py \
  --pth out/pretrain/best.pth \
  --out_dir out/hf_babylm_ckpt \
  --dim 512 --n_layers 8 --n_heads 8 --n_kv_heads 8 \
  --vocab_size 64793 --multiple_of 32 --max_seq_len 512 --dropout 0.0
```

### HF 侧加载方式

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("out/hf_babylm_ckpt", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("out/hf_babylm_ckpt", trust_remote_code=True)
```

---

## 常见问题

### Q1：为什么 `.bin` 是 `uint16`？

因为使用的词表大小 `64793` 可以安全落在 `uint16` 范围内，训练数据存储更省空间。

### Q2：`pretrain.py` 想换 `.bin` 路径怎么办？

当前路径写在 `pretrain.py` 的 `data_path_list` 里（默认是 `./data/merged_multilingual_zh4_en3_nl3_100m.bin`）。

### Q3：导出 HF 时需要哪些文件？

`convert_pth_to_hf.py` 会自动把 `hf_remote_code/` 里的 modeling/config 文件，以及仓库根目录的 `chatglm_tokenizer/` 复制到导出目录；只要你的 `.pth` 路径与模型超参填写正确即可。