#!/usr/bin/env python3
"""
验证权重绑定是否正确的脚本
"""
import torch
import sys
from pathlib import Path

def verify_pth_weight_tying(pth_path):
    """检查 .pth 文件中的权重绑定"""
    print(f"\n=== 检查 {pth_path} ===")
    state = torch.load(pth_path, map_location='cpu')
    
    # 去除可能的 _orig_mod. 前缀
    if any(k.startswith('_orig_mod.') for k in state.keys()):
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    
    tok_key = 'tok_embeddings.weight'
    out_key = 'output.weight'
    
    has_tok = tok_key in state
    has_out = out_key in state
    
    print(f"  tok_embeddings.weight 存在: {has_tok}")
    print(f"  output.weight 存在: {has_out}")
    
    if has_tok and has_out:
        # 检查是否是同一个张量（内存地址）
        are_same = state[tok_key].data_ptr() == state[out_key].data_ptr()
        are_equal = torch.equal(state[tok_key], state[out_key])
        print(f"  是否共享内存: {are_same}")
        print(f"  数值是否相等: {are_equal}")
        
        if not are_equal:
            diff = (state[tok_key] - state[out_key]).abs().max().item()
            print(f"  ⚠️  WARNING: 权重不相等！最大差异: {diff}")
            return False
        elif not are_same:
            print(f"  ⚠️  WARNING: 权重相等但不共享内存（可能在保存时被复制了）")
            return True  # 数值相等就可以
        else:
            print(f"  ✅ 权重绑定正确")
            return True
    elif has_tok and not has_out:
        print(f"  ✅ 只有 tok_embeddings.weight（符合预期，权重绑定）")
        return True
    elif has_out and not has_tok:
        print(f"  ⚠️  WARNING: 只有 output.weight，缺少 tok_embeddings.weight")
        return False
    else:
        print(f"  ❌ ERROR: 两个权重都不存在！")
        return False


def verify_hf_weight_tying(hf_dir):
    """检查 HF 格式模型的权重绑定"""
    print(f"\n=== 检查 HF 模型 {hf_dir} ===")
    hf_dir = Path(hf_dir)
    
    # 检查 config
    import json
    config_path = hf_dir / "config.json"
    if config_path.exists():
        config = json.load(open(config_path))
        tie_word_embeddings = config.get('tie_word_embeddings', True)
        print(f"  config.json 中 tie_word_embeddings: {tie_word_embeddings}")
    
    # 检查 pytorch_model.bin
    model_path = hf_dir / "pytorch_model.bin"
    if not model_path.exists():
        print(f"  ❌ pytorch_model.bin 不存在")
        return False
    
    state = torch.load(model_path, map_location='cpu')
    
    tok_key = 'model.tok_embeddings.weight'
    out_key = 'model.output.weight'
    
    has_tok = tok_key in state
    has_out = out_key in state
    
    print(f"  model.tok_embeddings.weight 存在: {has_tok}")
    print(f"  model.output.weight 存在: {has_out}")
    
    if has_tok and has_out:
        are_equal = torch.equal(state[tok_key], state[out_key])
        print(f"  数值是否相等: {are_equal}")
        if not are_equal:
            diff = (state[tok_key] - state[out_key]).abs().max().item()
            print(f"  ⚠️  WARNING: 权重不相等！最大差异: {diff}")
            return False
        else:
            print(f"  ⚠️  WARNING: 两个权重都存在且相等，但应该只保存一个")
            print(f"  （HF 会在加载时自动绑定，保存两份浪费空间）")
            return True
    elif has_tok and not has_out:
        print(f"  ✅ 只有 tok_embeddings.weight（正确！HF 会自动绑定）")
        return True
    elif has_out and not has_tok:
        print(f"  ⚠️  WARNING: 只有 output.weight，缺少 tok_embeddings.weight")
        return False
    else:
        print(f"  ❌ ERROR: 两个权重都不存在！")
        return False


def verify_hf_loading(hf_dir):
    """验证 HF 模型加载后权重是否正确绑定"""
    print(f"\n=== 验证 HF 模型加载 {hf_dir} ===")
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(hf_dir, trust_remote_code=True)
        
        tok_weight = model.model.tok_embeddings.weight
        out_weight = model.model.output.weight
        
        are_same = tok_weight.data_ptr() == out_weight.data_ptr()
        are_equal = torch.equal(tok_weight, out_weight)
        
        print(f"  加载后是否共享内存: {are_same}")
        print(f"  加载后数值是否相等: {are_equal}")
        
        if are_same:
            print(f"  ✅ 权重绑定正确！")
            return True
        else:
            print(f"  ❌ ERROR: 权重没有绑定！")
            return False
    except Exception as e:
        print(f"  ❌ ERROR: 加载失败: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  验证 .pth 文件: python verify_weight_tying.py path/to/model.pth")
        print("  验证 HF 目录: python verify_weight_tying.py path/to/hf_model_dir")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if path.endswith('.pth'):
        verify_pth_weight_tying(path)
    else:
        verify_hf_weight_tying(path)
        verify_hf_loading(path)
