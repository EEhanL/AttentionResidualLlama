import os
from datasets import load_dataset

# 定义要下载的语言列表
languages = ["nld", "zho", "2026-Strict"] 
save_name = ["nld", "zho", "eng_strict"]
base_dir = "data/babylm"

for i in range(len(languages)):
    dataset_name = f"BabyLM-community/babylm-{languages[i]}"
    print(f"正在下载: {dataset_name}...")
    
    # 加载数据集
    ds = load_dataset(dataset_name)
    
    # 定义保存路径
    save_path = os.path.join(base_dir, f"{save_name[i]}")
    
    # 保存到磁盘
    ds.save_to_disk(save_path)
    print(f"已保存至: {save_path}")