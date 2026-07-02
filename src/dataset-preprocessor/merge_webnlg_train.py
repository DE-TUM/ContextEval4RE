import os
import json

# --- 配置区域 ---
DATASET_PATH = r"\RE-analysis\datasets\webnlg"
TEST_DIR = os.path.join(DATASET_PATH, "test")
TRAIN_ROOT_DIR = os.path.join(DATASET_PATH, "train")  # 包含 1triples, 2triples 等的父目录
OUTPUT_DIR = os.path.join(DATASET_PATH, "train_merged")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def merge_train_data():
    # 1. 从 test 文件夹获取类别映射
    # 映射关系示例: "airport" -> {"num": "01", "name": "Airport"}
    category_map = {}
    for filename in os.listdir(TEST_DIR):
        if filename.endswith(".jsonl") and filename.startswith("ont_"):
            parts = filename.split('_')
            num = parts[1]
            cat_name = parts[2]
            category_map[cat_name.lower()] = {"num": num, "real_name": cat_name}

    print(f"Detected {len(category_map)} categories from test files.")

    # 2. 准备一个字典来存储每个类别合并后的数据
    merged_content = {cat_lower: [] for cat_lower in category_map.keys()}

    # 3. 遍历 train 文件夹下的所有子文件夹 (1triples, 2triples, ...)
    triple_folders = [f for f in os.listdir(TRAIN_ROOT_DIR) if 'triples' in f]
    
    for folder in triple_folders:
        folder_path = os.path.join(TRAIN_ROOT_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        
        print(f"Processing folder: {folder}")
        
        for file in os.listdir(folder_path):
            if not file.endswith(".jsonl"):
                continue
            
            # 提取文件名中的类别部分，例如 "Airport_allSolutions.jsonl" -> "airport"
            # 或者 "Airport.jsonl" -> "airport"
            raw_cat_name = file.replace("_allSolutions.jsonl", "").replace(".jsonl", "").lower()
            
            if raw_cat_name in merged_content:
                file_path = os.path.join(folder_path, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            merged_content[raw_cat_name].append(line.strip())
            else:
                # 某些类别（如 Film, MusicalWork）可能在某些 triple 文件夹中不存在
                pass

    # 4. 写入合并后的文件
    for cat_lower, lines in merged_content.items():
        info = category_map[cat_lower]
        num = info["num"]
        real_name = info["real_name"]
        
        # 构造目标文件名: ont_01_Airport_train.jsonl
        output_filename = f"ont_{num}_{real_name}_train.jsonl"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        if not lines:
            print(f"Warning: No training data found for {real_name}")
            continue
            
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + "\n")
        
        print(f"Saved {len(lines)} entries to {output_filename}")

if __name__ == "__main__":
    merge_train_data()