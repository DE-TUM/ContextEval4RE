import os
import json
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# --- 配置区域 ---
os.environ["OPENAI_API_KEY"] = "########################################"  # 替换为你的 OpenAI API Key
client = OpenAI()

DATASET_PATH = r"ContextEval4RE\datasets\dbpedia_webnlg"
TEST_DIR = os.path.join(DATASET_PATH, "test")
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
OLD_SIM_DIR = os.path.join(DATASET_PATH, "test_train_sent_similarity_5")
OUTPUT_DIR = os.path.join(DATASET_PATH, "test_train_sent_similarity")
LOG_FILE = os.path.join(DATASET_PATH, "update_similarity.log")
SCORE_FILE = os.path.join(DATASET_PATH, "similarity_with_score.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_embeddings(text_list):
    """获取 OpenAI Embeddings"""
    # 增加一个小保护：如果列表为空直接返回
    if not text_list: return []
    response = client.embeddings.create(
        input=text_list,
        model="text-embedding-3-small"
    )
    return [data.embedding for data in response.data]

def load_jsonl(file_path):
    data = {}
    if not os.path.exists(file_path): return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data[item['id']] = item['sent']
    return data

# --- 主程序 ---
all_scores_record = {}
log_entries = []
total_test_sentences = 0
inconsistent_count = 0

# 1. 更加健壮的类别提取：提取 ont_{N}_ 中的 N
# 确保 1-19 都能被识别为独立的数字
test_files = [f for f in os.listdir(TEST_DIR) if f.endswith('.jsonl')]
cat_ids = []
for f in test_files:
    parts = f.split('_')
    if len(parts) > 1:
        cat_ids.append(parts[1]) # 这里的 parts[1] 就是数字 ID

for cat_idx in cat_ids:
    print(f"Checking Category {cat_idx}...")
    
    # 2. 精确匹配文件：确保 ont_1_ 不会匹配到 ont_11_
    # 使用 f"ont_{cat_idx}_" 作为前缀匹配
    test_file = next((f for f in test_files if f.startswith(f"ont_{cat_idx}_")), None)
    
    train_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.jsonl')]
    train_file = next((f for f in train_files if f.startswith(f"ont_{cat_idx}_")), None)
    
    old_sim_files = os.listdir(OLD_SIM_DIR)
    # 原始相似度文件格式多样（有的是 1_... 有的是 10_...），
    # 我们可以通过匹配文件名开头的数字加下划线来精确定位
    old_sim_file = next((f for f in old_sim_files if f.startswith(f"{cat_idx}_")), None)

    if not (test_file and train_file and old_sim_file):
        print(f"  [Skip] Missing files for Category {cat_idx}. (Train: {bool(train_file)}, OldSim: {bool(old_sim_file)})")
        continue

    print(f"  [Processing] {test_file}")

    # --- (后续逻辑与之前一致) ---
    test_data = load_jsonl(os.path.join(TEST_DIR, test_file))
    train_data = load_jsonl(os.path.join(TRAIN_DIR, train_file))
    with open(os.path.join(OLD_SIM_DIR, old_sim_file), 'r', encoding='utf-8') as f:
        old_sim_data = json.load(f)

    train_ids = list(train_data.keys())
    train_sents = list(train_data.values())
    test_ids = list(test_data.keys())
    test_sents = list(test_data.values())

    # 调用 API
    train_embeds = get_embeddings(train_sents)
    test_embeds = get_embeddings(test_sents)

    new_sim_records = {}
    similarity_matrix = cosine_similarity(test_embeds, train_embeds)

    for i, t_id in enumerate(test_ids):
        total_test_sentences += 1
        scores = [{"id": train_ids[j], "score": float(similarity_matrix[i][j])} for j in range(len(train_ids))]
        scores_sorted = sorted(scores, key=lambda x: x['score'], reverse=True)
        all_scores_record[t_id] = scores_sorted

        original_5 = old_sim_data.get(t_id, [])
        new_5 = []
        for item in scores_sorted:
            if item['id'] not in original_5:
                new_5.append(item['id'])
            if len(new_5) == 5: break
        
        new_sim_records[t_id] = original_5 + new_5

        current_top_10_ids = [s['id'] for s in scores_sorted[:10]]
        if not all(oid in current_top_10_ids for oid in original_5):
            inconsistent_count += 1
            log_entries.append(f"Inconsistent: {t_id}")

    with open(os.path.join(OUTPUT_DIR, old_sim_file), 'w', encoding='utf-8') as f:
        json.dump(new_sim_records, f, indent=1)

# 保存全局结果
with open(SCORE_FILE, 'w', encoding='utf-8') as f:
    json.dump(all_scores_record, f, indent=2)

with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write("\n".join(log_entries) + f"\n\nTotal: {total_test_sentences}, Inconsistent: {inconsistent_count}")

print(f"Successfully processed all categories.")