import os
import json
import sys
import time
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 用户配置
# ============================================================
# 修改点 1：定义数据集列表
datasets = ["wikidata_tekgen", "dbpedia_webnlg", "webnlg"]
shot_number = "10shot" 

# ---------------------sm model 配置 ---------------------
# model_id = "meta-llama/Llama-3.2-1B-Instruct"
# model_id = "google/gemma-3-1b-it"
# model_id = "Qwen/Qwen2.5-1.5B-Instruct"

# ---------------------md model 配置 ---------------------
# model_id = "meta-llama/Llama-3.1-8B-Instruct"
# model_id = "google/gemma-3-12b-it"
model_id = "Qwen/Qwen2.5-14B-Instruct"

max_new_tokens = 512
do_sample = False

# ============================================================
# 工具函数与类
# ============================================================

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_checkpoint_path(output_file):
    return output_file + ".checkpoint"

def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_checkpoint(checkpoint_path, processed_ids):
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(list(processed_ids), f)

def progress_bar(current, total, bar_length=30):
    percent = current / total if total > 0 else 0
    filled = int(bar_length * percent)
    bar = "#" * filled + "-" * (bar_length - filled)
    print(f"\r[{bar}] {current}/{total} prompts processed", end="", flush=True)

# ============================================================
# 模型加载 (只需加载一次)
# ============================================================

print(f"\n=== Loading model {model_id} ===")
if torch.cuda.is_available():
    device = "cuda"
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("❌ No GPU detected.")
    sys.exit(1)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype="auto",
    trust_remote_code=True,
    local_files_only=True
)
print(f"✅ Model loaded successfully.\n")

# 保存原始 stdout 以便在切换日志时恢复
original_stdout = sys.stdout

# ============================================================
# 主循环：遍历多个数据集
# ============================================================

for dataset_name in datasets:
    # --------------------- 路径初始化 ---------------------
    root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    input_folder = os.path.join(root_folder, "prompts", f"{dataset_name}_prompts", f"{shot_number}_prompts")
    output_root = os.path.join(root_folder, "llm_outputs", f"{dataset_name}_llm_outputs", model_id.split("/")[-1], shot_number)
    log_dir = os.path.join(root_folder, "logs")
    
    ensure_dir(log_dir)
    ensure_dir(output_root)

    # --------------------- 独立日志设置 ---------------------
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_log_{dataset_name}_{model_id.split('/')[-1]}_{shot_number}_{start_time}.txt")
    
    # 切换 stdout 到当前数据集的日志文件
    sys.stdout = Logger(log_path)
    
    print(f"🚀 Starting process for dataset: {dataset_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # --------------------- 处理当前数据集的文件 ---------------------
    file_count = 0
    if not os.path.exists(input_folder):
        print(f"⚠️ Warning: Input folder not found: {input_folder}")
        sys.stdout = original_stdout # 恢复以便打印错误
        continue

    for root, _, files in os.walk(input_folder):
        for file in files:
            if not file.endswith(".jsonl"):
                continue

            file_count += 1
            rel_path = os.path.relpath(root, input_folder)
            output_subdir = os.path.join(output_root, rel_path)
            ensure_dir(output_subdir)

            input_path = os.path.join(root, file)
            output_path = os.path.join(output_subdir, file.replace("_prompt", "_results"))
            checkpoint_path = get_checkpoint_path(output_path)

            print(f"\nProcessing file {file_count}: {file}")
            data_list = load_jsonl(input_path)
            processed_ids = load_checkpoint(checkpoint_path)

            total = len(data_list)
            processed = len(processed_ids)

            with open(output_path, "a", encoding="utf-8") as f_out:
                for i, item in enumerate(data_list, start=1):
                    data_id = item.get("id")
                    prompt = item.get("prompt", "")

                    if data_id in processed_ids:
                        processed += 1
                        progress_bar(processed, total)
                        continue

                    try:
                        messages = [{"role": "user", "content": prompt}]
                        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

                        start_t = time.time()
                        with torch.no_grad():
                            outputs = model.generate(
                                **model_inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=do_sample,
                            )
                        end_t = time.time()

                        generated_ids = [out[len(ins):] for ins, out in zip(model_inputs.input_ids, outputs)]
                        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                        output_item = {
                            "id": data_id,
                            "prompt": prompt,
                            "result": decoded,
                            "run_time": end_t - start_t
                        }

                        f_out.write(json.dumps(output_item, ensure_ascii=False) + "\n")
                        f_out.flush()

                        processed_ids.add(data_id)
                        processed += 1
                        progress_bar(processed, total)

                        if i % 20 == 0:
                            save_checkpoint(checkpoint_path, processed_ids)

                    except Exception as e:
                        print(f"\n❌ Error for id={data_id} in {dataset_name}: {e}")
                        continue

            save_checkpoint(checkpoint_path, processed_ids)
            print(f"\n✅ Finished {file} ({processed}/{total})")

    print(f"\n✨ COMPLETED Dataset: {dataset_name}")
    print(f"Log closed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 完成一个数据集后，恢复原始 stdout 
    sys.stdout = original_stdout

print("\n\n🎉 All datasets processed successfully!")