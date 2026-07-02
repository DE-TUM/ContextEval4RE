import os
import json
import sys
import time
from datetime import datetime
from openai import OpenAI
from llm_utils import load_jsonl  # must return a list of dicts
from dotenv import load_dotenv
load_dotenv()

import os
print(os.getenv("OPENAI_API_KEY"))


# ============================================================
# USER CONFIGURATION
# ============================================================

# ---- Select dataset and shot number manually ----

# dataset_name = "webnlg"
# dataset_name = "wikidata_tekgen"
dataset_name = "dbpedia_webnlg"
shot_number = "10shot"  # choose among "1shot", "3shot", "5shot", "10shot"

# ---- Model configuration ----
model_name = "gpt-4.1-nano"  
temperature = 0
max_tokens = 512

# ---- Path configuration ----
root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


input_folder = os.path.join(
    root_folder, "prompts", f"{dataset_name}_prompts", f"{shot_number}_prompts"
)
output_root = os.path.join(
    root_folder, "llm_outputs_lg", f"{dataset_name}_llm_outputs", model_name, shot_number
)
log_dir = os.path.join(root_folder, "logs")
os.makedirs(log_dir, exist_ok=True)

# ============================================================
# LOGGING
# ============================================================

start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(log_dir, f"run_log_{dataset_name}_{model_name}_{shot_number}_{start_time}.txt")

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

sys.stdout = Logger(log_path)

# ============================================================
# SET UP MODEL
# ============================================================
print(f"\n=== LLM Runner Started ===")
print(f"Dataset: {dataset_name}")
print(f"Shot number: {shot_number}")
print(f"Model: {model_name}")
print(f"Input folder: {input_folder}")
print(f"Output folder: {output_root}\n")

client = OpenAI()


# ============================================================
# PROCESS FILES
# ============================================================

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
    percent = current / total
    filled = int(bar_length * percent)
    bar = "#" * filled + "-" * (bar_length - filled)
    print(f"\r[{bar}] {current}/{total} prompts processed", end="", flush=True)

file_count = 0
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

        print(f"\n\nProcessing file {file_count}: {input_path}")
        print("-" * 80)
        print(f"Results will be saved to: {output_path}")

        data_list = load_jsonl(input_path)
        processed_ids = load_checkpoint(checkpoint_path)

        total = len(data_list)
        processed_count = len(processed_ids)
        
        # 在终端与日志中同时显示断点信息
        resume_msg = f"检测到已处理 {processed_count} 条，将从第 {processed_count + 1} 条继续运行（共 {total} 条）"
        print(resume_msg)

        # 额外写入日志文件，确保提示保存
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(resume_msg + "\n")

        processed = 0
        printed = 0

        with open(output_path, "a", encoding="utf-8") as f_out:
            for i, item in enumerate(data_list, start=1):
                data_id = item.get("id")
                prompt = item.get("prompt", "")

                if data_id in processed_ids:
                    processed += 1
                    progress_bar(processed, total)
                    continue

                try:
                    start_t = time.time()
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                    # Extract text and usage info
                    result = response.choices[0].message.content if response.choices else ""
                    usage_metadata = response.usage.to_dict() if hasattr(response, "usage") else {}
                    end_t = time.time()

                    output_item = {
                        "id": data_id,
                        "prompt": prompt,
                        "result": result,
                        "run_time": end_t - start_t,
                        "usage_metadata": usage_metadata
                    }

                    f_out.write(json.dumps(output_item, ensure_ascii=False) + "\n")
                    f_out.flush()

                    processed_ids.add(data_id)
                    processed += 1
                    progress_bar(processed, total)

                    # 💾 每处理一条数据就立即保存断点
                    save_checkpoint(checkpoint_path, processed_ids)

                    if i % 10 == 0 or i == 1:
                        print(f"\n\n--- Result {i} ---")
                        print(result)
                        print("-" * 80)

                except Exception as e:
                    print(f"\nError for id={data_id}: {e}")
                    continue

        save_checkpoint(checkpoint_path, processed_ids)
        print(f"\n✅ Completed {file} ({processed}/{total})")

print("\n\n=== All files processed successfully ===")
print(f"Logs saved to: {log_path}")
