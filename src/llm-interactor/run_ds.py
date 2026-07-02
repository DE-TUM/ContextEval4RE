from openai import OpenAI
import time
import os
import json
from datetime import datetime
import sys
from llm_utils import load_jsonl, parse_result 

# --- configurations ---
client = OpenAI(api_key="###########", base_url="https://api.deepseek.com")

# dataset_name = "dbpedia_webnlg"
# dataset_name = "webnlg"
dataset_name = "wikidata_tekgen"
prompt_folder_name = "10shot_prompts"   
root_folder = r""

# run deepseek-v3
model_name = "deepseek-v3"

# --- 路径设置 ---
prompts_root = os.path.join(root_folder, "prompts")
prompts_folder = os.path.join(prompts_root, f"{dataset_name}_prompts", prompt_folder_name)
output_folder_path = os.path.join(root_folder, "llm_outputs_lg", dataset_name, model_name)   
os.makedirs(output_folder_path, exist_ok=True) 

# --- 日志设置 ---
start_processing_time = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'run_log_{dataset_name}_{model_name}_{prompt_folder_name}_{start_processing_time}.txt'
log_filepath = os.path.join(root_folder, 'logs')
os.makedirs(log_filepath, exist_ok=True)
log_filepath = os.path.join(log_filepath, log_filename)

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_filepath)

# --- 断点续传核心函数 ---
def get_processed_ids(file_path):
    """读取已存在的输出文件，返回已处理的 ID 集合"""
    processed_ids = set()
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "id" in data:
                        processed_ids.add(data["id"])
                except json.JSONDecodeError:
                    continue
    return processed_ids

print(f"Starting processing at {start_processing_time}")
print(f"Dataset: {dataset_name} | Model: {model_name} | Configuration: {prompt_folder_name}")

# --- 主循环 ---
for root, dirs, files in os.walk(prompts_folder):   
    for file in files: 
        if file.endswith('.jsonl'):
            print(f"\n{'='*40}\nProcessing file: {file}")
            
            rel_path = os.path.relpath(root, prompts_folder)
            output_subdir = os.path.join(output_folder_path, rel_path)
            os.makedirs(output_subdir, exist_ok=True)

            input_file_path = os.path.join(root, file)
            # 输出文件名包含模型、shot、数据集信息（通过路径和文件名体现）
            output_filename = file.split('_prompt')[0] + '_results.jsonl'
            output_file_path = os.path.join(output_subdir, output_filename)

            # 加载已处理的断点
            processed_ids = get_processed_ids(output_file_path)
            if processed_ids:
                print(f"Checkpoint found: {len(processed_ids)} items already processed. Resuming...")

            input_data = load_jsonl(input_file_path)

            # 使用 "a" (append) 模式打开文件
            with open(output_file_path, "a", encoding="utf-8") as f_out:
                for data in input_data:
                    current_id = data.get("id")
                    
                    # 检查是否在断点中
                    if current_id in processed_ids:
                        continue 

                    prompt = data.get("prompt")
                    if prompt is None:
                        print(f"ID {current_id}: Prompt not found, skipping.")
                        continue

                    try:
                        start_time = time.time()
                        response = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[{"role": "user", "content": prompt}],
                            stream=False,
                            temperature=0,
                            max_tokens=512
                        )
                        end_time = time.time()
                        
                        result = response.choices[0].message.content
                        run_time = end_time - start_time
                        usage = str(response.usage)
                        triples = parse_result(result)

                        output_item = {
                            "id": current_id,
                            "prompt": prompt,
                            "result": result,
                            "run_time": run_time,
                            "usage_metadata": usage,
                            "triples": triples
                        }

                        f_out.write(json.dumps(output_item, ensure_ascii=False) + "\n")
                        print(f"Done ID: {current_id} ({run_time:.2f}s)")

                    except Exception as e:
                        print(f"Error processing ID {current_id}: {e}")
                        print("Waiting 5 seconds before retrying...")
                        time.sleep(5)
                        

# --- 结束处理 ---
end_processing_time = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"\nProcessing completed at {end_processing_time}")