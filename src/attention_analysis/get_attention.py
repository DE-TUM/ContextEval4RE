import os
import json
import torch
import gc
import numpy as np
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-1.5B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype="auto",
    trust_remote_code=True,
    output_attentions=True 
)
print(f"✅ Qwen Model loaded successfully on {model.device}\n")

def run_qwen_inference(input_file, output_root):
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data_id, category = data['id'], data.get('category', 'unknown')
            prompts_dict = data['prompts']
            
            case_dir = os.path.join(output_root, f"{category}_{data_id}")
            os.makedirs(case_dir, exist_ok=True)

            for folder_name, prompt_content in prompts_dict.items():
                print(f"🚀 Processing: {data_id} - {folder_name}")
                

                messages = [{"role": "user", "content": prompt_content}]
                text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
                input_len = model_inputs.input_ids.shape[-1]

                with torch.no_grad():
                    outputs = model.generate(
                        **model_inputs,
                        max_new_tokens=256,
                        do_sample=False, 
                        output_attentions=True,
                        return_dict_in_generate=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )

                # --- 提取注意力 (取最后一层，所有头平均) ---
                all_step_attn = []
                for step_attn_layers in outputs.attentions:
                    # step_attn_layers[layer_idx] -> (batch, head, query, key)
                    last_layer_attn = step_attn_layers[-1].mean(1)[0].to(torch.float32).cpu().numpy()
                    
                    if last_layer_attn.ndim == 2:
                        last_layer_attn = last_layer_attn[-1]
                    
                    # 只保存对 Input Prompt 的注意力
                    all_step_attn.append(last_layer_attn[:input_len])

                # --- 保存数据 ---
                prefix = os.path.join(case_dir, folder_name)
                
                # 1. 保存原始注意力矩阵
                np.save(f"{prefix}_attn.npy", np.array(all_step_attn))
                
                # 2. 保存输入 Token (用于后续 find_token_indices)
                in_toks = tokenizer.convert_ids_to_tokens(model_inputs.input_ids[0])
                with open(f"{prefix}_in_toks.json", "w") as f_in:
                    json.dump(in_toks, f_in)
                
                # 3. 保存输出 Token (用于 X 轴标签)
                out_tok_ids = outputs.sequences[0][input_len:]
                out_toks = tokenizer.convert_ids_to_tokens(out_tok_ids)
                with open(f"{prefix}_out_toks.json", "w") as f_out:
                    json.dump(out_toks, f_out)
                
                # 4. 保存人眼可读文本
                gen_text = tokenizer.decode(out_tok_ids, skip_special_tokens=True)
                with open(f"{prefix}_gen_text.txt", "w") as f_txt:
                    f_txt.write(gen_text)

                # 清理显存
                del model_inputs, outputs, all_step_attn
                torch.cuda.empty_cache()
                gc.collect()

run_qwen_inference('webnlg_prompts_sample.jsonl', 'qwen_attention_data')