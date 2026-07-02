import os
import numpy as np
import json
import pandas as pd

def find_token_indices(tokens, keywords):
    # 1. 预处理：去掉 Qwen 特有的空格(\u0120)和换行(\u010a)符号
    clean_tokens = [
        t.replace('\u0120', '').replace('\u010a', '').lower().strip() 
        for t in tokens
    ]
    
    # 2. 构造目标紧凑字符串（例如 "ontologyrelations"）
    target = "".join(keywords).lower().replace(" ", "")
    
    # 3. 滑动窗口
    for window_size in [1, 2, 3, 4]: 
        for i in range(len(clean_tokens) - window_size + 1):
            combined_window = "".join(clean_tokens[i : i + window_size])
            if target in combined_window:
                return i
                
    # 4. 兜底逻辑：如果完全匹配失败，尝试寻找包含 target 的单个 token
    for i, t in enumerate(clean_tokens):
        if len(t) > 2 and (target in t or t in target):
            return i
            
    return -1

def run_quantification(data_dir, output_csv):
    results = []
    folders = [
        "gt_relations_3shot_prompts", 
        "merged_relations_3shot_prompts", 
        "random_example_3shot_prompts", 
        "similar_example_3shot_prompts"
    ]

    cases = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for case_name in cases:
        case_path = os.path.join(data_dir, case_name)
        
        for folder in folders:
            attn_path = os.path.join(case_path, f"{folder}_attn.npy")
            in_tok_path = os.path.join(case_path, f"{folder}_in_toks.json")
            
            if not os.path.exists(attn_path): continue
            
            # 加载数据
            attn = np.load(attn_path) # (Gen_Len, In_Len)
            with open(in_tok_path, 'r') as f: in_toks = json.load(f)

            # 定位关键索引
            # 1. 定位 Ontology 定义区的开始 (匹配 "Ontology Relations:")
            idx_ont = find_token_indices(in_toks, ["ont", "ology", "relations"])

            # 2. 定位示例区的开始 (匹配 "Example Sentence:")
            idx_ex = find_token_indices(in_toks, ["example", "sentence"])

            # 3. 定位测试区的开始 (匹配 "Test Sentence:")
            idx_test = find_token_indices(in_toks, ["test", "sentence"])

            entry = {
                "Case_ID": case_name,
                "Prompt_Type": folder.replace("_3shot_prompts", "")
            }

            # --- A. Mean_Test_Focus ---
            if idx_test != -1:
                test_focus_seq = attn[:, idx_test:].mean(axis=1)
                entry["Mean_Test_Focus"] = np.mean(test_focus_seq)
            else:
                entry["Mean_Test_Focus"] = np.nan

            # --- B. Mean_Example_Focus ---
            if idx_ex != -1 and idx_test > idx_ex:
                ex_focus_seq = attn[:, idx_ex:idx_test].mean(axis=1)
                entry["Mean_Example_Focus"] = np.mean(ex_focus_seq)
            else:
                entry["Mean_Example_Focus"] = np.nan

            # --- C. Mean_Ontology_Focus (新添加) ---
            # 计算区间：从 Ontology 开始到 Example 开始之前
            if idx_ont != -1 and idx_ex > idx_ont:
                ont_focus_seq = attn[:, idx_ont:idx_ex].mean(axis=1)
                entry["Mean_Ontology_Focus"] = np.mean(ont_focus_seq)
            else:
                entry["Mean_Ontology_Focus"] = np.nan

            results.append(entry)

    # 转换为 DataFrame
    df = pd.DataFrame(results)
    
    # 计算汇总
    summary = df.groupby("Prompt_Type")[["Mean_Test_Focus", "Mean_Ontology_Focus", "Mean_Example_Focus"]].mean()
    summary = summary.sort_values(by="Mean_Test_Focus", ascending=False)
    
    print("\n--- 汇总结果预览 (按 Mean_Test_Focus 排序) ---")
    print(summary)
    
    df.to_csv(output_csv, index=False)
    print(f"\n✅ 包含 Ontology Focus 的数据已保存至: {output_csv}")

if __name__ == "__main__":
    run_quantification('qwen_attention_data_10', 'attention_quant_results_10.csv')