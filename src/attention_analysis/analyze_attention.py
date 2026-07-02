import os
import numpy as np
import json
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = False 

def find_token_indices(tokens, keywords):

    clean_tokens = [
        t.replace('\u0120', '').replace('\u010a', '').lower().strip() 
        for t in tokens
    ]

    target = "".join(keywords).lower().replace(" ", "")
    
    for window_size in [1, 2, 3, 4]:
        for i in range(len(clean_tokens) - window_size + 1):
            combined_window = "".join(clean_tokens[i : i + window_size])
            if target in combined_window:
                return i
    return -1

def analyze_qwen_and_save(data_dir, output_viz_dir):
    if not os.path.exists(output_viz_dir):
        os.makedirs(output_viz_dir)


    folder_mapping = {
        "gt_relations_3shot_prompts": r"$P_{\mathrm{ref}}$",
        "similar_example_3shot_prompts": r"$P_{\mathrm{base}}$",
        "merged_relations_3shot_prompts": r"$P_{\mathrm{noise}}$",
        "random_example_3shot_prompts": r"$P_{\mathrm{random}}$"
    }
    ordered_folders = [
        "gt_relations_3shot_prompts",
        "similar_example_3shot_prompts",
        "merged_relations_3shot_prompts",
        "random_example_3shot_prompts"
    ]

    for case_name in os.listdir(data_dir):
        case_path = os.path.join(data_dir, case_name)
        if not os.path.isdir(case_path): continue
        
        print(f"📊 正在生成高清语义图: {case_name}")
        fig, axes = plt.subplots(4, 1, figsize=(20, 26))
        
        for idx, folder in enumerate(ordered_folders):
            ax = axes[idx]
            attn_path = os.path.join(case_path, f"{folder}_attn.npy")
            in_tok_path = os.path.join(case_path, f"{folder}_in_toks.json")
            out_tok_path = os.path.join(case_path, f"{folder}_out_toks.json")
            
            if not os.path.exists(attn_path):
                ax.set_title(f"Data missing for {folder}")
                continue
            

            attn = np.load(attn_path)
            with open(in_tok_path, 'r') as f: in_toks = json.load(f)
            with open(out_tok_path, 'r') as f: out_toks = json.load(f)


            idx_ont = find_token_indices(in_toks, ["ont", "ology", "relations"])
            idx_ex = find_token_indices(in_toks, ["example", "sentence"])
            idx_test = find_token_indices(in_toks, ["test", "sentence"])

            steps = np.arange(len(out_toks))
            clean_out_toks = [t.replace('\u0120', ' ').replace('\u010a', '\n') for t in out_toks]

            if idx_test != -1:
                test_focus = attn[:, idx_test:].mean(axis=1)
                ax.plot(steps, test_focus, label=f"Test Focus (avg: {np.mean(test_focus):.6f})", 
                        color='tab:blue', marker='o', markersize=3, linewidth=2)
            
            if idx_ex != -1 and idx_test > idx_ex:
                ex_focus = attn[:, idx_ex:idx_test].mean(axis=1)
                ax.plot(steps, ex_focus, label=f"Example Focus (avg: {np.mean(ex_focus):.6f})", 
                        color='tab:orange', linestyle='--', alpha=0.7)
            
            if idx_ont != -1 and idx_ex > idx_ont:
                ont_focus = attn[:, idx_ont:idx_ex].mean(axis=1)
                ax.plot(steps, ont_focus, label=f"Ontology Focus (avg: {np.mean(ont_focus):.6f})", 
                        color='tab:green', linestyle=':', alpha=0.6)

            ax.set_xticks(steps)
            ax.set_xticklabels(clean_out_toks, rotation=45, ha='right', fontsize=9)
            ax.set_title(f"Prompt Strategy: {folder_mapping[folder]}", loc='left', fontsize=16, fontweight='bold')
            ax.set_ylabel("Mean Attention Intensity")
            ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=10)
            ax.grid(True, axis='y', alpha=0.2)

        plt.suptitle(f"Semantic Attention Analysis: {case_name}\n(Higher Blue Peak indicates better context localization)", fontsize=22, y=0.99)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        save_base = os.path.join(output_viz_dir, f"{case_name}_semantic")
        plt.savefig(f"{save_base}.png", dpi=300) 
        plt.savefig(f"{save_base}.pdf")          
        plt.close()

if __name__ == "__main__":
    analyze_qwen_and_save('qwen_attention_data_10', 'final_qwen_viz_10')