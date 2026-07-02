import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--shot", required=True)
args = parser.parse_args()

dataset_name = args.dataset
model_name = args.model
subfolder_name = args.shot

# Add the parent directory to the path so utils and eval_utils can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import eval_utils

############################## Change dataset and model directory
root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

data_folder = os.path.join(root_folder, "datasets")
result_folder = os.path.join(root_folder, "llm_outputs")

# Define evaluation results root
eval_root_folder = os.path.join(root_folder, "eval_results_robust")
os.makedirs(eval_root_folder, exist_ok=True)

############################## 在 root_folder 定义后添加
missing_log_path = os.path.join(eval_root_folder, "evaluation_issues_log.txt")
print(f"DEBUG: Log file will be saved at: {os.path.abspath(missing_log_path)}") # 添加这行
# 清理旧日志
if os.path.exists(missing_log_path):
    os.remove(missing_log_path)

# 在脚本开头定义完路径后立即执行
with open(missing_log_path, "w", encoding="utf-8") as f:
    f.write("--- Evaluation Issues Log Start ---\n")

def log_issue(reason, folder, filename, sent_id="N/A"):
    """统一记录错误的函数"""
    with open(missing_log_path, "a", encoding="utf-8") as f:
        log_entry = f"[{reason}] | Folder: {folder} | File: {filename} | ID: {sent_id}\n"
        f.write(log_entry)

############################## Define folders
ontology_folder = os.path.join(data_folder, dataset_name, "ontologies")
ground_truth_folder = os.path.join(data_folder, dataset_name, "ground_truth")

dataset_output_folder = f"{dataset_name}_llm_outputs"
model_folder = os.path.join(result_folder, dataset_output_folder, model_name)
output_folder = os.path.join(model_folder, subfolder_name)

# Create eval result path
dataset_eval_folder = f"{dataset_name}_eval_results"
# os.makedirs(dataset_eval_folder, exist_ok=True)
eval_model_folder = os.path.join(eval_root_folder, dataset_eval_folder, model_name)
os.makedirs(eval_model_folder, exist_ok=True)
eval_shot_folder = os.path.join(eval_model_folder, subfolder_name)
os.makedirs(eval_shot_folder, exist_ok=True)

############################## Walk through all subdirectories in the model folder
for setting_folder in os.listdir(output_folder):
    setting_path = os.path.join(output_folder, setting_folder)
    if not os.path.isdir(setting_path):
        log_issue("SKIP_NON_DIR", subfolder_name, setting_folder)
        continue

    eval_folder = os.path.join(eval_shot_folder, f"{setting_folder}_eval_results")
    os.makedirs(eval_folder, exist_ok=True)

    global_eval_folder = os.path.join(eval_model_folder, "global_eval")
    os.makedirs(global_eval_folder, exist_ok=True)

    # Initialize global counters
    total_files = 0
    t_p, t_r, t_f1, t_rc_cand, t_rc_dom = 0, 0, 0, 0, 0
    total_sentences = 0

    for filename in os.listdir(setting_path):
        if not filename.endswith(".jsonl"):
            log_issue("SKIP_NON_JSONL", setting_folder, filename)
            continue

        total_files += 1
        file_path = os.path.join(setting_path, filename)
        print(f"Processing file: {file_path}")
        system_output = eval_utils.load_jsonl(file_path)
        system_output = eval_utils.convert_to_dict(system_output)

        ontology_id = "_".join(filename.split("_")[:2])
        ground_truth_file_path = os.path.join(
            ground_truth_folder, f"ont_{ontology_id}_ground_truth.jsonl"
        )

        ground_truth = eval_utils.load_jsonl(ground_truth_file_path)
        ground_truth = eval_utils.convert_to_dict(ground_truth)

        ########## Load ontology and determine candidate relations
        if "merged" in filename:
            # For merged setting, use merged ontology relations
            ontology_file_path = os.path.join(ontology_folder, "merged_ontology.json")
            cand_relation = eval_utils.load_json(ontology_file_path)
            prompt_rels = [
                rel["label"].replace(" ", "_")
                for rel in cand_relation.get("relations", [])
            ]
        elif "gt" in filename:
            # For gt setting, use ground truth relations directly
            prompt_rels = set()
        else:
            # For dom setting, use domain ontology relations
            ontology_file_path = os.path.join(
                ontology_folder, f"{ontology_id}_ontology.json"
            )
            cand_relation = eval_utils.load_json(ontology_file_path)
            prompt_rels = [
                rel["label"].replace(" ", "_")
                for rel in cand_relation.get("relations", [])
            ]

        ########## Load domain ontology for rc_dom
        domain_ontology_file_path = os.path.join(
            ontology_folder, f"{ontology_id}_ontology.json"
        )
        domain_ontology = eval_utils.load_json(domain_ontology_file_path)
        domain_rels = [rel['label'].replace(" ", "_") for rel in domain_ontology.get("relations", [])]
        
        ########## Initialize per-file metrics
        f_p, f_r, f_f1, f_rc_cand, f_rc_dom = 0, 0, 0, 0, 0
        file_eval_metrics = []

        for sent_id, data in system_output.items():
            if "result" not in data:
                log_issue("MISSING_RESULT", setting_folder, filename, sent_id)
                continue
            # post-process LLM result and get predicted triples 
            llm_result = data["result"]
            system_triples = eval_utils.robust_parse_triples(llm_result)
            if sent_id not in ground_truth:
                log_issue("ID_NOT_IN_GROUND_TRUTH", setting_folder, filename, sent_id)
                continue
            # post-process ground truth triples    
            gt_triples = [
                [tr["sub"], tr["rel"], tr["obj"]]
                for tr in ground_truth[sent_id]["triples"]
            ]
            gt_relations = {tr[1].replace(" ", "_") for tr in gt_triples}
            # not using filtering anymore
            #filtered_system_triples = [
            #    tr for tr in system_triples if tr[1] in gt_relations
            #]
            sentence = ground_truth[sent_id]["sent"]
            # Normalize triples for evaluation
            normalized_system = {
                eval_utils.normalize_triple(*tr) for tr in system_triples
            }
            normalized_gt = {
                eval_utils.normalize_triple(*tr) for tr in gt_triples
            }
            # Calculate precision, recall, f1
            precision, recall, f1 = eval_utils.calculate_precision_recall_f1(
                normalized_gt, normalized_system
            )

            # Compute rc_cand
            if "gt" in filename:
                # Use gt relations as prompt_rels
                prompt_rels = gt_relations
            rc_cand = eval_utils.get_rc_cand(system_triples, prompt_rels)
            rc_dom = eval_utils.get_rc_dom(system_triples, domain_rels)


            f_p += precision
            f_r += recall
            f_f1 += f1
            f_rc_cand += rc_cand
            f_rc_dom += rc_dom

            file_eval_metrics.append(
                {
                    "id": sent_id,
                    "sent": sentence,
                    "gt_triples": gt_triples,
                    "llm_triples": system_triples,
                    "total_gt_triples": len(gt_triples),  # ✅ new field
                    "total_llm_triples": len(system_triples),  # ✅ new field
                    "gt_relations": [tr[1] for tr in gt_triples],  # ✅ new field
                    "llm_relations": [tr[1] for tr in system_triples],  # ✅ new field
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "rc_cand": round(rc_cand, 4),
                    "rc_dom": round(rc_dom, 4),
                    "normalized_llm_triples": list(normalized_system),
                    "normalized_gt_triples": list(normalized_gt),
                    "candidate relations": len(list(prompt_rels)),  # ✅ new field,
                    "domain relations": len(domain_rels),  # ✅ new field
                }
            )

        total_sents = len(system_output)
        total_sentences += total_sents
        eval_utils.save_json(
            file_eval_metrics,
            os.path.join(eval_folder, f"{filename.split('.')[0]}_eval_output.json"),
        )

        if total_sents > 0:
            file_avg_metrics = {
                "avg_precision": round(f_p / total_sents, 4),
                "avg_recall": round(f_r / total_sents, 4),
                "avg_f1": round(f_f1 / total_sents, 4),
                "avg_rc_cand": round(f_rc_cand / total_sents, 4),
                "avg_rc_dom": round(f_rc_dom / total_sents, 4),
            }
            eval_utils.save_json(
                file_avg_metrics,
                os.path.join(eval_folder, f"{filename.split('.')[0]}_metrics.json"),
            )

            # Update global counters
            t_p += f_p
            t_r += f_r
            t_f1 += f_f1
            t_rc_cand += f_rc_cand
            t_rc_dom += f_rc_dom

    if total_files > 0 and total_sentences > 0:
        folder_avg_metrics = {
            "avg_precision": round(t_p / total_sentences, 4),
            "avg_recall": round(t_r / total_sentences, 4),
            "avg_f1": round(t_f1 / total_sentences, 4),
            "avg_rc_cand": round(t_rc_cand / total_sentences, 4),
            "avg_rc_dom": round(t_rc_dom / total_sentences, 4),
        }
        eval_utils.save_json(
            folder_avg_metrics,
            os.path.join(global_eval_folder, f"{setting_folder}_global_metrics.json"),
        )
