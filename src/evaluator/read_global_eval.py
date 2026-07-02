import os
import json
import csv

# === Locate root directories ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
# EVAL_ROOT = os.path.join(ROOT_DIR, "eval_results")
EVAL_ROOT = os.path.join(ROOT_DIR, "eval_results")

# === Output folder for analysis ===
ANALYSIS_DIR = os.path.join(ROOT_DIR, "evaluation_results")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(ANALYSIS_DIR, "global_summary.csv")

rows = []

print(f"🔍 Scanning evaluation results in: {EVAL_ROOT}\n")

for dataset_dir in os.listdir(EVAL_ROOT):
    dataset_path = os.path.join(EVAL_ROOT, dataset_dir)
    if not os.path.isdir(dataset_path):
        continue

    dataset_name = dataset_dir.replace("_eval_results", "")
    print(f"📁 Dataset: {dataset_name}")

    for model_dir in os.listdir(dataset_path):
        model_path = os.path.join(dataset_path, model_dir)
        if not os.path.isdir(model_path):
            continue

        model_name = model_dir
        print(f"  🤖 Model: {model_name}")

        # global_eval at model level
        global_eval_path = os.path.join(model_path, "global_eval")
        if not os.path.exists(global_eval_path):
            print(f"    ⚠️ No 'global_eval' folder found under {model_name}, skipping.")
            continue
        else:
            print(f"    ✅ Found global_eval folder: {global_eval_path}")

        for file in os.listdir(global_eval_path):
            if not file.endswith("_global_metrics.json"):
                continue

            file_path = os.path.join(global_eval_path, file)
            print(f"      📄 Reading metrics file: {file}")

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            except Exception as e:
                print(f"        ❌ Failed to load JSON: {e}")
                continue

            # Check if required keys exist
            expected_keys = ["avg_precision", "avg_recall", "avg_f1", "avg_rc_cand", "avg_rc_dom"]
            missing = [k for k in expected_keys if k not in metrics]
            if missing:
                print(f"        ⚠️ Missing keys in file: {missing}")
            else:
                print(f"        ✅ All required keys found.")

            # Parse setting and shot info
            base_name = file.replace("_global_metrics.json", "")
            parts = base_name.split("_")
            if len(parts) >= 3:
                shot = parts[-2]  # e.g., 1shot, 3shot, 5shot
                setting = "_".join(parts[:-2]) + "_result"
            else:
                shot = "unknown"
                setting = base_name

            def get_val(key):
                val = metrics.get(key)
                if isinstance(val, (int, float)):
                    return val
                try:
                    return float(str(val).replace("%", ""))
                except Exception:
                    return None

            row = {
                "dataset": dataset_name,
                "model": model_name,
                "shot": shot,
                "setting": setting,
                "avg_precision": get_val("avg_precision"),
                "avg_recall": get_val("avg_recall"),
                "avg_f1": get_val("avg_f1"),
                "avg_rc_cand": get_val("avg_rc_cand"),
                "avg_rc_dom": get_val("avg_rc_dom"),
            }

            rows.append(row)
            print(f"        ✅ Metrics added: precision={row['avg_precision']}, rc_dom={row['avg_rc_dom']}\n")

# === Sort rows ===
rows.sort(key=lambda r: (r["dataset"], r["model"], r["shot"], r["setting"]))

# === Write CSV ===
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = [
        "dataset", "model", "shot", "setting",
        "avg_precision", "avg_recall", "avg_f1",
        "avg_rc_cand", "avg_rc_dom"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\n✅ Evaluation summary saved to:", OUTPUT_CSV)
print(f"📊 Total entries collected: {len(rows)}")
