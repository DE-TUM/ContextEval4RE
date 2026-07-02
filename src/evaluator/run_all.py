import os
import subprocess

# Absolute path to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Absolute path to project root (two levels up)
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
# Absolute path to evaluate.py
EVAL_SCRIPT = os.path.join(SCRIPT_DIR, "evaluate.py")

# datasets = ["webnlg"]
# models = ["deepseek-v3", "gpt-4.1-nano"]
# models = ["gemma-3-12b-it", "Llama-3.1-8B-Instruct", "Qwen2.5-14B-Instruct"]
# models = ["gemma-3-1b-it", "Llama-3.2-1B-Instruct", "Qwen2.5-1.5B-Instruct"]
datasets = ["webnlg", "dbpedia_webnlg", "wikidata_tekgen"]
# datasets = ["wikidata_tekgen"]
models = ["deepseek-v3", "gpt-4.1-nano", "gemma-3-12b-it", "Llama-3.1-8B-Instruct", "Qwen2.5-14B-Instruct", "gemma-3-1b-it", "Llama-3.2-1B-Instruct", "Qwen2.5-1.5B-Instruct"]
shots = ["1shot", "3shot", "5shot", "10shot"]

for dataset in datasets:
    for model in models:
        for shot in shots:
            print(f"\n=== Running {dataset} | {model} | {shot} ===")
            cmd = [
                "python",
                EVAL_SCRIPT,           # use absolute path to evaluate.py
                "--dataset", dataset,
                "--model", model,
                "--shot", shot
            ]
            # Run from project root so relative paths inside evaluate.py work correctly
            subprocess.run(cmd, cwd=ROOT_DIR)
