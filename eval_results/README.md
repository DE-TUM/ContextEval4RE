# eval_results

This folder contains **evaluation outputs and aggregated metrics** computed from `llm_outputs/`
using the evaluation scripts in `src/evaluator/`.

## Expected structure

```text
eval_results/
└── <dataset>_eval_results/
    └── <model_name>/
        ├── global_eval/
        │   ├── <setting>_global_metrics.json
        │   └── ...
        └── <k-shot>/
            └── <setting>_eval_results/
                ├── *_eval_output.json
                ├── *_metrics.json
                └── ...
```
- `<dataset>`: `dbpedia_webnlg`, `webnlg`, `wikidata_tekgen`
- `<model_name>`: e.g., `gpt-4.1-nano`, `deepseek-v3`, `Llama-3.1-8B-Instruct`, `Qwen2.5-14B-Instruct`
- `<k-shot>`: `1shot`, `3shot`, `5shot`, `10shot`
- `<setting>`: prompt/context strategy used in the paper (e.g., `gt_relations`, `merged_relations`, `random_example`, `similar_example`)

## What is inside?

### 1) Per-file evaluation outputs
Located under:  
`<k-shot>/<setting>_eval_results/`

- `*_eval_output.json`: detailed per-instance records (predicted triples, GT triples, precision/recall/F1, rc metrics, etc.)
- `*_metrics.json`: average metrics for that file

### 2) Global aggregated metrics
Located under:  
`global_eval/`

- `<setting>_global_metrics.json`: metrics aggregated over all files for a given setting

## How to generate

Example (`webnlg` + `gpt-4.1-nano` + `1shot`):

**Windows**
```bash
python src\evaluator\evaluate.py --dataset webnlg --model gpt-4.1-nano --shot 1shot
``` 
**Linux/macOS**
```bash
python src/evaluator/evaluate.py --dataset webnlg --model gpt-4.1-nano --shot 1shot
``` 
## Notes

- The evaluator assumes the folder hierarchy under `llm_outputs/` is unchanged.
- If you rename files or move folders, evaluation may fail or produce incomplete summaries.
- To generate summarized CSV files, run:
```bash
python src\evaluator\read_global_eval.py
``` 
Outputs will be saved under `evaluation_results/`.
