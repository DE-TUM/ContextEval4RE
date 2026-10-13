# ContextEval4RE

This repository contains the source code, datasets, prompts, LLM outputs, and evaluation results for our paper: 

**From Prompt to Context: Analyzing Context Engineering for LLM-based Closed-Schema Relation Extraction**  

---

## Contents

- [Repository Overview](#repository-overview)
- [Directory Structure](#directory-structure)
- [Quick Start](#quick-start)
- [Mapping to the Paper](#mapping-to-the-paper)
- [AI Assistance Disclosure](#ai-assistance-disclosure)
- [License](#license)

---

## Repository Overview
![Framework overview](assets/framework.png)

This repo supports:
- Loading multiple datasets with train/test splits and ontologies
- Composing context (relation option set + demonstration example set)
- Composing prompts (instruction, context, input sentence, output indicator)
- Interacting with LLMs (API or local deployment, depending on your setup)
- Evaluating extracted triples with standard metrics
- (Optional) Attention analysis on model outputs

---
## Directory Structure

High-level structure after unzipping:

```text
.
├── attention_analysis_results/
│   ├── attention_analysis_results/
│   └── attention_raw_results/
├── datasets/
│   ├── dbpedia_webnlg/
│   │   ├── examples_random/
│   │   ├── ground_truth/
│   │   ├── ontologies/
│   │   │   └── owl/
│   │   ├── test/
│   │   ├── test_ground_truth_relations/
│   │   ├── test_train_sent_similarity/
│   │   └── train/
│   ├── webnlg/              # same structure as dbpedia_webnlg/
│   └── wikidata_tekgen/      # same structure as dbpedia_webnlg/
├── evaluation_results/
├── eval_results/
├── llm_outputs/
├── prompts/
└── src/
    ├── attention_analysis/
    ├── context-composer/
    │   ├── example-selector/
    │   └── relation-selector/
    ├── dataset-preprocessor/
    ├── evaluator/
    ├── llm-interactor/
    └── prompt-composer/

```

Folder notes:

`datasets/`: processed dataset splits, ontologies, ground truth, and helper files.

`prompts/`: saved prompts used in experiments.

`llm_outputs/`: raw model outputs grouped by dataset/model/shot/setting.

`eval_results/`: evaluation results grouped by dataset/model/shot/setting; includes global_eval/ summaries.

`evaluation_results/`: summaries of evaluation results in `csv` format.

`attention_analysis/`: optional analysis outputs (raw attention + aggregated results).

`src/`: implementation of the framework modules shown in the framework figure.

## Quick Start

### 0) unzip the file
Unzip the provided zip file **to the repository root** (the same folder as `README.md`).

After extraction, you should see folders as shown in the Directory Structure

### 1) Environment setup

We recommend using **conda**:

```bash
conda create -n llm4kgc python=3.10
conda activate llm4kgc
pip install -r requirements.txt
```

### 2) Reproducing Experiments

#### Run the 16 prompt variations (paper settings) and evaluate

1) **Run API-based models (DeepSeek-V3 / GPT-4.1-nano)**  
Set your API key as an environment variable (`DEEPSEEK_API_KEY` or `OPENAI_API_KEY`, the latter possibly via a `.env` file), and select the `dataset` and `prompt_folder_name` (or `shot_number`) in:
- `src/llm-interactor/run_ds.py` (DeepSeek-V3)
- `src/llm-interactor/run_gpt.py` (GPT-4.1-nano)

Run:
```bash
python ./src/llm-interactor/run_ds.py
```
or
```bash
python ./src/llm-interactor/run_gpt.py
```
2) **Run locally deployed small/medium models**
Select the `model_id`, `datasets`, and `shot_number` in:

- `src/llm-interactor/run_local_llm.py` 

Run: 
```bash
python ./src/llm-interactor/run_local_llm.py
```
Expected outputs (raw LLM responses) will be saved under: `llm_outputs/` 


3) **Evaluate (same evaluation script as in the paper)**  
Make sure the folder structure matches the unzipped artifact. Then run (example: webnlg + gpt-4.1-nano + 1shot):
```bash
python src\evaluator\evaluate.py --dataset webnlg --model gpt-4.1-nano --shot 1shot
```
Expected evaluation outputs will be saved under: `eval_results/` 

4)  **Generate the summarized CSV** 
Run:
```bash
python src\evaluator\read_global_eval.py 
```
Expected CSV output will be saved under: `evaluation_results`



## Mapping to the Paper

Folder/setting names in the artifacts differ from the notation in the paper:

| Paper | Relation options $R$ | Examples $E$ | Setting folder / name in this repo |
|---|---|---|---|
| $P_{ref}$ | $R_{ref}$ (reference relations) | $E_{sim}$ | `gt_relations` |
| $P_{base}$ | $R_{domain}$ | $E_{sim}$ | `similar_example` |
| $P_{full}$ | $R_{full}$ | $E_{sim}$ | `merged_relations` |
| $P_{rand}$ | $R_{domain}$ | $E_{rand}$ | `random_example` |

Each setting is run with $k \in \{1, 3, 5, 10\}$ examples (`1shot`, `3shot`, `5shot`, `10shot`), giving 16 prompt variants per sentence. In the code, the paper's "schema" is referred to as "ontology".

## AI Assistance Disclosure

The code that produces the figures and the statistical analyses in the paper (plotting scripts, the paired bootstrap confidence intervals, and the attention-share/selectivity analysis in `src/attention_analysis/attention_selectivity.py`) was written with the help of Claude (Anthropic) as a coding assistant. The authors specified the analyses, reviewed the code, and checked the outputs against the paper's tables. The experiment pipeline itself (dataset preprocessing, context/prompt composition, LLM interaction, and evaluation under `src/`) was developed by the authors.

## License

- **Code**: MIT License. See `LICENSE`.
- **Data**: Please refer to the original dataset sources and their respective licenses (if applicable).
- **Precomputed outputs/results**: Provided for research and reproducibility purposes. Redistribution and downstream use should comply with the licenses/terms of the underlying datasets and the model providers.
