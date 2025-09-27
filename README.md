## Humor Detection with Setup–Punchline Attention and Incongruity
Modeling - Humor Recognition Task

This repository contains a small NLP research codebase for humor detection on a combined dataset of jokes and puns. It includes:

- A raw data preprocessing script that creates clean, contamination-free splits
- Three model architectures (baseline CNN, BERT-based, proposed SPA model)
- A YAML-driven experiment runner with ready-made experiment suites
- Lightweight evaluation utilities and metrics


### TL;DR – Quickstart

```bash

# 1) Install dependencies
pip install -U torch transformers pandas numpy scikit-learn pyyaml tqdm

# 2) Prepare data (place the three CSVs under data/raw; see Data section)
python src/data/preprocess_raw_data.py

# 3) Run a quick test experiment
python run_combined_experiments.py --experiment quick_test
```


### Project Structure

```text
nlp_proj_new/
├─ data/
│  ├─ raw/                          # Place input CSVs here
│  └─ processed/                    # Preprocessed splits written here (humor_*.csv)
├─ experiments/
│  └─ yaml_driven/                  # Results, logs, summaries per run
├─ models/
│  └─ three_architectures/          # (Artifacts or related assets if any)
├─ src/
│  ├─ data/
│  │  └─ preprocess_raw_data.py     # Raw → processed CSVs with dedup & split
│  ├─ evaluation/
│  │  └─ metrics.py                 # Accuracy, Precision, Recall, F1; CSV loader
│  ├─ models/
│  │  ├─ baseline_model.py          # Simple CNN baseline (embedding + conv + max-pool)
│  │  ├─ transformer_model.py       # BERT/DistilBERT backbone + linear head
│  │  ├─ proposed_humor_model.py    # SPA (Setup–Punchline Attention) prototype
│  │  └─ model_factory.py           # Factory to instantiate models
│  ├─ training/
│  │  └─ trainer.py                 # Three-model comparison training flow
│  └─ utils/
│     ├─ config_loader.py           # YAML → dataclass configs
│     └─ focused_humor_experiments.py # Experiment runner and logging helpers
├─ experiment_configs_combined.yaml  # Main YAML config suite
├─ run_combined_experiments.py       # CLI to run YAML-defined experiments
└─ NLP_Project_Proposal.pdf          # Project proposal (reference)
```


### Data

This project combines three sources:

1) `shortjokes.csv` – jokes (all humorous → label 1)
2) `puns_pos_neg_data.csv` – puns (1 → humorous, -1 → not humorous; converted to 1/0)
3) `kaggle_dataset.csv` – mixed data; only rows with humor flag False are used (label 0)

Place the CSVs under `data/raw/` with exactly these filenames:

- `data/raw/shortjokes.csv`
- `data/raw/puns_pos_neg_data.csv`
- `data/raw/kaggle_dataset.csv`

Then run:

```bash
python src/data/preprocess_raw_data.py
```

What it does:

- Normalizes and deduplicates texts 
- Merges sources and enforces contamination-free splits (70/15/15)
- Balances the test set across positive/negative
- Writes processed CSVs to `data/processed/`:
  - `humor_train.csv`, `humor_val.csv`, `humor_test.csv`


### Models

- Baseline CNN: A minimal text-CNN with single conv layer and global max pooling.
- Transformer-based: HuggingFace `AutoModel` (`distilbert-base-uncased`) + linear classifier.
- Proposed SPA model: A prototype “humor-aware” transformer with Setup–Punchline Attention.

Models are constructed via `src/models/model_factory.py` and trained via utilities in `src/training/trainer.py`.


### Experiments (YAML-driven)

Experiments are specified in `experiment_configs_combined.yaml`. Key ideas/fields:

- `base_config`: dataset name (`humor`), tokenizer/model defaults, training hyperparameters
- Experiment groups like `setup_ratio_optimization`, `backbone_freezing`, `regularization_study`, `baseline_comparison`, `quick_test`, `sample_test`, `comprehensive_comparison`
- Each group has `base_params` and a list of `variations` (each with a unique `experiment_id`)

Run via the CLI:

```bash
# Run a specific experiment group (loads its variations from YAML)
python run_combined_experiments.py --experiment baseline_comparison

# Run a fast dev check
python run_combined_experiments.py --experiment quick_test

# Run all configured experiment groups
python run_combined_experiments.py --experiment all

# Use a different YAML config file if needed
python run_combined_experiments.py --config experiment_configs_combined.yaml --experiment setup_ratio_optimization
```

Outputs and logs are stored under `experiments/yaml_driven/<experiment_name>/<timestamp>/`, including an `experiment_summary.json`. A comprehensive summary for `--experiment all` is saved under `experiments/yaml_driven/comprehensive_summary/`.


### Evaluation

`src/evaluation/metrics.py` provides:

- `HumorEvaluationMetrics`: Accuracy, Precision, Recall, F1; confusion matrix counts
- `HumorDataLoader`: loads `text,label` CSVs from `data/processed/`

Training utilities in `src/training/trainer.py` compute metrics on validation/test and log histories.


### Dependencies

Tested with Python 3.10+. Install the following packages:

```bash
pip install -U torch transformers pandas numpy scikit-learn pyyaml tqdm
```


### Reproducibility

- The YAML `base_config` includes `random_seed: 42`.
- Data preprocessing sets Python/NumPy seeds.


