# HybridGNN: miRNA–Disease Association Prediction

## 📋 Overview
**HybridGNN** is a novel graph neural network architecture that integrates **Graph Convolutional Networks (GCN)**, **Graph Attention Networks (GAT)**, and **Matrix Decomposition with Matrix Factorization (MDMF)** for predicting miRNA–disease associations.  
This repository contains the complete implementation and extensive experimental evaluation across multiple data split strategies.

---

## 🎯 Key Features
- **Hybrid Architecture**: Combines GCN, GAT, and MDMF for comprehensive feature learning  
- **Multiple Split Strategies**: Random, cold-disease, and cold-miRNA evaluation scenarios  
- **Stratified Analysis**: Performance breakdown by disease categories and miRNA families  
- **Comprehensive Evaluation**: 5-fold cross-validation with rigorous hyperparameter optimization  

---

## 📊 Performance Results

### Overall Performance Across Split Strategies

| Metric       | Random Split        | Cold-Disease         | Cold-miRNA          |
|--------------|---------------------|----------------------|---------------------|
| AUC-ROC      | 0.9765 ± 0.0006     | 0.9365 ± 0.0054      | 0.9678 ± 0.0009     |
| Accuracy     | 0.9186 ± 0.0028     | 0.5756 ± 0.0279      | 0.9034 ± 0.0024     |
| Precision    | 0.9013 ± 0.0057     | 0.5424 ± 0.0170      | 0.8966 ± 0.0097     |
| Recall       | 0.9403 ± 0.0056     | 0.9809 ± 0.0079      | 0.9123 ± 0.0106     |
| F1-Score     | 0.9204 ± 0.0026     | 0.6983 ± 0.0123      | 0.9042 ± 0.0022     |

---

### Disease Category Performance (AUC)

| Category             | Random | Cold-Disease | Cold-miRNA |
|----------------------|--------|--------------|------------|
| Cancer               | 0.9780 | 0.9661       | 0.9770     |
| Fibrotic             | 0.9643 | 0.9725       | 0.9798     |
| Neurological         | 0.9791 | 0.9140       | 0.9778     |
| Cardiovascular       | 0.9762 | 0.9207       | 0.9730     |
| Inflammatory         | 0.9662 | 0.9246       | 0.9693     |
| Genetic / Syndrome   | 0.9747 | 0.8904       | 0.9576     |
| Other                | 0.9745 | 0.9117       | 0.9633     |

---

### Top Performing miRNA Families (AUC ≥ 0.95)

| Family   | Random | Cold-Disease | Cold-miRNA |
|----------|--------|--------------|------------|
| miR-21   | 1.000  | 0.900        | –          |
| miR-145  | 1.000  | 0.975        | –          |
| miR-125  | 0.986  | 0.964        | 0.929      |
| miR-520  | 0.968  | 0.990        | 0.945      |
| miR-378  | 0.947  | 0.965        | –          |
| miR-199  | –      | 0.972        | –          |

---

### Challenging miRNA Families (AUC < 0.88)

| Family   | Random | Cold-Disease | Cold-miRNA |
|----------|--------|--------------|------------|
| let-7    | 0.889  | 0.891        | 0.876      |
| miR-29  | 0.872  | 0.886        | 0.822      |
| miR-126 | 0.988  | –            | 0.837      |
| miR-181 | 0.906  | 0.917        | 0.788      |

---

## 🗂️ Repository Structure

```text
├── data/                    # Dataset files (Upload data to this folder)
├── cv_mirna_split.py              # Main cross-validation experiments
├── miRNA_ablation_stratified.py   # Stratified analysis (released after paper acceptance)
├── miRNA_ablation_study_final.py  # Ablation study (released after paper acceptance)
├── models_cv/              # Saved model checkpoints
│   ├── models_cv_random/
│   ├── models_cv_cold_disease/
│   └── models_cv_cold_mirna/
├── requirements.txt        # Python dependencies
└── README.md               # This file
```
## Environment Setup

### Prerequisites
- Python 3.10
- CUDA-capable GPU (recommended) or CPU

### Installation
Using requirements.txt
```bash
# Create and activate virtual environment
python3.10 -m venv hybridgnn_env
source hybridgnn_env/bin/activate  
```

## Experiment: Cross-Validation (Choose a Split Mode)

### Mode Mapping
- `--mode 0` = random split
- `--mode 1` = cold-disease split  
- `--mode 2` = cold-miRNA split

## 1. Run with Optuna Tuning ON
This performs hyperparameter optimization and saves best parameters.

### Random Split
```bash
python cv_mirna_split.py \
  --mode 0 \
  --data_path data/alldata.xlsx \
  --best_params_file_path best_params_cv.json \
  --optuna_tuning True
```

### Cold-Disease Split
```bash
python cv_mirna_split.py \
  --mode 1 \
  --data_path data/alldata.xlsx \
  --best_params_file_path best_params_cv.json \
  --optuna_tuning True
```

### Cold-miRNA Split
```bash
python cv_mirna_split.py \
  --mode 2 \
  --data_path data/alldata.xlsx \
  --best_params_file_path best_params_cv.json \
  --optuna_tuning True
```

Output: Creates best_params_cv_random.json, best_params_cv_cold_disease.json, or best_params_cv_cold_mirna.json

## 2. Run with Optuna Tuning OFF
Use after tuning has generated the JSON file to train/evaluate with saved parameters.

###  Random Split (loads best_params_cv_random.json)
```bash
python cv_mirna_split.py \
  --mode 0 \
  --data_path data/alldata.xlsx \
  --best_params_file_path best_params_cv.json \
  --optuna_tuning False
```

### Cold-Disease Split (loads best_params_cv_cold_disease.json)
```bash
python cv_mirna_split.py \
  --mode 1 \
  --data_path data/alldata.xlsx \
  --best_params_file_path best_params_cv.json \
  --optuna_tuning False
```

### Cold-miRNA Split (loads best_params_cv_cold_mirna.json)
```bash
python cv_mirna_split.py \
  --mode 2 \
  --data_path data/alldata.xlsx \
  --best_params_file_path best_params_cv.json \
  --optuna_tuning False
```

### Expected Outputs
```text
models_cv_random/              # For mode 0
├── best_model_fold_1.pth
├── best_model_fold_2.pth
├── best_model_fold_3.pth
├── best_model_fold_4.pth
├── best_model_fold_5.pth
└── cross_validation_summary.json

models_cv_cold_disease/        # For mode 1
├── best_model_fold_1.pth
├── ...
└── cross_validation_summary.json

models_cv_cold_mirna/          # For mode 2
├── best_model_fold_1.pth
├── ...
└── cross_validation_summary.json
```
