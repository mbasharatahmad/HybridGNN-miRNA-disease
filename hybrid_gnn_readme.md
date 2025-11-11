# HybridGNN: A Graph Neural Network Approach for Human miRNA–Disease Association Prediction

## Overview
HybridGNN is a novel deep learning model designed to predict potential associations between human microRNAs (miRNAs) and diseases. miRNAs are small non-coding RNAs (18-24 nucleotides) that play pivotal roles in gene regulation. Dysregulation of miRNAs has been linked to the onset and progression of complex human diseases. HybridGNN integrates Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), and Matrix Decomposition with Matrix Factorization (MDMF) to efficiently predict miRNA-disease associations by leveraging multi-source similarity information.

## Key Features
- **Integration of Multi-source Similarity Networks:** Uses five types of similarity networks — three from miRNAs and two from diseases — to capture complementary biological interactions.
- **Hybrid Graph Neural Network Modules:** Combines GCN, GAT, and MDMF to generate robust node embeddings in heterogeneous networks.
- **Oversmoothing Mitigation:** Complementary interactions among modules help reduce the oversmoothing problem common in deep graph models.
- **Mini-batch Gradient Descent:** Partitions the network into smaller subgraphs to improve computational efficiency, speed, and scalability.
- **High Predictive Performance:** Achieved an AUC-ROC of 0.9715 with a dot product classifier, outperforming existing methods.

## Installation
To get started with HybridGNN, clone the repository:

```bash
git clone https://github.com/YourUsername/HybridGNN-miRNA-disease.git
cd HybridGNN-miRNA-disease
```

Ensure you have Python 3.8+ installed and install the required packages:

```bash
pip install -r requirements.txt
```

## Data
The dataset used for this project is collected from the **HMDD database** and is provided in CSV format. It contains miRNA-disease association information and similarity matrices for feature extraction.

**Dataset Files**
- `miRNA_similarity.csv`: miRNA similarity information.
- `disease_similarity.csv`: Disease similarity information.
- `miRNA_disease_association.csv`: Known associations for training and testing.

## How to Run

### 1. Prepare the Data
Place your CSV files in the `data/` directory. Ensure that all similarity matrices and association files are correctly formatted.

### 2. Run the Notebook
Open the `HybridGNN.ipynb` notebook in Jupyter Notebook or JupyterLab to explore the workflow.

### 3. Model Training and Evaluation
The notebook guides you through:
- Constructing the miRNA-disease heterogeneous network.
- Extracting features using three modules (GCN, GAT, MDMF).
- Predicting miRNA-disease associations using a fully connected network.
- Evaluating model performance using metrics like Accuracy, Precision, Recall, and F1 Score.

## Model Architecture
1. **Dataset Preparation:** Construct a heterogeneous network integrating miRNA and disease similarity information.
2. **Feature Extraction:** Generate node embeddings using three modules — GCN, GAT, and MDMF.
3. **Prediction:** Use a fully connected network on the embeddings to predict miRNA-disease associations.

## Evaluation Metrics
- **Accuracy:** Proportion of correctly predicted associations.
- **Precision:** Fraction of true positive predictions among all predicted positives.
- **Recall:** Fraction of actual positives correctly predicted.
- **F1 Score:** Harmonic mean of precision and recall.

## References
- For details, see the paper: *HybridGNN: A Graph Neural Network Approach for Human miRNA–Disease Association Prediction*.

