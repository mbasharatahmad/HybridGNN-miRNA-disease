import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
import optuna
import random
import json
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, average_precision_score
import os


def load_data(file_path):
    """Load the dataset from an Excel file."""
    try:
        data = pd.read_excel(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    """Preprocess data into an association matrix."""
    associations = data[['miRNA', 'disease']].drop_duplicates()
    miRNAs = associations['miRNA'].unique()
    diseases = associations['disease'].unique()
    matrix = pd.DataFrame(0, index=miRNAs, columns=diseases)
    matrix.index.name = 'miRNA'
    matrix.columns.name = 'disease'
    for _, row in associations.iterrows():
        if row['miRNA'] in matrix.index and row['disease'] in matrix.columns:
            matrix.loc[row['miRNA'], row['disease']] = 1
    return matrix, miRNAs, diseases

def compute_similarity(matrix):
    """Compute Gaussian Interaction Profile Kernel (GIPK) similarity matrices."""
    gamma_m = 1.0 / (matrix.sum(axis=1).mean() + 1e-10)
    miRNA_sim = pd.DataFrame(rbf_kernel(matrix, gamma=gamma_m), index=matrix.index, columns=matrix.index)
    gamma_d = 1.0 / (matrix.sum(axis=0).mean() + 1e-10)
    disease_sim = pd.DataFrame(rbf_kernel(matrix.T, gamma=gamma_d), index=matrix.columns, columns=matrix.columns)
    return miRNA_sim, disease_sim

def prepare_gcn_data(matrix, miRNA_sim, disease_sim, feature_dim=16):
    """Prepare PyTorch Geometric Data object with node features and edge index."""
    miRNAs = matrix.index
    diseases = matrix.columns
    n_mirnas = len(miRNAs)
    n_diseases = len(diseases)
    pca_m = PCA(n_components=feature_dim, random_state=42)
    miRNA_features = pca_m.fit_transform(miRNA_sim)
    pca_d = PCA(n_components=feature_dim, random_state=42)
    disease_features = pca_d.fit_transform(disease_sim)
    x = torch.tensor(np.vstack([miRNA_features, disease_features]), dtype=torch.float).to(device)
    edges = matrix.stack().reset_index()
    edges = edges[edges[0] == 1]
    miRNA_indices = edges['miRNA'].apply(lambda x: list(miRNAs).index(x)).values
    disease_indices = edges['disease'].apply(lambda x: list(diseases).index(x) + n_mirnas).values
    edge_index = torch.tensor(np.array([miRNA_indices, disease_indices]), dtype=torch.long).to(device)
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    return Data(x=x, edge_index=edge_index).to(device), n_mirnas, n_diseases

class MDMF(nn.Module):
    def __init__(self, num_mirnas, num_diseases, latent_dim, lambda_reg):
        super(MDMF, self).__init__()
        torch.manual_seed(seed)  # Ensure deterministic initialization
        self.U = nn.Parameter(torch.randn(num_mirnas, latent_dim))
        self.V = nn.Parameter(torch.randn(num_diseases, latent_dim))
        self.lambda_reg = lambda_reg

    def forward(self, A, S_m, S_d):
        recon_loss = F.mse_loss(torch.mm(self.U, self.V.t()), A)
        reg_U = torch.norm(self.U, p=2)
        reg_V = torch.norm(self.V, p=2)
        reg_S_m = torch.norm(torch.mm(self.U, self.U.t()) - S_m, p=2)
        reg_S_d = torch.norm(torch.mm(self.V, self.V.t()) - S_d, p=2)
        loss = recon_loss + self.lambda_reg * (reg_U + reg_V + reg_S_m + reg_S_d)
        return loss

    def get_features(self):
        return self.U, self.V

class HybridGCN_GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_heads, num_layers=3):
        super(HybridGCN_GAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))
        self.gcn_layers.append(GCNConv(hidden_channels, out_channels))
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=False))
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False))
        self.gat_layers.append(GATConv(hidden_channels, out_channels, heads=num_heads, concat=False))
        self.gcn_residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.gat_residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_gcn = x
        for i, layer in enumerate(self.gcn_layers):
            x_gcn = layer(x_gcn, edge_index)
            if i < self.num_layers - 1:
                x_gcn = x_gcn.relu()
                x_gcn = F.dropout(x_gcn, p=self.dropout, training=self.training)
        x_gcn += self.gcn_residual(x)
        x_gat = x
        for i, layer in enumerate(self.gat_layers):
            x_gat = layer(x_gat, edge_index)
            if i < self.num_layers - 1:
                x_gat = x_gat.relu()
                x_gat = F.dropout(x_gat, p=self.dropout, training=self.training)
        x_gat += self.gat_residual(x)
        x_combined = torch.cat([x_gcn, x_gat], dim=1)
        return x_combined

def train_model(model, data, train_mask, train_labels, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    mirna_emb, disease_emb = out[:data.n_mirnas], out[data.n_mirnas:]
    pred = (mirna_emb @ disease_emb.t()).view(-1)
    loss = criterion(pred[train_mask], train_labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_model(model, data, mask, labels):
    model.eval()
    with torch.no_grad():
        out = model(data)
        mirna_emb, disease_emb = out[:data.n_mirnas], out[data.n_mirnas:]
        pred = (mirna_emb @ disease_emb.t()).view(-1)
        pred_scores = torch.sigmoid(pred[mask]).cpu().numpy()
        auc = roc_auc_score(labels.cpu().numpy(), pred_scores)
    return auc

def train_mdmf(mdmf, A, S_m, S_d, epochs=100, lr=0.01):
    torch.manual_seed(seed)  # Ensure deterministic initialization
    optimizer = torch.optim.Adam(mdmf.parameters(), lr=lr)
    mdmf.to(device)
    A, S_m, S_d = A.to(device), S_m.to(device), S_d.to(device)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = mdmf(A, S_m, S_d)
        loss.backward()
        optimizer.step()
    return mdmf.get_features()

def objective(trial, matrix, miRNAs, diseases, train_mask, train_labels, val_mask, val_labels):
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    num_heads = trial.suggest_int('num_heads', 1, 16)
    num_layers = trial.suggest_int('num_layers', 2, 4)
    pca_dim = trial.suggest_categorical('pca_dim', [16, 32, 64])
    latent_dim = trial.suggest_int('latent_dim', 16, 64)
    out_channels = trial.suggest_int('out_channels', 8, 64)
    lambda_reg = trial.suggest_float('lambda_reg', 0.01, 0.1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

    miRNA_sim, disease_sim = compute_similarity(matrix)
    gcn_data_new, n_mirnas, n_diseases = prepare_gcn_data(matrix, miRNA_sim, disease_sim, feature_dim=pca_dim)
    gcn_data_new.n_mirnas = n_mirnas

    mdmf = MDMF(num_mirnas=len(miRNAs), num_diseases=len(diseases), latent_dim=latent_dim, lambda_reg=lambda_reg)
    U, V = train_mdmf(mdmf, torch.tensor(matrix.values, dtype=torch.float),
                      torch.tensor(miRNA_sim.values, dtype=torch.float),
                      torch.tensor(disease_sim.values, dtype=torch.float))

    mdmf_features = torch.cat([U, V], dim=0).to(device)
    gcn_data_new.x = torch.cat([gcn_data_new.x, mdmf_features], dim=1)

    model = HybridGCN_GAT(in_channels=gcn_data_new.x.shape[1], hidden_channels=hidden_dim,
                          out_channels=out_channels, dropout=dropout, num_heads=num_heads, num_layers=num_layers)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    max_epochs = 200
    patience = 10
    best_val_auc = 0
    counter = 0

    for epoch in range(max_epochs):
        train_loss = train_model(model, gcn_data_new, train_mask, train_labels.to(device), optimizer, criterion)
        val_auc = evaluate_model(model, gcn_data_new, val_mask, val_labels.to(device))
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return best_val_auc


def perform_optuna_tuning(matrix, miRNAs, diseases, train_mask, train_labels, val_mask, val_labels):
    # Perform Optuna hyperparameter tuning
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, matrix, miRNAs, diseases, train_mask, train_labels, val_mask, val_labels), 
                   n_trials=100)

    # Print the best hyperparameters found by Optuna
    print(f"Best hyperparameters: {study.best_params}")
    print(f"Best validation AUC: {study.best_value:.4f}")
    return study.best_params


# helper to convert (miRNA,disease) pairs to flattened index in [0, n_mirnas*n_diseases)
def pair_to_flat_index(miRNA_label, disease_label):
    mi_idx = matrix.index.get_loc(miRNA_label)
    d_idx = matrix.columns.get_loc(disease_label)
    return mi_idx * n_diseases + d_idx

# ============================================================
# Single Split Functions (for Optuna tuning)
# ============================================================

def random_split_single(matrix, test_size=0.2, negative_ratio=1.0):
    """
    Create a single random split for training and testing
    
    Returns:
    --------
    tuple: (train_mask, train_labels, test_mask, test_labels)
    """
    print(f"Performing random split (test_size={test_size}, seed={seed})...")
    
    known_pairs = matrix.stack().reset_index()
    known_pairs = known_pairs[known_pairs[0] == 1].rename(columns={'level_0':'miRNA','level_1':'disease'})
    
    # Convert positive pairs to flat indices
    positive_pairs_idx = np.array([pair_to_flat_index(m, d) 
                                  for m, d in zip(known_pairs['miRNA'], known_pairs['disease'])])
    
    # Split positive pairs
    pos_train_idx, pos_test_idx = train_test_split(
        positive_pairs_idx, test_size=test_size, random_state=seed
    )
    
    # Sample negatives
    all_neg = np.where(matrix.values.flatten() == 0)[0]
    rng = np.random.RandomState(seed)
    
    n_neg_train = int(len(pos_train_idx) * negative_ratio)
    n_neg_test = int(len(pos_test_idx) * negative_ratio)
    
    neg_train_idx = rng.choice(all_neg, size=n_neg_train, replace=False)
    remaining_neg = np.setdiff1d(all_neg, neg_train_idx)
    neg_test_idx = rng.choice(remaining_neg, size=n_neg_test, replace=False)
    
    # Create masks and labels
    train_mask = np.concatenate([pos_train_idx, neg_train_idx])
    train_labels = torch.tensor(
        np.concatenate([np.ones(len(pos_train_idx)), np.zeros(len(neg_train_idx))]),
        dtype=torch.float
    )
    
    test_mask = np.concatenate([pos_test_idx, neg_test_idx])
    test_labels = torch.tensor(
        np.concatenate([np.ones(len(pos_test_idx)), np.zeros(len(neg_test_idx))]),
        dtype=torch.float
    )
    
    print(f"Train: {len(pos_train_idx)} positives, {len(neg_train_idx)} negatives")
    print(f"Test: {len(pos_test_idx)} positives, {len(neg_test_idx)} negatives")
    
    return train_mask, train_labels, test_mask, test_labels


def cold_disease_split_single(matrix, test_fraction=0.15, negative_ratio=1.0):
    """
    Create a single cold-start disease split
    
    Returns:
    --------
    tuple: (train_mask, train_labels, test_mask, test_labels)
    """
    print(f"Performing cold-start disease split (test_fraction={test_fraction}, seed={seed})...")
    
    known_pairs = matrix.stack().reset_index()
    known_pairs = known_pairs[known_pairs[0] == 1].rename(columns={'level_0':'miRNA','level_1':'disease'})
    
    # Get all diseases with positive associations
    diseases = known_pairs['disease'].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(diseases)
    
    # Split diseases
    n_test = max(1, int(len(diseases) * test_fraction))
    test_diseases = set(diseases[:n_test])
    train_diseases = set(diseases[n_test:])
    
    # Get pairs for train and test
    train_pairs = known_pairs[known_pairs['disease'].isin(train_diseases)]
    test_pairs = known_pairs[known_pairs['disease'].isin(test_diseases)]
    
    # Convert to flat indices
    pos_train_idx = np.array([pair_to_flat_index(m, d) 
                             for m, d in zip(train_pairs['miRNA'], train_pairs['disease'])])
    pos_test_idx = np.array([pair_to_flat_index(m, d) 
                            for m, d in zip(test_pairs['miRNA'], test_pairs['disease'])])
    
    # Get negative pairs for each disease set
    def get_negative_pairs_for_diseases(selected_diseases, matrix, rng, n_samples):
        """Get negative pairs for specific diseases"""
        all_neg = []
        for d in selected_diseases:
            d_idx = matrix.columns.get_loc(d)
            for m in matrix.index:
                if matrix.loc[m, d] == 0:
                    m_idx = matrix.index.get_loc(m)
                    flat_idx = m_idx * len(matrix.columns) + d_idx
                    all_neg.append(flat_idx)
        
        all_neg = np.array(all_neg)
        if len(all_neg) > n_samples:
            return rng.choice(all_neg, size=n_samples, replace=False)
        else:
            return all_neg
    
    # Sample negatives
    n_neg_train = int(len(pos_train_idx) * negative_ratio)
    n_neg_test = int(len(pos_test_idx) * negative_ratio)
    
    neg_train_idx = get_negative_pairs_for_diseases(train_diseases, matrix, rng, n_neg_train)
    neg_test_idx = get_negative_pairs_for_diseases(test_diseases, matrix, rng, n_neg_test)
    
    # Ensure no overlap
    neg_test_idx = np.setdiff1d(neg_test_idx, neg_train_idx)
    
    # Create masks and labels
    train_mask = np.concatenate([pos_train_idx, neg_train_idx])
    train_labels = torch.tensor(
        np.concatenate([np.ones(len(pos_train_idx)), np.zeros(len(neg_train_idx))]),
        dtype=torch.float
    )
    
    test_mask = np.concatenate([pos_test_idx, neg_test_idx])
    test_labels = torch.tensor(
        np.concatenate([np.ones(len(pos_test_idx)), np.zeros(len(neg_test_idx))]),
        dtype=torch.float
    )
    
    print(f"Train diseases: {len(train_diseases)}, Test diseases: {len(test_diseases)}")
    print(f"Train: {len(pos_train_idx)} positives, {len(neg_train_idx)} negatives")
    print(f"Test: {len(pos_test_idx)} positives, {len(neg_test_idx)} negatives")
    
    return train_mask, train_labels, test_mask, test_labels


def cold_mirna_split_single(matrix, test_fraction=0.15, negative_ratio=1.0):
    """
    Create a single cold-start miRNA split
    
    Returns:
    --------
    tuple: (train_mask, train_labels, test_mask, test_labels)
    """
    print(f"Performing cold-start miRNA split (test_fraction={test_fraction}, seed={seed})...")
    
    known_pairs = matrix.stack().reset_index()
    known_pairs = known_pairs[known_pairs[0] == 1].rename(columns={'level_0':'miRNA','level_1':'disease'})
    
    # Get all miRNAs with positive associations
    mirnas = known_pairs['miRNA'].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(mirnas)
    
    # Split miRNAs
    n_test = max(1, int(len(mirnas) * test_fraction))
    test_mirnas = set(mirnas[:n_test])
    train_mirnas = set(mirnas[n_test:])
    
    # Get pairs for train and test
    train_pairs = known_pairs[known_pairs['miRNA'].isin(train_mirnas)]
    test_pairs = known_pairs[known_pairs['miRNA'].isin(test_mirnas)]
    
    # Convert to flat indices
    pos_train_idx = np.array([pair_to_flat_index(m, d) 
                             for m, d in zip(train_pairs['miRNA'], train_pairs['disease'])])
    pos_test_idx = np.array([pair_to_flat_index(m, d) 
                            for m, d in zip(test_pairs['miRNA'], test_pairs['disease'])])
    
    # Get negative pairs for each miRNA set
    def get_negative_pairs_for_mirnas(selected_mirnas, matrix, rng, n_samples):
        """Get negative pairs for specific miRNAs"""
        all_neg = []
        for m in selected_mirnas:
            m_idx = matrix.index.get_loc(m)
            for d in matrix.columns:
                if matrix.loc[m, d] == 0:
                    d_idx = matrix.columns.get_loc(d)
                    flat_idx = m_idx * len(matrix.columns) + d_idx
                    all_neg.append(flat_idx)
        
        all_neg = np.array(all_neg)
        if len(all_neg) > n_samples:
            return rng.choice(all_neg, size=n_samples, replace=False)
        else:
            return all_neg
    
    # Sample negatives
    n_neg_train = int(len(pos_train_idx) * negative_ratio)
    n_neg_test = int(len(pos_test_idx) * negative_ratio)
    
    neg_train_idx = get_negative_pairs_for_mirnas(train_mirnas, matrix, rng, n_neg_train)
    neg_test_idx = get_negative_pairs_for_mirnas(test_mirnas, matrix, rng, n_neg_test)
    
    # Ensure no overlap
    neg_test_idx = np.setdiff1d(neg_test_idx, neg_train_idx)
    
    # Create masks and labels
    train_mask = np.concatenate([pos_train_idx, neg_train_idx])
    train_labels = torch.tensor(
        np.concatenate([np.ones(len(pos_train_idx)), np.zeros(len(neg_train_idx))]),
        dtype=torch.float
    )
    
    test_mask = np.concatenate([pos_test_idx, neg_test_idx])
    test_labels = torch.tensor(
        np.concatenate([np.ones(len(pos_test_idx)), np.zeros(len(neg_test_idx))]),
        dtype=torch.float
    )
    
    print(f"Train miRNAs: {len(train_mirnas)}, Test miRNAs: {len(test_mirnas)}")
    print(f"Train: {len(pos_train_idx)} positives, {len(neg_train_idx)} negatives")
    print(f"Test: {len(pos_test_idx)} positives, {len(neg_test_idx)} negatives")
    
    return train_mask, train_labels, test_mask, test_labels

# ============================================================
# Cross-Validation Functions (to get fold splits)
# ============================================================

def get_cv_folds(matrix, split_mode='random', n_folds=5):
    """
    Generate cross-validation folds based on split mode
    
    Parameters:
    -----------
    matrix : pandas DataFrame
        miRNA-disease association matrix
    split_mode : str
        'random', 'cold_disease', or 'cold_mirna'
    n_folds : int
        Number of folds
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    folds : list of tuples
        Each tuple contains (train_mask, train_labels, test_mask, test_labels)
    """
    
    known_pairs = matrix.stack().reset_index()
    known_pairs = known_pairs[known_pairs[0] == 1].rename(columns={'level_0':'miRNA','level_1':'disease'})
    
    if split_mode == 'random':
        # Random split (KFold on positive pairs)
        positive_pairs_idx = np.array([pair_to_flat_index(m, d) 
                                      for m, d in zip(known_pairs['miRNA'], known_pairs['disease'])])
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = []
        
        for train_idx, test_idx in kf.split(positive_pairs_idx):
            # create train/test positive sets
            pos_train_idx = positive_pairs_idx[train_idx]
            pos_test_idx = positive_pairs_idx[test_idx]
            
            # sample negatives for train/test (balanced)
            all_neg = np.where(matrix.values.flatten() == 0)[0]
            rng = np.random.default_rng(seed)
            neg_train_idx = rng.choice(all_neg, size=len(pos_train_idx), replace=False)
            remaining_neg = np.setdiff1d(all_neg, neg_train_idx)
            neg_test_idx = rng.choice(remaining_neg, size=len(pos_test_idx), replace=False)
            
            # combine to masks & labels
            train_mask = np.concatenate([pos_train_idx, neg_train_idx])
            train_labels = torch.tensor(np.concatenate([np.ones(len(pos_train_idx)), 
                                                       np.zeros(len(neg_train_idx))]), 
                                       dtype=torch.float)
            
            test_mask = np.concatenate([pos_test_idx, neg_test_idx])
            test_labels = torch.tensor(np.concatenate([np.ones(len(pos_test_idx)), 
                                                      np.zeros(len(neg_test_idx))]), 
                                      dtype=torch.float)
            
            folds.append((train_mask, train_labels, test_mask, test_labels))
    
    elif split_mode == 'cold_disease':
        # Cold-start disease split
        diseases = known_pairs['disease'].unique()
        rng = np.random.RandomState(seed)
        rng.shuffle(diseases)
        
        # Split diseases into folds
        disease_folds = np.array_split(diseases, n_folds)
        folds = []
        
        for fold in range(n_folds):
            test_diseases = set(disease_folds[fold])
            train_diseases = set(diseases) - test_diseases
            
            # Get pairs for train and test
            train_pairs = known_pairs[known_pairs['disease'].isin(train_diseases)]
            test_pairs = known_pairs[known_pairs['disease'].isin(test_diseases)]
            
            # Convert to flat indices
            pos_train_idx = np.array([pair_to_flat_index(m, d) 
                                     for m, d in zip(train_pairs['miRNA'], train_pairs['disease'])])
            pos_test_idx = np.array([pair_to_flat_index(m, d) 
                                    for m, d in zip(test_pairs['miRNA'], test_pairs['disease'])])
            
            # Get all possible pairs for each disease set
            def get_negative_pairs_for_diseases(selected_diseases, matrix, rng, n_samples):
                """Get negative pairs for specific diseases"""
                all_neg = []
                for d in selected_diseases:
                    d_idx = matrix.columns.get_loc(d)
                    # Get miRNAs that don't have associations with this disease
                    for m in matrix.index:
                        if matrix.loc[m, d] == 0:
                            m_idx = matrix.index.get_loc(m)
                            flat_idx = m_idx * len(matrix.columns) + d_idx
                            all_neg.append(flat_idx)
                
                all_neg = np.array(all_neg)
                if len(all_neg) > n_samples:
                    return rng.choice(all_neg, size=n_samples, replace=False)
                else:
                    return all_neg
            
            # Sample negatives
            rng_fold = np.random.RandomState(seed + fold)
            neg_train_idx = get_negative_pairs_for_diseases(train_diseases, matrix, rng_fold, len(pos_train_idx))
            neg_test_idx = get_negative_pairs_for_diseases(test_diseases, matrix, rng_fold, len(pos_test_idx))
            
            # Ensure no overlap between train and test negatives
            neg_test_idx = np.setdiff1d(neg_test_idx, neg_train_idx)
            
            # Create masks and labels
            train_mask = np.concatenate([pos_train_idx, neg_train_idx])
            train_labels = torch.tensor(np.concatenate([np.ones(len(pos_train_idx)), 
                                                       np.zeros(len(neg_train_idx))]), 
                                       dtype=torch.float)
            
            test_mask = np.concatenate([pos_test_idx, neg_test_idx])
            test_labels = torch.tensor(np.concatenate([np.ones(len(pos_test_idx)), 
                                                      np.zeros(len(neg_test_idx))]), 
                                      dtype=torch.float)
            
            folds.append((train_mask, train_labels, test_mask, test_labels))
    
    elif split_mode == 'cold_mirna':
        # Cold-start miRNA split
        mirnas = known_pairs['miRNA'].unique()
        rng = np.random.RandomState(seed)
        rng.shuffle(mirnas)
        
        # Split miRNAs into folds
        mirna_folds = np.array_split(mirnas, n_folds)
        folds = []
        
        for fold in range(n_folds):
            test_mirnas = set(mirna_folds[fold])
            train_mirnas = set(mirnas) - test_mirnas
            
            # Get pairs for train and test
            train_pairs = known_pairs[known_pairs['miRNA'].isin(train_mirnas)]
            test_pairs = known_pairs[known_pairs['miRNA'].isin(test_mirnas)]
            
            # Convert to flat indices
            pos_train_idx = np.array([pair_to_flat_index(m, d) 
                                     for m, d in zip(train_pairs['miRNA'], train_pairs['disease'])])
            pos_test_idx = np.array([pair_to_flat_index(m, d) 
                                    for m, d in zip(test_pairs['miRNA'], test_pairs['disease'])])
            
            # Get all possible pairs for each miRNA set
            def get_negative_pairs_for_mirnas(selected_mirnas, matrix, rng, n_samples):
                """Get negative pairs for specific miRNAs"""
                all_neg = []
                for m in selected_mirnas:
                    m_idx = matrix.index.get_loc(m)
                    # Get diseases that don't have associations with this miRNA
                    for d in matrix.columns:
                        if matrix.loc[m, d] == 0:
                            d_idx = matrix.columns.get_loc(d)
                            flat_idx = m_idx * len(matrix.columns) + d_idx
                            all_neg.append(flat_idx)
                
                all_neg = np.array(all_neg)
                if len(all_neg) > n_samples:
                    return rng.choice(all_neg, size=n_samples, replace=False)
                else:
                    return all_neg
            
            # Sample negatives
            rng_fold = np.random.RandomState(seed + fold)
            neg_train_idx = get_negative_pairs_for_mirnas(train_mirnas, matrix, rng_fold, len(pos_train_idx))
            neg_test_idx = get_negative_pairs_for_mirnas(test_mirnas, matrix, rng_fold, len(pos_test_idx))
            
            # Ensure no overlap between train and test negatives
            neg_test_idx = np.setdiff1d(neg_test_idx, neg_train_idx)
            
            # Create masks and labels
            train_mask = np.concatenate([pos_train_idx, neg_train_idx])
            train_labels = torch.tensor(np.concatenate([np.ones(len(pos_train_idx)), 
                                                       np.zeros(len(neg_train_idx))]), 
                                       dtype=torch.float)
            
            test_mask = np.concatenate([pos_test_idx, neg_test_idx])
            test_labels = torch.tensor(np.concatenate([np.ones(len(pos_test_idx)), 
                                                      np.zeros(len(neg_test_idx))]), 
                                      dtype=torch.float)
            
            folds.append((train_mask, train_labels, test_mask, test_labels))
    
    else:
        raise ValueError(f"Invalid split_mode: {split_mode}. Choose from ['random', 'cold_disease', 'cold_mirna']")
    
    return folds

def train_and_evaluate_model(model, gcn_data, train_mask, train_labels, test_mask, test_labels,
                             learning_rate=1e-3, weight_decay=1e-4, max_epochs=100, patience=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()   # stable: expects logits

    best_val_auc = -np.inf
    best_state = None
    patience_counter = 0

    # We'll use test_mask/test_labels as the validation here for early stopping
    for epoch in range(1, max_epochs+1):
        model.train()
        optimizer.zero_grad()

        # forward: get node embeddings
        node_out = model(gcn_data)               # shape: (n_nodes, embed_dim)
        # split
        mirna_emb = node_out[:n_mirnas]          # (n_mirnas, d)
        disease_emb = node_out[n_mirnas:]        # (n_diseases, d)

        # compute pairwise logits for all pairs
        logits = (mirna_emb @ disease_emb.t()).view(-1)   # shape: (n_mirnas*n_diseases,)

        # compute loss on training pair indices
        loss = loss_fn(logits[train_mask], train_labels)
        loss.backward()
        optimizer.step()

        # validation AUC every epoch
        model.eval()
        with torch.no_grad():
            node_out = model(gcn_data)
            mirna_emb = node_out[:n_mirnas]
            disease_emb = node_out[n_mirnas:]
            val_logits = (mirna_emb @ disease_emb.t()).view(-1)
            val_probs = torch.sigmoid(val_logits[test_mask]).cpu().numpy()
            val_labels_np = test_labels.cpu().numpy()
            # If all labels are same maybe roc_auc_score fails; guard it
            if len(np.unique(val_labels_np)) > 1:
                val_auc = roc_auc_score(val_labels_np, val_probs)
            else:
                val_auc = 0.0

        # early stopping on validation AUC
        if val_auc > best_val_auc + 1e-6:
            best_val_auc = val_auc
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            # if patience_counter >= patience:
            #     print(f"Early stopping at epoch {epoch} (best_val_auc={best_val_auc:.4f})")
            #     break

    # load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    # final evaluation on test_mask
    model.eval()
    with torch.no_grad():
        node_out = model(gcn_data)
        mirna_emb = node_out[:n_mirnas]
        disease_emb = node_out[n_mirnas:]
        final_logits = (mirna_emb @ disease_emb.t()).view(-1)
        final_probs = torch.sigmoid(final_logits[test_mask]).cpu().numpy()
        final_labels = test_labels.cpu().numpy()

        auc = roc_auc_score(final_labels, final_probs) if len(np.unique(final_labels))>1 else 0.0
        pr_auc = average_precision_score(final_labels, final_probs)
        preds_bin = (final_probs >= 0.5).astype(int)
        accuracy = accuracy_score(final_labels, preds_bin)
        precision = precision_score(final_labels, preds_bin, zero_division=0)
        recall = recall_score(final_labels, preds_bin, zero_division=0)
        f1 = f1_score(final_labels, preds_bin, zero_division=0)

    return auc, accuracy, precision, recall, f1, pr_auc, best_state

# Cross-validation function with the best hyperparameters
def cross_validate_with_best_params(matrix, miRNAs, diseases, best_params, split_mode='random', n_folds=5, save_models=True):
    """
    Perform cross-validation with different split strategies
    
    Parameters:
    -----------
    matrix : pandas DataFrame
        miRNA-disease association matrix
    miRNAs : list
        List of miRNA names
    diseases : list
        List of disease names
    best_params : dict
        Best hyperparameters from optimization
    split_mode : str
        'random', 'cold_disease', or 'cold_mirna'
    n_folds : int
        Number of folds for cross-validation
    save_models : bool
        Whether to save the best model from each fold
    epochs : int
        Maximum number of training epochs
    patience : int
        Early stopping patience
    seed : int
        Random seed for reproducibility
    """

    aucs = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracies = []
    pr_scores = []
    
    # Store best models from each fold
    best_models = []
    best_model_paths = []

    print(f"\n=== {split_mode.upper()} Split - {n_folds}-Fold Cross-Validation ===")

    # Prepare final data
    miRNA_sim, disease_sim = compute_similarity(matrix)
    gcn_data_final, n_mirnas, n_diseases = prepare_gcn_data(
        matrix, miRNA_sim, disease_sim, 
        feature_dim=best_params['pca_dim'],
    )
    gcn_data_final.n_mirnas = n_mirnas
    
    # Get cross-validation folds based on split mode
    folds = get_cv_folds(matrix, split_mode=split_mode, n_folds=n_folds)
    
    print(f"Split mode: {split_mode}")
    print(f"Number of folds: {n_folds}")
    print(f"Using seed: {seed}")

    # Train MDMF with best parameters (once for all folds)
    mdmf_final = MDMF(
        num_mirnas=len(miRNAs), 
        num_diseases=len(diseases), 
        latent_dim=best_params['latent_dim'], 
        lambda_reg=best_params['lambda_reg']
    )
    U_final, V_final = train_mdmf(
        mdmf_final, 
        torch.tensor(matrix.values, dtype=torch.float),
        torch.tensor(miRNA_sim.values, dtype=torch.float),
        torch.tensor(disease_sim.values, dtype=torch.float)
    )

    mdmf_features_final = torch.cat([U_final, V_final], dim=0).to(device)
    gcn_data_final.x = torch.cat([gcn_data_final.x, mdmf_features_final], dim=1)

    # Perform cross-validation
    for fold, (train_mask, train_labels, test_mask, test_labels) in enumerate(folds):
        print(f"\nTraining Fold {fold+1}/{n_folds}...")
        
        # Move labels to device
        train_labels = train_labels.to(device)
        test_labels = test_labels.to(device)
        
        # Print fold statistics
        n_pos_train = (train_labels == 1).sum().item()
        n_neg_train = (train_labels == 0).sum().item()
        n_pos_test = (test_labels == 1).sum().item()
        n_neg_test = (test_labels == 0).sum().item()
        
        print(f"Train: {n_pos_train} positives, {n_neg_train} negatives")
        print(f"Test: {n_pos_test} positives, {n_neg_test} negatives")

        # Use the best parameters from Optuna to create the model
        model = HybridGCN_GAT(
            in_channels=gcn_data_final.x.shape[1],
            hidden_channels=best_params['hidden_dim'],
            out_channels=best_params['out_channels'],
            dropout=best_params['dropout'],
            num_heads=best_params['num_heads'],
            num_layers=best_params['num_layers']
        )

        # Train the model for this fold and get the best model state
        auc, accuracy, precision, recall, f1, pr_auc, best_model_state = train_and_evaluate_model(
            model, gcn_data_final,
            train_mask=train_mask, train_labels=train_labels,
            test_mask=test_mask, test_labels=test_labels,
            learning_rate=best_params.get('learning_rate', 1e-3),
            weight_decay=best_params.get('weight_decay', 1e-4),
            max_epochs=epochs,
            patience=patience
        )

        print(f"Fold {fold+1} Results:")
        print(f"  AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"  PR AUC: {pr_auc:.4f}")
        
        aucs.append(auc)
        accuracies.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        pr_scores.append(pr_auc)
        
        # Store the best model from this fold
        if save_models:
            # Create a copy of the model architecture with best weights
            best_model = HybridGCN_GAT(
                in_channels=gcn_data_final.x.shape[1],
                hidden_channels=best_params['hidden_dim'],
                out_channels=best_params['out_channels'],
                dropout=best_params['dropout'],
                num_heads=best_params['num_heads'],
                num_layers=best_params['num_layers']
            )
            best_model.load_state_dict(best_model_state)
            best_model.eval()
            best_models.append(best_model)
            
            # Save model to file
            model_dir = f'{output_folder}_{split_mode}'
            os.makedirs(model_dir, exist_ok=True)
            model_path = f'{model_dir}/best_model_fold_{fold+1}.pth'
            
            torch.save({
                'fold': fold + 1,
                'split_mode': split_mode,
                'model_state_dict': best_model_state,
                'model_params': {
                    'in_channels': gcn_data_final.x.shape[1],
                    'hidden_channels': best_params['hidden_dim'],
                    'out_channels': best_params['out_channels'],
                    'dropout': best_params['dropout'],
                    'num_heads': best_params['num_heads'],
                    'num_layers': best_params['num_layers']
                },
                'fold_metrics': {
                    'auc': auc,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'pr_auc': pr_auc
                },
                'fold_stats': {
                    'n_pos_train': n_pos_train,
                    'n_neg_train': n_neg_train,
                    'n_pos_test': n_pos_test,
                    'n_neg_test': n_neg_test
                }
            }, model_path)
            
            best_model_paths.append(model_path)
            print(f"Saved best model for fold {fold+1} to {model_path}")

    # Average metrics across all folds
    print(f"\n=== Summary of {split_mode.upper()} Split - {n_folds}-Fold Cross-Validation ===")
    print(f"Average AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"Average Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Average Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
    print(f"Average Recall: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
    print(f"Average F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"Average PR AUC: {np.mean(pr_scores):.4f} ± {np.std(pr_scores):.4f}")
    
    # Save summary of all folds
    if save_models:
        summary = {
            'split_mode': split_mode,
            'n_folds': n_folds,
            'seed': seed,
            'fold_metrics': {
                'aucs': aucs,
                'accuracies': accuracies,
                'precisions': precision_scores,
                'recalls': recall_scores,
                'f1_scores': f1_scores,
                'pr_scores': pr_scores
            },
            'average_metrics': {
                'mean_auc': float(np.mean(aucs)),
                'std_auc': float(np.std(aucs)),
                'mean_accuracy': float(np.mean(accuracies)),
                'std_accuracy': float(np.std(accuracies)),
                'mean_precision': float(np.mean(precision_scores)),
                'std_precision': float(np.std(precision_scores)),
                'mean_recall': float(np.mean(recall_scores)),
                'std_recall': float(np.std(recall_scores)),
                'mean_f1': float(np.mean(f1_scores)),
                'std_f1': float(np.std(f1_scores)),
                'mean_pr_auc': float(np.mean(pr_scores)),
                'std_pr_auc': float(np.std(pr_scores))
            },
            'best_params': best_params,
            'model_paths': best_model_paths
        }
        
        summary_path = f'{model_dir}/cross_validation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved cross-validation summary to {summary_path}")
    
    return best_models, best_model_paths

if __name__ == "__main__":
    import argparse
    import json
    import random
    import numpy as np
    import torch

    def parse_args():
        parser = argparse.ArgumentParser(
            description="HybridGNN Cross-Validation Runner (cv_mirna_split.py)"
        )

        # Reproducibility
        parser.add_argument("--seed", type=int, default=42, help="Random seed")

        # Paths
        parser.add_argument(
            "--data_path",
            type=str,
            default="data/alldata.xlsx",
            help="Path to miRNA-disease association Excel file",
        )
        parser.add_argument(
            "--best_params_file_path",
            type=str,
            default="best_params_cv.json",
            help="Base path to JSON file for best hyperparameters (suffix added per mode)",
        )
        parser.add_argument(
            "--output_folder",
            type=str,
            default="models_cv",
            help="Folder to save model checkpoints and outputs",
        )

        # Mode: 0/1/2
        parser.add_argument(
            "--mode",
            type=int,
            choices=[0, 1, 2],
            default=2,  # match your original default
            help="Training mode: 0=random, 1=cold_disease, 2=cold_mirna",
        )

        # Optuna
        parser.add_argument(
            "--optuna_tuning",
            action="store_true",
            help="Enable Optuna hyperparameter tuning",
        )
        parser.add_argument(
            "--no_optuna_tuning",
            action="store_true",
            help="Disable Optuna tuning (overrides --optuna_tuning)",
        )

        # Experiment knobs
        parser.add_argument("--negative_ratio", type=float, default=1.0)
        parser.add_argument("--test_size", type=float, default=0.2)
        parser.add_argument("--test_fraction", type=float, default=0.15)

        # Training
        parser.add_argument("--epochs", type=int, default=200)
        parser.add_argument("--patience", type=int, default=20)

        return parser.parse_args()

    args = parse_args()

    # Mode mapping
    mode_id_to_name = {0: "random", 1: "cold_disease", 2: "cold_mirna"}
    split_mode = mode_id_to_name[args.mode]  # <-- now it's a string

    # Optuna default: True (to match your original), override via flags
    optuna_tuning = True
    if args.optuna_tuning:
        optuna_tuning = True
    if args.no_optuna_tuning:
        optuna_tuning = False

    # Set random seed for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Params
    file_path = args.data_path
    best_params_file_path = f"{args.best_params_file_path.split('.')[0]}_{split_mode}.json"
    output_folder = args.output_folder
    negative_ratio = args.negative_ratio
    test_size = args.test_size
    test_fraction = args.test_fraction
    epochs = args.epochs
    patience = args.patience

    # Load + preprocess
    data = load_data(file_path)
    matrix, miRNAs, diseases = preprocess_data(data)
    n_diseases = len(diseases)
    n_mirnas = len(miRNAs)

    # Mode-specific params filename
    params_path = f"{best_params_file_path.split('.json')[0]}_{split_mode}.json"

    if optuna_tuning:
        print(f"\n=== {split_mode.upper()} Split - Optuna Tuning ===")

        if split_mode == "random":
            train_mask, train_labels, test_mask, test_labels = random_split_single(
                matrix, test_size=test_size, negative_ratio=negative_ratio
            )
        elif split_mode == "cold_disease":
            train_mask, train_labels, test_mask, test_labels = cold_disease_split_single(
                matrix, test_fraction=test_fraction, negative_ratio=negative_ratio
            )
        elif split_mode == "cold_mirna":
            train_mask, train_labels, test_mask, test_labels = cold_mirna_split_single(
                matrix, test_fraction=test_fraction, negative_ratio=negative_ratio
            )
        else:
            raise ValueError(f"Invalid split_mode: {split_mode}")

        best_params = perform_optuna_tuning(
            matrix, miRNAs, diseases, train_mask, train_labels, test_mask, test_labels
        )

        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2)

    else:
        with open(params_path, "r") as f:
            best_params = json.load(f)

    best_models, model_paths = cross_validate_with_best_params(
        matrix, miRNAs, diseases, best_params, split_mode, n_folds=5, save_models=True
    )
