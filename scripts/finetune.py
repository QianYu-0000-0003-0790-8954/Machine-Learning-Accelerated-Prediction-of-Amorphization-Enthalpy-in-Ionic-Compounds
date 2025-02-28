#!/usr/bin/env python
"""
finetune_5fold_compare.py

This script performs 5-fold cross-validation comparing a fine-tuned GNN model against a model trained from scratch.
It loads data from the dataset folder, uses utility functions for data processing and model training,
and saves model checkpoints and prediction results.
"""
import sys
import os
# Insert the repository root (one directory above the 'scripts' folder) into sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch_geometric as tg
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from ase import Atom
from ase.neighborlist import neighbor_list
from sklearn import metrics
from sklearn.model_selection import KFold
import itertools

# Import custom modules from utils
from utils.utils_data_finetune import deload_data
from utils.utils_model_finetune import PeriodicNetwork, train

# ---------------------------------------------
# 1. Setup
# ---------------------------------------------
torch.set_default_dtype(torch.float64)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('Torch device:', device)

property_names = ['E per atom above ground state']

# ---------------------------------------------
# 2. Load Data
# ---------------------------------------------
# Use relative path for the input CSV file
data_csv = "dataset/finetune_input.csv"
df, species, mean, std = deload_data(data_csv, property_names)

run_name = 'finetune_vs_scratch_' + time.strftime("%y%m%d", time.localtime())
print('Run name:', run_name)

# Encode atom types (mapping element symbol to index)
type_encoding = {Atom(Z).symbol: Z - 1 for Z in range(1, 119)}
type_onehot = torch.eye(len(type_encoding), dtype=torch.float64)

def build_data(entry, type_encoding, type_onehot, r_max=8.0, mean=None, std=None):
    """
    Build a torch_geometric Data object from a structure entry.
    """
    symbols = list(entry.structure.symbols)
    positions = torch.from_numpy(entry.structure.positions).double()
    lattice = torch.from_numpy(entry.structure.cell.array).double().unsqueeze(0)
    feature = torch.tensor(entry.feature, dtype=torch.float64)

    # Get neighbor list within cutoff radius
    edge_src, edge_dst, edge_shift = neighbor_list("ijS", a=entry.structure, cutoff=r_max, self_interaction=True)
    edge_shift = torch.tensor(edge_shift, dtype=torch.float64)
    lattice_expanded = lattice.repeat(edge_shift.size(0), 1, 1)
    edge_shift_vectors = torch.einsum('ni,nij->nj', edge_shift, lattice_expanded)
    edge_vec = positions[edge_dst] - positions[edge_src] + edge_shift_vectors

    targets = torch.tensor(entry.prop, dtype=torch.float64)
    # Normalize targets
    targets = (targets - torch.tensor(mean, dtype=torch.float64)) / torch.tensor(std, dtype=torch.float64)

    data = tg.data.Data(
        pos=positions,
        lattice=lattice,
        x=type_onehot[[type_encoding[s] for s in symbols]],
        edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
        edge_vec=edge_vec,
        y=targets,
        feature=feature.unsqueeze(0)  # shape: (1, feature_dim)
    )
    return data

print("Building data (one-time only)...")
df['data'] = df.apply(lambda x: build_data(x, type_encoding, type_onehot, r_max=8.0, mean=mean, std=std), axis=1)

# ---------------------------------------------
# 3. Evaluate and Save Predictions (Helper Function)
# ---------------------------------------------
def evaluate_and_save(model, dataloader, device, mean, std, property_names, dataset_name,
                      df, idx, run_name, fold_number, model_tag):
    """
    Evaluate the model and save predictions as a CSV file.
    """
    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        for data in tqdm(dataloader, desc=f"Evaluating {dataset_name}, Fold {fold_number}, {model_tag}"):
            data = data.to(device)
            output = model(data)
            data_y = data.y.view(data.num_graphs, -1)
            trues.append(data_y.cpu().numpy())
            preds.append(output.cpu().numpy())

    # Reverse normalization
    trues = np.vstack(trues) * std + mean
    preds = np.vstack(preds) * std + mean

    metrics_dict = {}
    print(f"\nMetrics for {dataset_name} Set, Fold {fold_number} ({model_tag}):")
    for i, prop_name in enumerate(property_names):
        mae = metrics.mean_absolute_error(trues[:, i], preds[:, i])
        r2 = metrics.r2_score(trues[:, i], preds[:, i])
        metrics_dict[prop_name] = {'MAE': mae, 'R2': r2}
        print(f"  Property: {prop_name}")
        print(f"    MAE = {mae:.4f}, R2 = {r2:.4f}")

    # Save predictions to CSV
    save_dict = {'id': df.iloc[idx]['ids'].values if 'ids' in df.columns else df.iloc[idx].index.values}
    for i, prop_name in enumerate(property_names):
        save_dict[f'true_{prop_name}'] = trues[:, i]
        save_dict[f'pred_{prop_name}'] = preds[:, i]
    out_filename = f"{run_name}_fold{fold_number}_{dataset_name.lower()}_{model_tag}_predictions.csv"
    pd.DataFrame(save_dict).to_csv(out_filename, index=False)
    print(f"{dataset_name} predictions saved as {out_filename}\n")
    return metrics_dict

# ---------------------------------------------
# 4. 5-Fold Cross Validation Training and Evaluation
# ---------------------------------------------
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
batch_size = 32
all_metrics = {'finetune': {'train': [], 'valid': []}, 'scratch': {'train': [], 'valid': []}}

# Use a relative path for the pretrained checkpoint
checkpoint_path = "checkpoints/pretrain/model_241212_best.pth"

for fold_number, (train_idx, valid_idx) in enumerate(kfold.split(df), start=1):
    print(f"\n{'='*40}\nFOLD {fold_number}: Train size = {len(train_idx)}, Valid size = {len(valid_idx)}\n{'='*40}\n")

    # Build datasets and dataloaders
    train_data = df.iloc[train_idx]['data'].tolist()
    valid_data = df.iloc[valid_idx]['data'].tolist()
    dataloader_train = tg.loader.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    dataloader_valid = tg.loader.DataLoader(valid_data, batch_size=batch_size)

    # Model hyperparameters
    in_dim = 118  # Number of atom types
    em_dim = 64
    feature_dim = len(df.iloc[0]['feature'])
    num_properties = len(property_names)
    out_dim = 64

    # ---------------------------
    # A) Finetuned Model
    # ---------------------------
    model_finetune = PeriodicNetwork(
        in_dim=in_dim,
        em_dim=em_dim,
        device=device,
        feature_dim=feature_dim,
        num_properties=num_properties,
        irreps_in=f"{em_dim}x0e",
        irreps_out=f"{out_dim}x0e",
        irreps_node_attr=f"{em_dim}x0e",
        layers=2,
        mul=32,
        lmax=2,
        max_radius=8.0,
        num_neighbors=20,
        reduce_output=True,
        out_dim=out_dim
    ).to(device)

    # Load pretrained weights (partial load allowed)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_finetune.load_state_dict(checkpoint['model_state'], strict=False)

    # Setup optimizer, scheduler, and loss functions
    opt_ft = torch.optim.AdamW(model_finetune.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler_ft = torch.optim.lr_scheduler.ExponentialLR(opt_ft, gamma=0.95)
    loss_fn = torch.nn.MSELoss()
    loss_fn_mae = torch.nn.L1Loss()

    run_name_fold_finetune = f"{run_name}_fold{fold_number}_finetune"
    print(f"--- Training FINETUNE model for fold {fold_number} ---")
    history_ft = train(model=model_finetune,
                       optimizer=opt_ft,
                       dataloader_train=dataloader_train,
                       dataloader_valid=dataloader_valid,
                       loss_fn=loss_fn,
                       loss_fn_mae=loss_fn_mae,
                       run_name=run_name_fold_finetune,
                       max_iter=200,
                       scheduler=scheduler_ft,
                       device=device,
                       patience=40)
    torch.save({
        'model_state': model_finetune.state_dict(),
        'optimizer_state': opt_ft.state_dict(),
        'scheduler_state': scheduler_ft.state_dict(),
        'history': history_ft
    }, f"{run_name_fold_finetune}_model.pth")
    pd.DataFrame(history_ft).to_csv(f"{run_name_fold_finetune}_history.csv", index=False)

    train_metrics_ft = evaluate_and_save(model_finetune, dataloader_train, device, mean, std,
                                         property_names, 'Train', df, train_idx, run_name, fold_number, 'finetune')
    valid_metrics_ft = evaluate_and_save(model_finetune, dataloader_valid, device, mean, std,
                                         property_names, 'Validation', df, valid_idx, run_name, fold_number, 'finetune')
    all_metrics['finetune']['train'].append(train_metrics_ft)
    all_metrics['finetune']['valid'].append(valid_metrics_ft)

    # ---------------------------
    # B) From-Scratch Model
    # ---------------------------
    model_scratch = PeriodicNetwork(
        in_dim=in_dim,
        em_dim=em_dim,
        device=device,
        feature_dim=feature_dim,
        num_properties=num_properties,
        irreps_in=f"{em_dim}x0e",
        irreps_out=f"{out_dim}x0e",
        irreps_node_attr=f"{em_dim}x0e",
        layers=2,
        mul=32,
        lmax=2,
        max_radius=8.0,
        num_neighbors=20,
        reduce_output=True,
        out_dim=out_dim
    ).to(device)

    opt_sc = torch.optim.AdamW(model_scratch.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler_sc = torch.optim.lr_scheduler.ExponentialLR(opt_sc, gamma=0.95)

    run_name_fold_scratch = f"{run_name}_fold{fold_number}_scratch"
    print(f"--- Training SCRATCH model for fold {fold_number} ---")
    history_sc = train(model=model_scratch,
                       optimizer=opt_sc,
                       dataloader_train=dataloader_train,
                       dataloader_valid=dataloader_valid,
                       loss_fn=loss_fn,
                       loss_fn_mae=loss_fn_mae,
                       run_name=run_name_fold_scratch,
                       max_iter=1000,
                       scheduler=scheduler_sc,
                       device=device,
                       patience=40)
    torch.save({
        'model_state': model_scratch.state_dict(),
        'optimizer_state': opt_sc.state_dict(),
        'scheduler_state': scheduler_sc.state_dict(),
        'history': history_sc
    }, f"{run_name_fold_scratch}_model.pth")
    pd.DataFrame(history_sc).to_csv(f"{run_name_fold_scratch}_history.csv", index=False)

    train_metrics_sc = evaluate_and_save(model_scratch, dataloader_train, device, mean, std,
                                         property_names, 'Train', df, train_idx, run_name, fold_number, 'scratch')
    valid_metrics_sc = evaluate_and_save(model_scratch, dataloader_valid, device, mean, std,
                                         property_names, 'Validation', df, valid_idx, run_name, fold_number, 'scratch')
    all_metrics['scratch']['train'].append(train_metrics_sc)
    all_metrics['scratch']['valid'].append(valid_metrics_sc)

# ---------------------------------------------
# 5. Summarize Performance Across Folds
# ---------------------------------------------
summary = {'finetune': {}, 'scratch': {}}
for prop_name in property_names:
    # Finetune metrics
    train_maes_ft = [fold_metrics[prop_name]['MAE'] for fold_metrics in all_metrics['finetune']['train']]
    train_r2s_ft  = [fold_metrics[prop_name]['R2'] for fold_metrics in all_metrics['finetune']['train']]
    valid_maes_ft = [fold_metrics[prop_name]['MAE'] for fold_metrics in all_metrics['finetune']['valid']]
    valid_r2s_ft  = [fold_metrics[prop_name]['R2'] for fold_metrics in all_metrics['finetune']['valid']]
    # Scratch metrics
    train_maes_sc = [fold_metrics[prop_name]['MAE'] for fold_metrics in all_metrics['scratch']['train']]
    train_r2s_sc  = [fold_metrics[prop_name]['R2'] for fold_metrics in all_metrics['scratch']['train']]
    valid_maes_sc = [fold_metrics[prop_name]['MAE'] for fold_metrics in all_metrics['scratch']['valid']]
    valid_r2s_sc  = [fold_metrics[prop_name]['R2'] for fold_metrics in all_metrics['scratch']['valid']]

    summary['finetune'][prop_name] = {
        'train_MAE_mean': np.mean(train_maes_ft),
        'train_MAE_std': np.std(train_maes_ft),
        'train_R2_mean': np.mean(train_r2s_ft),
        'train_R2_std': np.std(train_r2s_ft),
        'valid_MAE_mean': np.mean(valid_maes_ft),
        'valid_MAE_std': np.std(valid_maes_ft),
        'valid_R2_mean': np.mean(valid_r2s_ft),
        'valid_R2_std': np.std(valid_r2s_ft),
    }
    summary['scratch'][prop_name] = {
        'train_MAE_mean': np.mean(train_maes_sc),
        'train_MAE_std': np.std(train_maes_sc),
        'train_R2_mean': np.mean(train_r2s_sc),
        'train_R2_std': np.std(train_r2s_sc),
        'valid_MAE_mean': np.mean(valid_maes_sc),
        'valid_MAE_std': np.std(valid_maes_sc),
        'valid_R2_mean': np.mean(valid_r2s_sc),
        'valid_R2_std': np.std(valid_r2s_sc),
    }

print("\n=== Cross-Validation Summary ===\n")
for model_tag in ['finetune', 'scratch']:
    print(f"== {model_tag.upper()} MODEL RESULTS ==")
    for prop_name in property_names:
        print(f"Property: {prop_name}")
        for k, v in summary[model_tag][prop_name].items():
            print(f"  {k}: {v}")
        print("")
    print("")

# End of finetune_5fold_compare.py
