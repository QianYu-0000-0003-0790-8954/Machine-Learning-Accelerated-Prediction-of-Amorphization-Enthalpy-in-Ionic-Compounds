# main.py
import sys
import os
# Insert the repository root (one directory above the 'scripts' folder) into sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from e3nn import o3
from typing import Dict, Union

from ase import Atom
from ase.neighborlist import neighbor_list
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import itertools

# Import custom modules
from utils.utils_data import deload_data, train_valid_test_split
from utils.utils_model import PeriodicNetwork, train
from utils.utils_plot import plot_training_history

from sklearn import metrics  # Ensure to import metrics here

import re
path_to_finetune_csv = 'dataset/finetune_input.csv'

def parse_formula(formula):
    """
    Given a formula string like 'KNO2', return a set of the element symbols, e.g. {'K','N','O'}.
    This is a simple parser that looks for capital letters optionally followed by a lowercase letter.
    Numbers are ignored for the purpose of retrieving just the unique elements.
    """
    pattern = r'([A-Z][a-z]?)'
    elements = re.findall(pattern, formula)
    return set(elements)

def custom_train_valid_test_split(
    df,
    finetune_csv="dataset/finetune_input.csv",
    train_ratio=0.8,
    valid_ratio=0.1,
    test_ratio=0.1,
    random_seed=42
):
    """
    Split df into train/valid/test sets with custom logic:
      1. If df row's symbols match any system in finetune CSV, force it to training.
      2. Among the remaining, do random 80:10:10 split to keep overall ratio near 8:1:1.

    Returns: idx_train (list), idx_valid (list), idx_test (list)
    """
    # 1) Read and parse the "System" column from finetune_input.csv
    finetune_df = pd.read_csv(finetune_csv)
    systems = finetune_df['System'].dropna().unique().tolist()  # e.g. ['KNO2', 'SiO2', ...]

    # Convert each system into a set of symbols
    system_symbol_sets = [parse_formula(s) for s in systems]

    # 2) For each row in df, see if its species set matches any of the system_symbol_sets
    forced_train_idx = []
    leftover_idx = []

    for idx, row in df.iterrows():
        # row['species'] might be a list like ['K','N','O','O']
        data_symbols = set(row['species'])  # e.g. {'K', 'N', 'O'}

        # Check if it matches ANY system in system_symbol_sets
        put_in_train = False
        for system_set in system_symbol_sets:
            if data_symbols == system_set:
                put_in_train = True
                break

        if put_in_train:
            forced_train_idx.append(idx)
        else:
            leftover_idx.append(idx)

    forced_train_idx = list(forced_train_idx)
    leftover_idx = list(leftover_idx)

    # 3) Among leftover, we do random 80:10:10 split to achieve overall ratio close to 8:1:1
    total_size = len(df)
    desired_train_size = int(train_ratio * total_size)
    desired_valid_size = int(valid_ratio * total_size)
    desired_test_size = total_size - desired_train_size - desired_valid_size

    forced_count = len(forced_train_idx)

    if forced_count >= desired_train_size:
        print(f"Warning: forced train data ({forced_count}) exceeds or equals desired train size ({desired_train_size}).")
        idx_train = forced_train_idx
        leftover_idx = []
        idx_valid = []
        idx_test = []
        return idx_train, idx_valid, idx_test

    rng = np.random.default_rng(random_seed)
    rng.shuffle(leftover_idx)

    needed_in_train = desired_train_size - forced_count
    train_leftover = leftover_idx[:needed_in_train]
    leftover_idx = leftover_idx[needed_in_train:]

    # Split the leftover into valid and test sets
    valid_leftover = leftover_idx[:desired_valid_size]
    test_leftover = leftover_idx[desired_valid_size:desired_valid_size + desired_test_size]

    idx_train = forced_train_idx + train_leftover
    idx_valid = valid_leftover
    idx_test = test_leftover

    print(f"Total data: {len(df)}")
    print(f"Forced training data: {forced_count}")
    print(f"Overall training data: {len(idx_train)} (desired ~ {desired_train_size})")
    print(f"Validation data: {len(idx_valid)} (desired ~ {desired_valid_size})")
    print(f"Test data: {len(idx_test)} (desired ~ {desired_test_size})")

    return idx_train, idx_valid, idx_test

# Set default dtype and device
torch.set_default_dtype(torch.float64)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('Torch device:', device)

# Define properties to predict
property_names = ['Shear Modulus Voigt']

# Load data
df_train, species, mean, std = deload_data(
    "dataset/features.csv",
    property_names
)
run_name = 'model_' + time.strftime("%y%m%d", time.localtime())
print('Run name:', run_name)

# Encoding atom types
type_encoding = {Atom(Z).symbol: Z - 1 for Z in range(1, 119)}
type_onehot = torch.eye(len(type_encoding), dtype=torch.float64)  # Ensure float64

# Build data
def build_data(entry, type_encoding, type_onehot, r_max=5.0, mean=None, std=None):
    symbols = list(entry.structure.symbols)
    positions = torch.from_numpy(entry.structure.positions).double()  # float64
    lattice = torch.from_numpy(entry.structure.cell.array).double().unsqueeze(0)  # float64
    feature = torch.tensor(entry.feature, dtype=torch.float64)  # float64

    # Edge information
    edge_src, edge_dst, edge_shift = neighbor_list("ijS", a=entry.structure, cutoff=r_max, self_interaction=True)

    # Compute edge vectors considering periodic boundary conditions
    edge_batch = torch.zeros(positions.shape[0], dtype=torch.long)  # Assuming single structure
    edge_batch = edge_batch[edge_src]
    edge_shift = torch.tensor(edge_shift, dtype=torch.float64)  # float64
    lattice_expanded = lattice.repeat(edge_shift.size(0), 1, 1)  # [num_edges, 3, 3]
    edge_shift_vectors = torch.einsum('ni,nij->nj', edge_shift, lattice_expanded)

    edge_vec = positions[edge_dst] - positions[edge_src] + edge_shift_vectors

    # Prepare targets
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

r_max = 8.0

# Build data
print("Building training data...")
df_train['data'] = df_train.apply(
    lambda x: build_data(x, type_encoding, type_onehot, r_max, mean, std), axis=1
)

# Split the dataset into train, validation, and test sets
idx_train, idx_valid, idx_test = custom_train_valid_test_split(df_train, finetune_csv=path_to_finetune_csv)

# Create train, valid, and test loaders
batch_size = 16
dataloader_train = tg.loader.DataLoader(
    df_train.loc[idx_train, 'data'].tolist(),
    batch_size=batch_size,
    shuffle=True
)
dataloader_valid = tg.loader.DataLoader(
    df_train.loc[idx_valid, 'data'].tolist(),
    batch_size=batch_size,
    shuffle=True
)
dataloader_test = tg.loader.DataLoader(
    df_train.loc[idx_test, 'data'].tolist(),
    batch_size=batch_size,
    shuffle=True
)

# Define the model
in_dim = 118  # Number of atom types
em_dim = 64
feature_dim = len(df_train.iloc[0]['feature'])
num_properties = len(property_names)
out_dim = 64

model = PeriodicNetwork(
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
    max_radius=r_max,
    num_neighbors=20,
    reduce_output=True,
    out_dim=out_dim
).to(device)
print(model)

# **Test a Single Forward Pass Before Full Training**
print("Testing a single forward pass...")
for data in dataloader_train:
    data = data.to(device)
    output = model(data)
    print(f"Output shape: {output.shape}")
    print(f"Targets shape: {data.y.shape}")
    break  # Exit after first batch

# Define optimizer and loss function
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)
loss_fn = torch.nn.MSELoss()
loss_fn_mae = torch.nn.L1Loss()

# Train the model
history = train(
    model=model,
    optimizer=opt,
    dataloader_train=dataloader_train,
    dataloader_valid=dataloader_valid,
    loss_fn=loss_fn,
    loss_fn_mae=loss_fn_mae,
    run_name=run_name,
    max_iter=1000,
    scheduler=scheduler,
    device=device
)

# Visualize training history
plot_training_history(history, run_name)

# === Evaluate and Save Predictions on Training Set ===
print("Evaluating predictions on training set...")
model.eval()
trues_train, preds_train = [], []
with torch.no_grad():
    for data in tqdm(dataloader_train, desc="Evaluating Train Set"):
        data = data.to(device)
        output = model(data)
        data_y = data.y.view(data.num_graphs, -1)
        trues_train.append(data_y.cpu().numpy())
        preds_train.append(output.cpu().numpy())
trues_train = np.vstack(trues_train) * std + mean
preds_train = np.vstack(preds_train) * std + mean

# Assumes df_train contains an 'ids' column for identification
df_train_preds = pd.DataFrame({'id': df_train.loc[idx_train, 'ids'].values})
for i, prop_name in enumerate(property_names):
    df_train_preds[f'true_{prop_name}'] = trues_train[:, i]
    df_train_preds[f'pred_{prop_name}'] = preds_train[:, i]
df_train_preds.to_csv(f"{run_name}_train_predictions.csv", index=False)
print("Training set predictions saved.")

# === Evaluate and Save Predictions on Validation Set ===
print("Evaluating predictions on validation set...")
trues_valid, preds_valid = [], []
with torch.no_grad():
    for data in tqdm(dataloader_valid, desc="Evaluating Validation Set"):
        data = data.to(device)
        output = model(data)
        data_y = data.y.view(data.num_graphs, -1)
        trues_valid.append(data_y.cpu().numpy())
        preds_valid.append(output.cpu().numpy())
trues_valid = np.vstack(trues_valid) * std + mean
preds_valid = np.vstack(preds_valid) * std + mean

df_valid_preds = pd.DataFrame({'id': df_train.loc[idx_valid, 'ids'].values})
for i, prop_name in enumerate(property_names):
    df_valid_preds[f'true_{prop_name}'] = trues_valid[:, i]
    df_valid_preds[f'pred_{prop_name}'] = preds_valid[:, i]
df_valid_preds.to_csv(f"{run_name}_valid_predictions.csv", index=False)
print("Validation set predictions saved.")

# === Evaluate and Save Predictions on Test Set ===
print("Evaluating predictions on test set...")
trues_test, preds_test = [], []
with torch.no_grad():
    for data in tqdm(dataloader_test, desc="Evaluating Test Set"):
        data = data.to(device)
        output = model(data)
        data_y = data.y.view(data.num_graphs, -1)
        trues_test.append(data_y.cpu().numpy())
        preds_test.append(output.cpu().numpy())
trues_test = np.vstack(trues_test) * std + mean
preds_test = np.vstack(preds_test) * std + mean

df_test_preds = pd.DataFrame({'id': df_train.loc[idx_test, 'ids'].values})
for i, prop_name in enumerate(property_names):
    df_test_preds[f'true_{prop_name}'] = trues_test[:, i]
    df_test_preds[f'pred_{prop_name}'] = preds_test[:, i]
df_test_preds.to_csv(f"{run_name}_test_predictions.csv", index=False)
print("Test set predictions saved.")
