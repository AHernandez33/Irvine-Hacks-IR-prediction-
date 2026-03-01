"""
schnet_ir.py
-------------
SchNet-based GNN to predict IR spectra (250 wavenumber points) from SMILES.

Architecture:
  SMILES → RDKit 3D conformer → atom/edge features → SchNet message passing
  → global pooling → MLP head → 250 transmittance values

Requirements:
    pip install torch torch-geometric rdkit-pypi pandas numpy scipy

Usage:
    # Train:
    python schnet_ir.py --mode train --csv ir_spectra.csv

    # Predict a single molecule:
    python schnet_ir.py --mode predict --smiles "CCO" --checkpoint best_model.pt
"""

import argparse
import math
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdchem import HybridizationType

# ── Constants ──────────────────────────────────────────────────────────────────

N_POINTS      = 250
CUTOFF        = 5.0      # Å — SchNet interaction cutoff radius
HIDDEN_DIM    = 64       # smaller model → less overfitting on ~130 molecules
NUM_FILTERS   = 64
NUM_INTER     = 2        # 2 interaction blocks is enough for small datasets
BATCH_SIZE    = 16
MAX_EPOCHS    = 300
LR            = 1e-3
WEIGHT_DECAY  = 1e-4     # L2 regularization
DROPOUT       = 0.15     # dropout rate throughout the model
PATIENCE      = 25       # early stopping patience
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Seed for reproducibility
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# Atom types we encode (one-hot)
ATOM_TYPES = ['H','C','N','O','F','S','Cl','Br','I','P','Si','B','Se','other']

# ── Feature extraction ─────────────────────────────────────────────────────────

def atom_features(atom):
    """Return a float tensor of atom features."""
    symbol = atom.GetSymbol()
    atom_type = ATOM_TYPES.index(symbol) if symbol in ATOM_TYPES else ATOM_TYPES.index('other')
    one_hot = [0.0] * len(ATOM_TYPES)
    one_hot[atom_type] = 1.0

    hyb = atom.GetHybridization()
    hyb_enc = [
        float(hyb == HybridizationType.SP),
        float(hyb == HybridizationType.SP2),
        float(hyb == HybridizationType.SP3),
    ]

    features = one_hot + hyb_enc + [
        float(atom.GetAtomicNum()) / 100.0,
        float(atom.GetTotalDegree()) / 6.0,
        float(atom.GetFormalCharge()) / 4.0,
        float(atom.GetTotalNumHs()) / 4.0,
        float(atom.IsInRing()),
        float(atom.GetIsAromatic()),
    ]
    return features


ATOM_FEAT_DIM = len(ATOM_TYPES) + 3 + 6  # = 23


def smiles_to_data(smiles, y=None):
    """
    Convert a SMILES string to a PyG Data object with 3D coordinates.
    Returns None if 3D embedding fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        # Fallback to distance geometry
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
    if result != 0:
        return None

    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()

    # Node features
    node_feats = []
    pos = []
    for atom in mol.GetAtoms():
        node_feats.append(atom_features(atom))
        p = conf.GetAtomPosition(atom.GetIdx())
        pos.append([p.x, p.y, p.z])

    x   = torch.tensor(node_feats, dtype=torch.float)
    pos = torch.tensor(pos, dtype=torch.float)

    data = Data(x=x, pos=pos)
    if y is not None:
        data.y = torch.tensor(y, dtype=torch.float).unsqueeze(0)  # (1, N_POINTS)
    data.smiles = smiles
    return data


# ── Dataset ────────────────────────────────────────────────────────────────────

class IRDataset(Dataset):
    """
    Expects a CSV in wide format:
        'Name of molecule', 'SMILES', point 1, point 2, ..., point 250
    One row per molecule, 250 transmittance columns.
    """

    def __init__(self, csv_path):
        super().__init__()
        df = pd.read_csv(csv_path)

        cols = list(df.columns)
        # Find name and SMILES columns flexibly
        name_col   = next((c for c in cols if 'name' in c.lower()), cols[0])
        smiles_col = next((c for c in cols if 'smile' in c.lower()), cols[1])

        # All remaining columns are the 250 transmittance point values
        point_cols = [c for c in cols if c not in (name_col, smiles_col)]
        if len(point_cols) != N_POINTS:
            raise ValueError(
                f"Expected {N_POINTS} point columns, found {len(point_cols)}. "
                f"Check your CSV — columns should be: name, smiles, p1, p2, ..., p250"
            )

        self.samples = []
        for _, row in df.iterrows():
            smiles = row[smiles_col]
            y = row[point_cols].values.astype(np.float32)
            data = smiles_to_data(smiles, y)
            if data is not None:
                self.samples.append(data)
            else:
                print(f"  [!] Skipping {row[name_col]} — 3D embedding failed")

        print(f"Dataset: {len(self.samples)} molecules loaded.")

    def len(self):
        return len(self.samples)

    def get(self, idx):
        return self.samples[idx]


# ── SchNet ─────────────────────────────────────────────────────────────────────

class GaussianSmearing(nn.Module):
    """Expands interatomic distances into a Gaussian basis."""
    def __init__(self, start=0.0, stop=CUTOFF, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer('offset', offset)
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1)) ** 2

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * dist.pow(2))


class ShiftedSoftplus(nn.Module):
    def forward(self, x):
        return F.softplus(x) - math.log(2.0)


class CFConv(nn.Module):
    """Continuous-filter convolution — the core SchNet operation."""
    def __init__(self, in_channels, out_channels, num_filters, num_gaussians):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, num_filters)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.filter_net = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters),
        )

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        W = self.filter_net(edge_attr)          # (E, num_filters)
        x_j = self.lin1(x)[col]                 # (E, num_filters)
        msg = x_j * W                            # element-wise
        # Aggregate messages
        agg = torch.zeros(x.size(0), W.size(-1), device=x.device)
        agg.scatter_add_(0, row.unsqueeze(-1).expand_as(msg), msg)
        return self.lin2(agg)


class InteractionBlock(nn.Module):
    def __init__(self, hidden_dim, num_filters, num_gaussians, dropout=DROPOUT):
        super().__init__()
        self.conv    = CFConv(hidden_dim, hidden_dim, num_filters, num_gaussians)
        self.act     = ShiftedSoftplus()
        self.out     = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        h = self.act(self.conv(x, edge_index, edge_attr))
        h = self.dropout(h)
        return self.out(h)


class SchNet(nn.Module):
    def __init__(
        self,
        atom_feat_dim  = ATOM_FEAT_DIM,
        hidden_dim     = HIDDEN_DIM,
        num_filters    = NUM_FILTERS,
        num_interactions = NUM_INTER,
        num_gaussians  = 50,
        cutoff         = CUTOFF,
        output_dim     = N_POINTS,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.embedding = nn.Linear(atom_feat_dim, hidden_dim)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = nn.ModuleList([
            InteractionBlock(hidden_dim, num_filters, num_gaussians)
            for _ in range(num_interactions)
        ])

        # Output MLP: graph-level pooling → spectrum
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            ShiftedSoftplus(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim // 2),
            ShiftedSoftplus(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid(),   # transmittance is in [0, 1]
        )

    def _radius_graph(self, pos, batch):
        """Pure PyTorch radius graph — no torch-cluster required."""
        row, col = [], []
        # Process each graph in the batch separately
        for b in batch.unique():
            mask = (batch == b)
            idx = mask.nonzero(as_tuple=True)[0]
            p = pos[idx]
            # Pairwise distances
            diff = p.unsqueeze(0) - p.unsqueeze(1)   # (N, N, 3)
            dist = diff.norm(dim=-1)                  # (N, N)
            # Find pairs within cutoff (excluding self-loops)
            src, dst = ((dist < self.cutoff) & (dist > 0)).nonzero(as_tuple=True)
            row.append(idx[src])
            col.append(idx[dst])
        return torch.stack([torch.cat(row), torch.cat(col)], dim=0)

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch

        # Build radius graph in pure PyTorch (no torch-cluster needed)
        edge_index = self._radius_graph(pos, batch)
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(dist)

        # Atom embedding
        h = self.embedding(x)

        # Message passing
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_attr)

        # Global pooling + prediction
        h_graph = global_add_pool(h, batch)
        return self.output_mlp(h_graph)


# ── Training ───────────────────────────────────────────────────────────────────

def train(csv_path, checkpoint_path='best_model.pt'):
    dataset = IRDataset(csv_path)
    if len(dataset) == 0:
        print("No valid molecules — check your CSV.")
        return

    # Train / val split (80/20)
    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE)

    model = SchNet().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    print(f"\nTraining on {DEVICE} | {n_train} train, {n_val} val molecules\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch)
            loss = F.mse_loss(pred, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs

        train_loss /= n_train

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                pred = model(batch)
                val_loss += F.mse_loss(pred, batch.y).item() * batch.num_graphs
        val_loss /= n_val

        scheduler.step(val_loss)

        # Also compute MAE for a human-readable metric
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                pred = model(batch)
                val_mae += F.l1_loss(pred, batch.y).item() * batch.num_graphs
        val_mae /= n_val

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | train MSE: {train_loss:.6f} | val MSE: {val_loss:.6f} | val MAE: {val_mae:.4f}")

        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\nBest val MSE: {best_val_loss:.6f}  →  saved to '{checkpoint_path}'")
    print("Tip: val MAE < 0.05 = good | < 0.02 = excellent (transmittance scale 0–1)")


# ── Inference ──────────────────────────────────────────────────────────────────

def predict(smiles, checkpoint_path='best_model.pt'):
    model = SchNet().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    data = smiles_to_data(smiles)
    if data is None:
        print(f"Could not generate 3D structure for: {smiles}")
        return None

    from torch_geometric.data import Batch
    batch = Batch.from_data_list([data]).to(DEVICE)

    with torch.no_grad():
        pred = model(batch).squeeze(0).cpu().numpy()

    wavenumbers = np.linspace(400, 4000, N_POINTS)
    out_df = pd.DataFrame({'wavenumber_cm1': wavenumbers, 'transmittance': pred})
    out_path = f"predicted_ir_{smiles[:20].replace('/', '_')}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Prediction saved to '{out_path}'")
    print(out_df.head())
    return pred


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SchNet IR spectrum predictor')
    parser.add_argument('--mode',       choices=['train', 'predict'], required=True)
    parser.add_argument('--csv',        default='ir_spectra.csv',  help='Training CSV path')
    parser.add_argument('--smiles',     default='CCO',             help='SMILES for prediction')
    parser.add_argument('--checkpoint', default='best_model.pt',   help='Model checkpoint path')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args.csv, args.checkpoint)
    elif args.mode == 'predict':
        predict(args.smiles, args.checkpoint)