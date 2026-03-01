"""
api.py
-------
FastAPI server wrapping the SchNet IR predictor.

Requirements:
    pip install fastapi uvicorn torch torch-geometric rdkit-pypi numpy

Usage:
    # Start the server (make sure best_model.pt is in the same folder):
    uvicorn api:app --reload --port 8000

    # Predict via browser:
    http://localhost:8000/docs        ← interactive Swagger UI

    # Predict via curl:
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"smiles": "CCO"}'

    # Predict via Python requests:
    import requests
    r = requests.post("http://localhost:8000/predict", json={"smiles": "CCO"})
    print(r.json())
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_add_pool
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import HybridizationType
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import io
from fastapi.responses import StreamingResponse, HTMLResponse

# ── Copy of model constants (must match schnet_ir.py) ─────────────────────────

N_POINTS     = 250
CUTOFF       = 5.0
HIDDEN_DIM   = 128
NUM_FILTERS  = 128
NUM_INTER    = 3
DROPOUT      = 0.001
ATOM_TYPES   = ['H','C','N','O','F','S','Cl','Br','I','P','Si','B','Se','other']
ATOM_FEAT_DIM = len(ATOM_TYPES) + 3 + 6   # 23
WAVENUMBERS  = np.linspace(400, 4000, N_POINTS).tolist()
CHECKPOINT   = "best_model.pt"
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Model definition (same as schnet_ir.py) ───────────────────────────────────

def atom_features(atom):
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
    return one_hot + hyb_enc + [
        float(atom.GetAtomicNum()) / 100.0,
        float(atom.GetTotalDegree()) / 6.0,
        float(atom.GetFormalCharge()) / 4.0,
        float(atom.GetTotalNumHs()) / 4.0,
        float(atom.IsInRing()),
        float(atom.GetIsAromatic()),
    ]

def smiles_to_data(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
    if result != 0:
        return None
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    node_feats, pos = [], []
    for atom in mol.GetAtoms():
        node_feats.append(atom_features(atom))
        p = conf.GetAtomPosition(atom.GetIdx())
        pos.append([p.x, p.y, p.z])
    return Data(
        x   = torch.tensor(node_feats, dtype=torch.float),
        pos = torch.tensor(pos, dtype=torch.float),
    )

class GaussianSmearing(nn.Module):
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
        W   = self.filter_net(edge_attr)
        x_j = self.lin1(x)[col]
        msg = x_j * W
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
    def __init__(self, atom_feat_dim=ATOM_FEAT_DIM, hidden_dim=HIDDEN_DIM,
                 num_filters=NUM_FILTERS, num_interactions=NUM_INTER,
                 num_gaussians=50, cutoff=CUTOFF, output_dim=N_POINTS):
        super().__init__()
        self.cutoff = cutoff
        self.embedding = nn.Linear(atom_feat_dim, hidden_dim)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.interactions = nn.ModuleList([
            InteractionBlock(hidden_dim, num_filters, num_gaussians)
            for _ in range(num_interactions)
        ])
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            ShiftedSoftplus(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim // 2),
            ShiftedSoftplus(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid(),
        )
    def _radius_graph(self, pos, batch):
        row, col = [], []
        for b in batch.unique():
            mask = (batch == b)
            idx = mask.nonzero(as_tuple=True)[0]
            p = pos[idx]
            diff = p.unsqueeze(0) - p.unsqueeze(1)
            dist = diff.norm(dim=-1)
            src, dst = ((dist < self.cutoff) & (dist > 0)).nonzero(as_tuple=True)
            row.append(idx[src])
            col.append(idx[dst])
        return torch.stack([torch.cat(row), torch.cat(col)], dim=0)

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch
        edge_index = self._radius_graph(pos, batch)
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(dist)
        h = self.embedding(x)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_attr)
        h_graph = global_add_pool(h, batch)
        return self.output_mlp(h_graph)

# ── Load model once at startup ─────────────────────────────────────────────────

print(f"Loading model from '{CHECKPOINT}' on {DEVICE}...")
model = SchNet().to(DEVICE)

# Load checkpoint safely (ignore mismatched layers)
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
model_dict = model.state_dict()

# Keep only compatible layers
filtered_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
model_dict.update(filtered_dict)
model.load_state_dict(model_dict)

model.eval()
print("Model ready. Compatible weights loaded, mismatched layers were skipped.")

# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="IR Spectrum Predictor",
    description="Predict IR transmittance spectra from SMILES strings using a SchNet GNN.",
    version="1.0.0",
)

# Allow requests from any frontend (useful if you build a UI later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response schemas ─────────────────────────────────────────────────

from pydantic import BaseModel, ConfigDict
from typing import List

class PredictRequest(BaseModel):
    smiles: str
    model_config = ConfigDict(
        json_schema_extra = {"example": {"smiles": "CCO"}}
    )

class PredictResponse(BaseModel):
    smiles:        str
    wavenumbers:   List[float]
    transmittance: List[float]

class BatchPredictRequest(BaseModel):
    smiles_list: List[str]
    model_config = ConfigDict(
        json_schema_extra = {"example": {"smiles_list": ["CCO", "CC(C)=O", "c1ccccc1"]}}
    )

class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    failed:  List[str]

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "message": "IR Spectrum Predictor is running."}


@app.post("/predict", response_model=PredictResponse, summary="Predict IR spectrum for one molecule")
def predict(req: PredictRequest):
    """
    Send a SMILES string, get back 250 wavenumber + transmittance pairs.
    """
    data = smiles_to_data(req.smiles)
    if data is None:
        raise HTTPException(
            status_code=422,
            detail=f"Could not generate 3D structure for SMILES: '{req.smiles}'. "
                   f"Check that it's a valid SMILES string."
        )

    batch = Batch.from_data_list([data]).to(DEVICE)
    with torch.no_grad():
        pred = model(batch).squeeze(0).cpu().numpy().tolist()

    return PredictResponse(
        smiles        = req.smiles,
        wavenumbers   = WAVENUMBERS,
        transmittance = pred,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, summary="Predict IR spectra for multiple molecules")
def predict_batch(req: BatchPredictRequest):
    """
    Send a list of SMILES strings, get back predictions for all valid ones.
    Invalid SMILES are returned in the 'failed' list.
    """
    results = []
    failed  = []

    for smiles in req.smiles_list:
        data = smiles_to_data(smiles)
        if data is None:
            failed.append(smiles)
            continue
        batch = Batch.from_data_list([data]).to(DEVICE)
        with torch.no_grad():
            pred = model(batch).squeeze(0).cpu().numpy().tolist()
        results.append(PredictResponse(
            smiles        = smiles,
            wavenumbers   = WAVENUMBERS,
            transmittance = pred,
        ))

    return BatchPredictResponse(results=results, failed=failed)


# ── Plot endpoint ─────────────────────────────────────────────────────────────

@app.post("/plot", summary="Predict IR spectrum and return a PNG plot")
def plot(req: PredictRequest):
    """
    Send a SMILES string, get back a PNG image of the predicted IR spectrum.
    Great for embedding in a frontend or viewing directly in the browser.
    """
    data = smiles_to_data(req.smiles)
    if data is None:
        raise HTTPException(
            status_code=422,
            detail=f"Could not generate 3D structure for SMILES: '{req.smiles}'."
        )

    batch = Batch.from_data_list([data]).to(DEVICE)
    with torch.no_grad():
        pred = model(batch).squeeze(0).cpu().numpy()

    # Build the plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(WAVENUMBERS, pred, color='steelblue', linewidth=1.5)
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Transmittance', fontsize=12)
    ax.set_title(f'Predicted IR Spectrum — {req.smiles}', fontsize=13)
    ax.invert_xaxis()   # conventional IR plot direction
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Stream the PNG directly as a response
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/plot", summary="Predict IR spectrum via GET (paste SMILES in URL)")
def plot_get(smiles: str):
    """
    Convenient GET version — use directly in a browser or <img> tag:
    http://localhost:8000/plot?smiles=CCO
    """
    return plot(PredictRequest(smiles=smiles))


# ── UI ────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>IR Spectrum Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #0a0e14;
    --surface:  #111720;
    --border:   #1e2d40;
    --accent:   #3b9eff;
    --accent2:  #00e5c0;
    --text:     #e2eaf4;
    --muted:    #5a7393;
    --danger:   #ff5f6d;
    --success:  #00e5c0;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 48px 24px 80px;
  }

  /* Subtle grid background */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(var(--border) 1px, transparent 1px),
      linear-gradient(90deg, var(--border) 1px, transparent 1px);
    background-size: 48px 48px;
    opacity: 0.35;
    pointer-events: none;
    z-index: 0;
  }

  .wrap {
    position: relative;
    z-index: 1;
    width: 100%;
    max-width: 780px;
  }

  /* Header */
  header {
    margin-bottom: 52px;
    text-align: center;
  }

  .eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 14px;
  }

  h1 {
    font-size: clamp(28px, 5vw, 42px);
    font-weight: 300;
    letter-spacing: -0.02em;
    line-height: 1.15;
    color: var(--text);
  }

  h1 span {
    color: var(--accent2);
    font-style: italic;
  }

  .subtitle {
    margin-top: 12px;
    color: var(--muted);
    font-size: 14px;
    letter-spacing: 0.01em;
  }

  /* Card */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 32px;
    margin-bottom: 24px;
  }

  .card-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
  }

  /* Input row */
  .input-row {
    display: flex;
    gap: 10px;
  }

  input[type=text] {
    flex: 1;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'DM Mono', monospace;
    font-size: 14px;
    color: var(--text);
    outline: none;
    transition: border-color 0.2s;
  }

  input[type=text]:focus {
    border-color: var(--accent);
  }

  input[type=text]::placeholder { color: var(--muted); }

  button {
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: opacity 0.15s, transform 0.1s;
    white-space: nowrap;
  }

  button:hover { opacity: 0.88; }
  button:active { transform: scale(0.97); }
  button:disabled { opacity: 0.4; cursor: not-allowed; }

  /* Quick examples */
  .examples {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
  }

  .chip {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    padding: 5px 11px;
    border-radius: 20px;
    border: 1px solid var(--border);
    color: var(--muted);
    cursor: pointer;
    transition: border-color 0.15s, color 0.15s;
    background: none;
  }

  .chip:hover {
    border-color: var(--accent);
    color: var(--accent);
    opacity: 1;
  }

  /* Status */
  #status {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    min-height: 20px;
    color: var(--muted);
    margin-top: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 1s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.2; }
  }

  /* Result */
  #result {
    display: none;
  }

  .result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }

  .smiles-tag {
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    color: var(--accent2);
  }

  .download-btn {
    background: none;
    border: 1px solid var(--border);
    color: var(--muted);
    font-size: 12px;
    padding: 6px 14px;
  }

  .download-btn:hover {
    border-color: var(--accent2);
    color: var(--accent2);
    opacity: 1;
  }

  #spectrum-img {
    width: 100%;
    border-radius: 8px;
    border: 1px solid var(--border);
    display: block;
  }

  /* Error */
  .error {
    color: var(--danger);
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    padding: 12px 16px;
    border: 1px solid var(--danger);
    border-radius: 8px;
    background: rgba(255,95,109,0.06);
  }

  /* Footer */
  footer {
    margin-top: 48px;
    text-align: center;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 0.08em;
  }
</style>
</head>
<body>
<div class="wrap">

  <header>
    <p class="eyebrow">SchNet GNN · Irvine Hacks</p>
    <h1>IR Spectrum<br><span>Predictor</span></h1>
    <p class="subtitle">Enter a SMILES string to predict its infrared transmittance spectrum</p>
  </header>

  <!-- Input card -->
  <div class="card">
    <p class="card-label">SMILES Input</p>
    <div class="input-row">
      <input type="text" id="smiles-input" placeholder="e.g. CCO" autocomplete="off" autocorrect="off" spellcheck="false"/>
      <button id="predict-btn" onclick="predict()">Predict</button>
    </div>
    <div class="examples">
      <span class="card-label" style="align-self:center; margin:0; margin-right:4px;">Try:</span>
      <button class="chip" onclick="setSmiles('CCO')">ethanol · CCO</button>
      <button class="chip" onclick="setSmiles('CC(C)=O')">acetone · CC(C)=O</button>
      <button class="chip" onclick="setSmiles('c1ccccc1')">benzene · c1ccccc1</button>
      <button class="chip" onclick="setSmiles('CC(=O)O')">acetic acid · CC(=O)O</button>
      <button class="chip" onclick="setSmiles('ClC(Cl)Cl')">chloroform · ClC(Cl)Cl</button>
    </div>
    <div id="status"></div>
  </div>

  <!-- Result card -->
  <div class="card" id="result">
    <div class="result-header">
      <div>
        <p class="card-label">Predicted Spectrum</p>
        <p class="smiles-tag" id="result-smiles"></p>
      </div>
      <button class="chip download-btn" id="dl-btn">Download PNG</button>
    </div>
    <img id="spectrum-img" alt="Predicted IR Spectrum"/>
  </div>

  <footer>GNN-based IR prediction · trained on NIST spectral data</footer>
</div>

<script>
  const input = document.getElementById('smiles-input');
  const btn   = document.getElementById('predict-btn');
  const status = document.getElementById('status');
  const result = document.getElementById('result');
  const img    = document.getElementById('spectrum-img');
  const dlBtn  = document.getElementById('dl-btn');
  const resultSmiles = document.getElementById('result-smiles');

  input.addEventListener('keydown', e => { if (e.key === 'Enter') predict(); });

  function setSmiles(s) {
    input.value = s;
    input.focus();
  }

  async function predict() {
    const smiles = input.value.trim();
    if (!smiles) return;

    btn.disabled = true;
    result.style.display = 'none';
    status.innerHTML = '<div class="dot"></div> Generating 3D structure and predicting spectrum...';

    try {
      const res = await fetch('/plot?smiles=' + encodeURIComponent(smiles));
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
        status.innerHTML = '<span class="error">Error: ' + (err.detail || res.statusText) + '</span>';
        return;
      }

      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);

      img.src = url;
      resultSmiles.textContent = smiles;
      result.style.display = 'block';
      status.innerHTML = '<span style="color: var(--success)">✓ Done</span>';

      dlBtn.onclick = () => {
        const a = document.createElement('a');
        a.href = url;
        a.download = 'ir_' + smiles.replace(/[^a-zA-Z0-9]/g, '_') + '.png';
        a.click();
      };

    } catch(e) {
      status.innerHTML = '<span class="error">Network error — is the server running?</span>';
    } finally {
      btn.disabled = false;
    }
  }
</script>
</body>
</html>"""


# ── Run directly ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)