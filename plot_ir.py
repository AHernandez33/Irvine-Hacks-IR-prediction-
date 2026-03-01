"""
plot_ir.py
-----------
Plots a predicted IR spectrum CSV.

Usage:
    python plot_ir.py predicted_ir_CCO.csv
    python plot_ir.py predicted_ir_CCO.csv --compare ir_spectra.csv --name ethanol
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('predicted_csv', help='Predicted IR CSV from schnet_ir.py')
parser.add_argument('--compare', help='Training CSV to overlay real spectrum', default=None)
parser.add_argument('--name', help='Molecule name in training CSV to compare against', default=None)
args = parser.parse_args()

fig, ax = plt.subplots(figsize=(12, 5))

# Plot prediction
pred_df = pd.read_csv(args.predicted_csv)
ax.plot(pred_df['wavenumber_cm1'], pred_df['transmittance'],
        label='Predicted', color='steelblue', linewidth=1.5)

# Optionally overlay the real spectrum from training CSV
if args.compare and args.name:
    train_df = pd.read_csv(args.compare)
    row = train_df[train_df['Name of molecule'].str.lower() == args.name.lower()]
    if not row.empty:
        point_cols = [c for c in train_df.columns if c.startswith('point')]
        y_real = row[point_cols].values.flatten()
        x_real = np.linspace(400, 4000, len(y_real))
        ax.plot(x_real, y_real, label='Real (NIST)', color='tomato',
                linewidth=1.5, linestyle='--')
    else:
        print(f"Could not find '{args.name}' in {args.compare}")

ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
ax.set_ylabel('Transmittance', fontsize=12)
ax.set_title(f'IR Spectrum — {args.predicted_csv}', fontsize=13)
ax.invert_xaxis()   # IR spectra are conventionally plotted right-to-left
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3)

out = args.predicted_csv.replace('.csv', '.png')
plt.tight_layout()
plt.savefig(out, dpi=150)
print(f"Saved plot → {out}")
plt.show()