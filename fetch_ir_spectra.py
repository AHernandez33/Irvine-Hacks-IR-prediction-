"""
fetch_ir_spectra.py
--------------------
Uses nistchempy v1.0.5 to retrieve IR spectra for compounds,
parses the JCAMP-DX (jdx_text) format, interpolates to 250 points,
and saves to CSV.

Requirements:
    pip install nistchempy pandas numpy scipy
"""

import re
import time
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import nistchempy as nist

# ── Configuration ──────────────────────────────────────────────────────────────

LONG_FORMAT = True
N_POINTS = 250
WAVENUMBER_MIN = 400        # cm⁻¹
WAVENUMBER_MAX = 4000       # cm⁻¹
OUTPUT_FILE = "ir_spectra.csv"
DELAY = 1.5                 # seconds between requests

# ── Compound list: (name, SMILES, NIST_ID) ────────────────────────────────────

COMPOUNDS = [
    ("ethanol",        "CCO",          "C64175"),
    ("acetone",        "CC(C)=O",      "C67641"),
    ("benzene",        "c1ccccc1",     "C71432"),
    ("toluene",        "Cc1ccccc1",    "C108883"),
    ("chloroform",     "ClC(Cl)Cl",    "C67663"),
    ("acetic acid",    "CC(=O)O",      "C64197"),
    ("methanol",       "CO",           "C67561"),
    ("cyclohexane",    "C1CCCCC1",     "C110827"),
    ("diethyl ether",  "CCOCC",        "C60297"),
    ("acetonitrile",   "CC#N",         "C75058"),
]

# ── JCAMP-DX parser ────────────────────────────────────────────────────────────

def parse_jdx(jdx_text):
    """
    Parse a JCAMP-DX string and return (wavenumbers, y_values) as numpy arrays.
    Handles both XYDATA=(X++(Y..Y)) and XY pairs formats.
    """
    lines = jdx_text.splitlines()

    # Extract header values
    def get_header(key):
        for line in lines:
            if line.upper().startswith(f'##{key}='):
                return line.split('=', 1)[1].strip()
        return None

    xfactor = float(get_header('XFACTOR') or 1.0)
    yfactor = float(get_header('YFACTOR') or 1.0)
    firstx  = float(get_header('FIRSTX')  or 0.0)
    lastx   = float(get_header('LASTX')   or 0.0)
    npoints = int(get_header('NPOINTS')   or 0)
    xydata_type = get_header('XYDATA') or ''

    x_vals, y_vals = [], []

    if 'X++' in xydata_type.upper():
        # JCAMP XYDATA=(X++(Y..Y)) compressed format
        in_data = False
        for line in lines:
            if '##XYDATA=' in line.upper():
                in_data = True
                continue
            if in_data:
                if line.startswith('##'):
                    break
                line = line.strip()
                if not line:
                    continue
                # First token is X, rest are Y values
                tokens = re.split(r'[\s,]+', line)
                if not tokens:
                    continue
                try:
                    x = float(tokens[0]) * xfactor
                except ValueError:
                    continue
                for tok in tokens[1:]:
                    try:
                        y_vals.append(float(tok) * yfactor)
                        x_vals.append(x)
                        # X increments linearly — we'll fix spacing after
                    except ValueError:
                        pass
        # Rebuild evenly spaced X from firstx/lastx/npoints
        if npoints > 0 and len(y_vals) > 0:
            x_vals = np.linspace(firstx * xfactor, lastx * xfactor, len(y_vals))
        else:
            x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)

    else:
        # Try simple ##PEAK TABLE or plain XY pairs
        in_data = False
        for line in lines:
            if line.startswith('##PEAK') or line.startswith('##XYPOINTS'):
                in_data = True
                continue
            if in_data:
                if line.startswith('##'):
                    break
                parts = re.split(r'[\s,;]+', line.strip())
                if len(parts) >= 2:
                    try:
                        x_vals.append(float(parts[0]) * xfactor)
                        y_vals.append(float(parts[1]) * yfactor)
                    except ValueError:
                        pass
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)

    if len(x_vals) < 2:
        return None, None

    # Ensure ascending x
    if x_vals[0] > x_vals[-1]:
        x_vals = x_vals[::-1]
        y_vals = y_vals[::-1]

    return x_vals, y_vals


# ── Main fetch logic ───────────────────────────────────────────────────────────

def get_ir_spectrum(name, nist_id):
    try:
        compound = nist.get_compound(nist_id)
    except Exception as e:
        print(f"  [!] get_compound failed: {e}")
        return None, None

    try:
        compound.get_ir_spectra()
    except Exception as e:
        print(f"  [!] get_ir_spectra() failed: {e}")
        return None, None

    if not compound.ir_specs:
        print(f"  [!] No IR spectra for {name}")
        return None, None

    spec = compound.ir_specs[0]

    if not hasattr(spec, 'jdx_text') or not spec.jdx_text:
        print(f"  [!] No jdx_text on spectrum for {name}")
        return None, None

    x, y = parse_jdx(spec.jdx_text)
    if x is None:
        print(f"  [!] Failed to parse JDX for {name}")
        return None, None

    return x, y


def interpolate(x, y, n=N_POINTS, x_min=WAVENUMBER_MIN, x_max=WAVENUMBER_MAX):
    mask = (x >= x_min) & (x <= x_max)
    xc, yc = x[mask], y[mask]
    if len(xc) < 2:
        return None, None
    f = interp1d(xc, yc, kind="linear", bounds_error=False,
                 fill_value=(yc[0], yc[-1]))
    xn = np.linspace(x_min, x_max, n)
    return xn, f(xn)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    rows = []

    for name, smiles, nist_id in COMPOUNDS:
        print(f"Fetching: {name} (NIST ID: {nist_id})")
        x_raw, y_raw = get_ir_spectrum(name, nist_id)
        time.sleep(DELAY)

        if x_raw is None:
            print(f"  Skipping {name}.\n")
            continue

        xi, yi = interpolate(x_raw, y_raw)
        if xi is None:
            print(f"  Interpolation failed for {name}.\n")
            continue

        print(f"  OK — {len(xi)} points. X range: {x_raw.min():.0f}–{x_raw.max():.0f} cm⁻¹\n")

        if LONG_FORMAT:
            for wn, tr in zip(xi, yi):
                rows.append({
                    "compound_name":  name,
                    "smiles":         smiles,
                    "wavenumber_cm1": round(float(wn), 2),
                    "transmittance":  round(float(tr), 6),
                })
        else:
            row = {"compound_name": name, "smiles": smiles}
            for wn, tr in zip(xi, yi):
                row[f"wn_{wn:.1f}"] = round(float(tr), 6)
            rows.append(row)

    if not rows:
        print("No data collected.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} rows -> '{OUTPUT_FILE}'")
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()